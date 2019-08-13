# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for automatically registering network layers for K-FAC."""
import collections
import warnings
import enum
import tensorflow as tf

from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import resource_variable_ops
from kfac.python.ops import utils
from kfac.python.ops.tensormatch import graph_matcher as gm
from kfac.python.ops.tensormatch import graph_patterns as gp
from kfac.python.ops.tensormatch import tensorflow_graph_util as graph_utils


class RecordType(enum.Enum):
  fully_connected = 1
  conv2d = 2


class AmbiguousRegistrationError(Exception):
  pass


class MatchRecord(object):
  """An object for storing data about graph pattern matches."""

  def __init__(self, record_type, params, tensor_set, data=None):
    """Construct a new `Record` object.

    Args:
      record_type: A `RecordType` representing the type of layer being recorded.
      params: A list of the variables used by this layer.
      tensor_set: A set of all tensors matched by the pattern. This is used
        for determining when one match is a subset of another.
      data: An optional dict for storing attributes specific to certain
        record types.
    """
    self.record_type = record_type
    self.params = params
    self.tensor_set = tensor_set
    if data is None:
      data = dict()
    self.data = data


def ensure_sequence(obj):
  """If `obj` isn't a tuple or list, return a tuple containing `obj`."""
  if isinstance(obj, (tuple, list)):
    return obj
  else:
    return (obj,)


def record_affine_from_bindings(bindings, consumed_tensors,
                                tensors_to_variables):
  """Construct a MatchRecord for the given Affine pattern bindings.

  Args:
    bindings: A dict representing a matched pattern. Strings representing
      components of the pattern are mapped to the matched Tensors.
    consumed_tensors: A set of all tensors consumed by the matched pattern.
      This should be a superset of the values of the bindings dict.
    tensors_to_variables: A dict mapping Tensors to the variables referencing
      them.

  Returns:
    A `MatchRecord` containing the information necessary to register the layer.

  Raises:
    ValueError: If the bindings contain biases but not weights.
  """
  if 'biases' in bindings:
    biases = tensors_to_variables.get(bindings['biases'])
  else:
    biases = None
  weights = tensors_to_variables.get(bindings['weights'], None)
  inputs = bindings['in']
  outputs = bindings['pre_activations']
  linear_op = bindings['linear_op']

  if biases is not None and weights is None:
    raise ValueError("Can't register linear layer part with only biases.")

  if weights is not None and biases is not None:
    params = (weights, biases)
  else:
    params = weights

  if params is not None:
    record_data = dict(inputs=inputs, outputs=outputs)

    if linear_op.type == 'MatMul':
      record_type = RecordType.fully_connected

    elif linear_op.type == 'Conv2D':
      record_type = RecordType.conv2d
      strides = tuple(map(int, linear_op.get_attr('strides')))
      padding = linear_op.get_attr('padding')
      data_format = linear_op.get_attr('data_format')
      # In Python 3 this might be class "bytes" so we convert to string.
      if not isinstance(padding, str):
        padding = padding.decode()
      if not isinstance(data_format, str):
        data_format = data_format.decode()
      record_data['strides'] = strides
      record_data['padding'] = padding
      record_data['data_format'] = data_format

    else:
      raise ValueError("Can't register operation: {}".format(linear_op))

    return MatchRecord(
        record_type=record_type,
        params=params,
        tensor_set=consumed_tensors,
        data=record_data)


def register_layers(layer_collection, varlist, batch_size=None):
  """Walk the graph and register all layers to layer_collection.

  Parameters used multiple times in the graph need to be handled differently
  depending on context: this could either mean the parameters represent an
  RNN layer, or that the graph has been replicated as multiple "towers"
  to allow data parallelism.
  We differentiate these cases by examining the loss functions registered by
  layer_collection: if losses have been registered multiple times with
  reuse=True, we separate the subgraphs corresponding to each tower and
  register layers independently for each with reuse=True.

  Args:
    layer_collection: A `LayerCollection` to use for registering layers.
    varlist: A list of the variables in the graph.
    batch_size: A `int` representing the batch size. Needs to specified if
      registering generic variables that don't match any layer patterns or
      if time/uses is folded. If the time/uses dimension is merged with
      batch then this is used to infer number of uses/time-steps.

  Returns:
    A `dict` of the entries registered to layer_collection.fisher_blocks.

  Raises:
    ValueError: If not all losses were registered the same number of times.
      If any variables specified as part of linked groups were not
      matched with their group.
      If the same variable is used in multiple layers types
      (e.g. fully connected and 2d convolution), or if the same variable is
      used in multiple layers of a type that doesn't support shared parameters.
    AmbiguousRegistrationError: If any variables must be registered as generic
      and batch_size is not specified, or if even after filtering, there are
      matches with overlapping but unequal sets of variables (see
      filter_records).
  """
  original_fisher_blocks = layer_collection.fisher_blocks.copy()
  user_registered_variables = set()
  for params in layer_collection.fisher_blocks.keys():
    for variable in ensure_sequence(params):
      user_registered_variables.add(variable)
  user_registered_variables = frozenset(user_registered_variables)

  if not layer_collection.losses:
    register_subgraph_layers(
        layer_collection,
        varlist,
        user_registered_variables=user_registered_variables,
        batch_size=batch_size)
  else:
    inputs_by_loss = tuple(tuple(loss.inputs for loss in loss_list)
                           for loss_list in layer_collection.towers_by_loss)

    num_towers = len(inputs_by_loss[0])

    if not all(
        (len(input_tensors) == num_towers for input_tensors in inputs_by_loss)):
      raise ValueError(
          'If losses are registered with reuse=True, each name must be '
          'registered the same number of times.')

    for tower_number, tower_input_tensors in enumerate(zip(*inputs_by_loss)):
      reuse = (tower_number > 0)
      with tf.variable_scope('tower_%d' % tower_number, reuse=reuse):
        subgraph = utils.SubGraph(tower_input_tensors)
        register_subgraph_layers(
            layer_collection,
            varlist,
            user_registered_variables=user_registered_variables,
            reuse=reuse,
            batch_size=batch_size,
            subgraph=subgraph)

  fisher_blocks = layer_collection.fisher_blocks
  return {
      params: fisher_blocks[params]
      for params in set(fisher_blocks) - set(original_fisher_blocks)
  }


def register_subgraph_layers(layer_collection,
                             varlist,
                             user_registered_variables=frozenset(),
                             reuse=False,
                             batch_size=None,
                             subgraph=None):
  """Walk a subgraph and register all layers to layer_collection.

  Args:
    layer_collection: A `LayerCollection` to use for registering layers.
    varlist: A list of the variables in the graph.
    user_registered_variables: A set of all the variables the user has manually
      registered. No layers using any of these variables should be registered.
    reuse: (OPTIONAL) bool. If True, then `layer_collection`
      selects a previously registered block with the same key as the key
      derived from `params` of that block. If False, a new block is
      registered.
    batch_size: A `int` representing the batch size. Needs to specified if
      registering generic variables that don't match any layer patterns or
      if the time/uses dimension is folded into batch. If the time/uses
      dimension is merged with batch then this is used to infer number of
      uses/time-steps.
    subgraph: The `SubGraph` to search. Defaults to
      `layer_collections.subgraph`; if this is None, searches the entire graph.

  Raises:
    ValueError: If any variables specified as part of linked groups were not
      matched with their group.
      If the same variable is used in multiple layers types
      (e.g. fully connected and 2d convolution), or if the same variable is
      used in multiple layers of a type that doesn't support shared parameters.
    AmbiguousRegistrationError: If any variables must be registered as generic
      and batch_size is not specified, or if even after filtering, there are
      matches with overlapping but unequal sets of variables (see
      filter_records).
  """

  # List of patterns and binding functions to use when we match one of them
  match_register_list = [(gm.matcher_with_consumed(gp.Affine),
                          record_affine_from_bindings)]

  # Patterns return bindings to raw tensors, so we need to be able to map back
  # to variables from the tensors those variables reference.
  def var_to_tensor(var):
    if resource_variable_ops.is_resource_variable(var):
      return var.handle
    if utils.is_reference_variable(var):
      return tf_ops.internal_convert_to_tensor(var, as_ref=True)
    raise ValueError('%s is not a recognized variable type.' % str(var))

  tensors_to_variables = {var_to_tensor(var): var for var in varlist}

  # Get all the ops from the graph.
  ops = layer_collection.graph.get_operations()

  # Filter out tf.identity ops since otherwise the matcher generates spurious
  # matches.
  ops = (op for op in ops if not graph_utils.is_identity(op))

  # Extract out the output tensors from the ops
  tensors = (out for op in ops for out in op.outputs)

  # Filter the tensors to include only those in the subgraph.
  if subgraph is None:
    subgraph = layer_collection.subgraph
  if subgraph is not None:
    tensors = subgraph.filter_list(tensors)

  # Go through each tensor and try to match each pattern to it.
  record_list_dict = dict()
  for tensor in tensors:
    for match, recfunc in match_register_list:
      match_res = match(tensor)
      if match_res:
        bindings, consumed_tensors = match_res
        record = recfunc(bindings, consumed_tensors, tensors_to_variables)
        if record is not None:
          if record.params not in record_list_dict:
            record_list_dict[record.params] = []
          record_list_dict[record.params].append(record)

  # Filter out records violating any rules.
  record_list_dict = filter_records(layer_collection, record_list_dict,
                                    user_registered_variables)

  # Register the layers by going through the lists of records for each param.
  register_records(layer_collection, record_list_dict, reuse, batch_size)

  # Determine which variables were registered either by the user or
  # in the current call to register_subgraph_layers.
  automatically_registered_variables = {
      var
      for params in record_list_dict
      for var in ensure_sequence(params)
  }
  registered_variables = (
      automatically_registered_variables | user_registered_variables)

  # Register any remaining parameters generically.
  for variable in varlist:
    if variable not in registered_variables:
      for specified_grouping in layer_collection.linked_parameters:
        assert isinstance(specified_grouping, frozenset)
        if variable in specified_grouping and len(specified_grouping) > 1:
          raise ValueError(
              'Variable {} in linked group {} was not matched.'.format(
                  variable, specified_grouping))

      generic_bad_string = ('generic registrations may be a symptom that the '
                            'scanner is failing to auto-detect your model. '
                            'Generic uses a last-resort approximation, and '
                            'should never be used for common layer types that '
                            'K-FAC properly supports, such as convs or '
                            'fully-connected layers.')
      if batch_size is None:
        raise AmbiguousRegistrationError(
            ('Tried to register {} as generic without knowledge of batch_size. '
             'You can pass batch_size in to fix this error. But please note, '
             + generic_bad_string).format(variable))
      warnings.warn(('Registering {} as generic because graph scanner '
                     'couldn\'t match a pattern for it. This can sometimes '
                     'be caused by the variable not being present in the graph '
                     'terminating at the registered losses. You might need to '
                     'pass an explicit list of parameters to tell the system '
                     'what parameters are actually in your model. Note that '
                     + generic_bad_string).format(variable))
      layer_collection.register_generic(variable, batch_size, reuse=reuse)


def filter_user_registered_records(record_list_dict, user_registered_variables):
  """Remove any matches that contain a variable registered by the user."""
  record_list_dict = record_list_dict.copy()
  for params in list(record_list_dict.keys()):
    for variable in ensure_sequence(params):
      if variable in user_registered_variables:
        del record_list_dict[params]
        break
  return record_list_dict


def filter_grouped_variable_records(layer_collection, record_list_dict):
  """Remove any matches violating user specified parameter groupings."""
  record_list_dict = record_list_dict.copy()
  for params in list(record_list_dict.keys()):
    for specified_grouping in layer_collection.linked_parameters:
      param_set = set(ensure_sequence(params))
      assert isinstance(specified_grouping, frozenset)
      if (param_set.intersection(specified_grouping) and
          param_set != specified_grouping):
        del record_list_dict[params]
        break
  return record_list_dict


def filter_subgraph_records(record_list_dict):
  """Remove any matches that correspond to strict subgraphs of other matches."""

  # Flatten the records dict to compare records with different parameters.
  flat_record_list = [
      record for records in record_list_dict.values() for record in records
  ]

  # Compare all pairs of records that share any variables. We perform two
  # passes, first marking variables for deletion by adding them to a set and
  # then removing all marked variables, in order to avoid traversing
  # flat_record_list on every removal while still maintaining record order.
  records_by_variable = collections.defaultdict(list)
  for record in flat_record_list:
    for variable in ensure_sequence(record.params):
      records_by_variable[variable].append(record)
  records_to_remove = set()
  for record in flat_record_list:
    for variable in ensure_sequence(record.params):
      for other_record in records_by_variable[variable]:
        if record.tensor_set < other_record.tensor_set:
          records_to_remove.add(record)
  flat_record_list = [
      record for record in flat_record_list if record not in records_to_remove
  ]

  # Unflatten the records list.
  record_list_dict = collections.defaultdict(list)
  for record in flat_record_list:
    record_list_dict[record.params].append(record)
    assert record is not None
  return dict(record_list_dict)


def filter_records(layer_collection, record_list_dict,
                   user_registered_variables):
  """Filter out recorded matches based on a set of rules.

  A match should be filtered out if any of the following are true:
    1. It contains any variables already registered by the user.
    2. It violates the user specified variable groupings.
    3. It corresponds to a strict subgraph of another match not already filtered
       out by the above steps.

  Args:
    layer_collection: A `LayerCollection` to use for registering layers.
    record_list_dict: A dict mapping tuples of variables to lists of
      `MatchRecord`s representing all of the places those variables are used
      in the graph.
    user_registered_variables: A set of all the variables the user has manually
      registered. No layers using any of these variables should be registered.

  Returns:
    A copy of `record_list_dict` with the records violating rules filtered out.

  Raises:
    AmbiguousRegistrationError: If even after filtering, there are matches
      with overlapping but unequal sets of variables. In these cases, the user
      will need to either manually register layers that use these variables,
      or specify a preferred variable grouping.
  """
  record_list_dict = filter_user_registered_records(record_list_dict,
                                                    user_registered_variables)
  record_list_dict = filter_grouped_variable_records(layer_collection,
                                                     record_list_dict)
  record_list_dict = filter_subgraph_records(record_list_dict)

  # Look for any violation in the consistency of the remaining matches.
  recorded_params = dict()
  ambiguous_registration_errors = []
  for params in record_list_dict:
    for variable in ensure_sequence(params):
      if variable in recorded_params:
        ambiguous_registration_errors.append(
            'Variable {} was recorded in multiple groups: {} and {}.'.format(
                variable, params, recorded_params[variable]))
      else:
        recorded_params[variable] = params
  if ambiguous_registration_errors:
    raise AmbiguousRegistrationError('\n'.join(ambiguous_registration_errors))

  return record_list_dict


def register_records(layer_collection,
                     record_list_dict,
                     reuse=False,
                     batch_size=None):
  """Registers the given records to layer_collection.

  Args:
    layer_collection: A `LayerCollection` to use for registering layers.
    record_list_dict: A dict mapping tuples of variables to lists of
      `MatchRecord`s representing all of the places those variables are used
      in the graph.
    reuse: (OPTIONAL) bool. If True, then `layer_collection`
      selects a previously registered block with the same key as the key
      derived from `params` of that block. If False, a new block is
      registered.
    batch_size: A `int` representing the batch size. Needs to specified if
      registering generic variables that don't match any layer patterns or
      if time/uses is folded. If the time/uses dimension is merged with
      batch then this is used to infer number of uses/time-steps.

  Raises:
    ValueError: If record_list_dict contains multiple record types for a single
      set of variables, or if there are multiple records for a set of variables
      of a type that doesn't support shared parameters.
  """

  mixed_record_type_errors = []

  # TODO(b/69627702): Layers must be registered in a deterministic order, else
  # FisherFactors may end up with different variable names.
  params_list = sorted(record_list_dict.keys(), key=str)
  for params in params_list:
    record_list = record_list_dict[params]
    # We don't support mixed types for the same params and probably never
    # will.
    if not all(record_list[0].record_type == record.record_type
               for record in record_list):
      mixed_record_type_errors.append(
          'Detected variables {} with mixed record types: {}.'.format(
              params, record_list))
      continue

    record_type = record_list[0].record_type
    if batch_size:
      # If the batch/time dimension is merged in the input then need to set
      # `num_uses`.
      first_dim = record_list[0].data['inputs'].shape.as_list()[0]
      is_batch_time_folded = not (first_dim is None or first_dim == batch_size)
      if is_batch_time_folded:
        num_uses = first_dim // batch_size

    if record_type is RecordType.fully_connected:
      if len(record_list) > 1:
        inputs = tuple(record.data['inputs'] for record in record_list)
        outputs = tuple(record.data['outputs'] for record in record_list)
        layer_collection.register_fully_connected_multi(
            params, inputs, outputs, reuse=reuse)
      else:
        record = record_list[0]
        inputs = record.data['inputs']
        outputs = record.data['outputs']
        if batch_size and is_batch_time_folded:
          layer_collection.register_fully_connected_multi(
              params, inputs, outputs, num_uses=num_uses, reuse=reuse)
        else:
          layer_collection.register_fully_connected(
              params, inputs, outputs, reuse=reuse)

    elif record_type is RecordType.conv2d:
      if len(record_list) > 1:
        inputs = tuple(record.data['inputs'] for record in record_list)
        outputs = tuple(record.data['outputs'] for record in record_list)
        strides = record_list[0].data['strides']
        padding = record_list[0].data['padding']
        data_format = record_list[0].data['data_format']
        layer_collection.register_conv2d_multi(
            params,
            strides,
            padding,
            inputs,
            outputs,
            data_format=data_format,
            reuse=reuse)
      else:
        record = record_list[0]
        inputs = record.data['inputs']
        outputs = record.data['outputs']
        strides = record.data['strides']
        padding = record.data['padding']
        data_format = record.data['data_format']
        if batch_size and is_batch_time_folded:
          layer_collection.register_conv2d_multi(
              params,
              strides,
              padding,
              inputs,
              outputs,
              data_format=data_format,
              num_uses=num_uses,
              reuse=reuse)
        else:
          layer_collection.register_conv2d(params, strides, padding, inputs,
                                           outputs, data_format=data_format,
                                           reuse=reuse)
    else:
      assert False, 'Invalid record type {}'.format(record_type)

  if mixed_record_type_errors:
    raise ValueError('\n'.join(mixed_record_type_errors))
