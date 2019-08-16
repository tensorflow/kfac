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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest

# Method used for inverting matrices.
POSDEF_INV_METHOD = "cholesky"
POSDEF_EIG_METHOD = "self_adjoint"

_TF_REPLICATOR = None


def smart_assign(variable, value, assign_fn=tf.assign):
  """Calls assign_fn on variable and value in a cross-replica context.

  When this function is called in a per-replica context, it will enter a cross-
  replica context before calling assign_fn(variable, value). During training
  with a tf.distribute.Strategy, optimizer.minimize is always called in a per-
  replica context (e.g. via experimental_run for TPUStrategy). Since with this
  function we assign a synchronized Tensor to a MirroredVariable with assign_fn,
  we use a merge_call to enter a cross-replica context, then use
  distribution.extended.update to assign value to variable with assign_fn.

  When this function is called in a cross-replica context or outside of a
  tf.distribute.Strategy scope, smart_assign will use assign_fn as is.
  Operations that happen inside of a tf.distribute.Strategy scope
  are typically in a cross replica context, unless, for example, they happen in
  an experimental_run call or a call_for_each_replica call.  In a cross-replica
  context, tf.distribute.get_replica_context() returns None.

  Args:
    variable: TF Variable. A MirroredVariable when in a distribution strategy.
    value: TF Tensor. This function will throw an error if value is a PerReplica
      type, which means it is an unsynchronized Tensor. You must reduce it using
      all_sum or all_average before using this function.
    assign_fn: assign_fn(variable, value) -> tf.Operation. The function
      used to update variable with value, typically tf.assign, tf.assign_add,
      or tf.assign_sub.

  Returns:
    tf.Tensor that contains the result of assign_fn(variable, value) called in
    a cross-replica context.
  """
  if not (tf.distribute.has_strategy() and tf.distribute.get_replica_context()):
    return assign_fn(variable, value)

  def merge_fn(distribution, variable, value):
    return distribution.extended.update(variable, assign_fn, args=(value,))

  return tf.distribute.get_replica_context().merge_call(
      merge_fn, args=(variable, value))


def set_global_constants(posdef_inv_method=None, tf_replicator=None):
  """Sets various global constants used by the classes in this module."""
  global POSDEF_INV_METHOD
  global _TF_REPLICATOR

  if posdef_inv_method is not None:
    POSDEF_INV_METHOD = posdef_inv_method
  if tf_replicator is not None:
    _TF_REPLICATOR = tf_replicator


class SequenceDict(object):
  """A dict convenience wrapper that allows getting/setting with sequences."""

  def __init__(self, iterable=None):
    self._dict = dict(iterable or [])

  def __getitem__(self, key_or_keys):
    if isinstance(key_or_keys, (tuple, list)):
      return list(map(self.__getitem__, key_or_keys))
    else:
      return self._dict[key_or_keys]

  def __setitem__(self, key_or_keys, val_or_vals):
    if isinstance(key_or_keys, (tuple, list)):
      for key, value in zip(key_or_keys, val_or_vals):
        self[key] = value
    else:
      self._dict[key_or_keys] = val_or_vals

  def items(self):
    return list(self._dict.items())


def tensors_to_column(tensors):
  """Converts a tensor or list of tensors to a column vector.

  Args:
    tensors: A tensor or list of tensors.

  Returns:
    The tensors reshaped into vectors and stacked on top of each other.
  """
  if isinstance(tensors, (tuple, list)):
    return tf.concat(
        tuple(tf.reshape(tensor, [-1, 1]) for tensor in tensors), axis=0)
  else:
    return tf.reshape(tensors, [-1, 1])


def column_to_tensors(tensors_template, colvec):
  """Converts a column vector back to the shape of the given template.

  Args:
    tensors_template: A tensor or list of tensors.
    colvec: A 2d column vector with the same shape as the value of
        tensors_to_column(tensors_template).

  Returns:
    X, where X is tensor or list of tensors with the properties:
     1) tensors_to_column(X) = colvec
     2) X (or its elements) have the same shape as tensors_template (or its
        elements)
  """
  if isinstance(tensors_template, (tuple, list)):
    offset = 0
    tensors = []
    for tensor_template in tensors_template:
      sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
      tensor = tf.reshape(colvec[offset:(offset + sz)], tensor_template.shape)
      tensors.append(tensor)
      offset += sz

    tensors = tuple(tensors)
  else:
    tensors = tf.reshape(colvec, tensors_template.shape)

  return tensors


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def layer_params_to_mat2d(vector):
  """Converts a vector shaped like layer parameters to a 2D matrix.

  In particular, we reshape the weights/filter component of the vector to be
  2D, flattening all leading (input) dimensions. If there is a bias component,
  we concatenate it to the reshaped weights/filter component.

  Args:
    vector: A Tensor or pair of Tensors shaped like layer parameters.

  Returns:
    A 2D Tensor with the same coefficients and the same output dimension.
  """
  if isinstance(vector, (tuple, list)):
    w_part, b_part = vector
    w_part_reshaped = tf.reshape(w_part, [-1, w_part.shape.as_list()[-1]])
    return tf.concat((w_part_reshaped, tf.reshape(b_part, [1, -1])), axis=0)
  elif isinstance(vector, tf.IndexedSlices):
    return vector
  else:  # Tensor or Tensor-like.
    return tf.reshape(vector, [-1, vector.shape.as_list()[-1]])


def mat2d_to_layer_params(vector_template, mat2d):
  """Converts a canonical 2D matrix representation back to a vector.

  Args:
    vector_template: A Tensor or pair of Tensors shaped like layer parameters.
    mat2d: A 2D Tensor with the same shape as the value of
        layer_params_to_mat2d(vector_template).

  Returns:
    A Tensor or pair of Tensors with the same coefficients as mat2d and the same
        shape as vector_template.
  """
  if isinstance(vector_template, (tuple, list)):
    w_part, b_part = mat2d[:-1], mat2d[-1]
    return tf.reshape(w_part, vector_template[0].shape), b_part
  elif isinstance(vector_template, tf.IndexedSlices):
    if not isinstance(mat2d, tf.IndexedSlices):
      raise TypeError(
          "If vector_template is an IndexedSlices, so should mat2d.")
    return mat2d
  else:
    return tf.reshape(mat2d, vector_template.shape)


def posdef_inv(tensor, damping):
  """Computes the inverse of tensor + damping * identity."""
  identity = tf.eye(tensor.shape.as_list()[0], dtype=tensor.dtype)
  damping = tf.cast(damping, dtype=tensor.dtype)
  return posdef_inv_functions[POSDEF_INV_METHOD](tensor, identity, damping)


def posdef_inv_matrix_inverse(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) directly."""
  return tf.matrix_inverse(tensor + damping * identity)


def posdef_inv_cholesky(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with Cholesky."""
  chol = tf.linalg.cholesky(tensor + damping * identity)
  return tf.linalg.cholesky_solve(chol, identity)


def posdef_inv_eig(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with eigendecomposition."""
  eigenvalues, eigenvectors = tf.self_adjoint_eig(tensor + damping * identity)
  return tf.matmul(eigenvectors / eigenvalues, eigenvectors, transpose_b=True)


posdef_inv_functions = {
    "matrix_inverse": posdef_inv_matrix_inverse,
    "cholesky": posdef_inv_cholesky,
    "eig": posdef_inv_eig,
}


def posdef_eig(mat):
  """Computes the eigendecomposition of a positive semidefinite matrix."""
  return posdef_eig_functions[POSDEF_EIG_METHOD](mat)


def posdef_eig_svd(mat):
  """Computes the singular values and left singular vectors of a matrix."""
  evals, evecs, _ = tf.svd(mat)

  return evals, evecs


def posdef_eig_self_adjoint(mat):
  """Computes eigendecomposition using self_adjoint_eig."""
  evals, evecs = tf.self_adjoint_eig(mat)
  evals = tf.abs(evals)  # Should be equivalent to svd approach.

  return evals, evecs


posdef_eig_functions = {
    "self_adjoint": posdef_eig_self_adjoint,
    "svd": posdef_eig_svd,
}


def cholesky(tensor, damping):
  """Computes the inverse of tensor + damping * identity."""
  identity = tf.eye(tensor.shape.as_list()[0], dtype=tensor.dtype)
  damping = tf.cast(damping, dtype=tensor.dtype)
  return tf.linalg.cholesky(tensor + damping * identity)


class SubGraph(object):
  """Defines a subgraph given by all the dependencies of a given set of outputs.
  """

  def __init__(self, outputs):
    # Set of all ancestor Tensors, Ops to 'outputs'.
    self._members = set()

    self._iter_add(outputs)

  def _iter_add(self, root):
    """Iteratively adds all of nodes' ancestors using depth first search."""
    stack = [root]
    while stack:
      nodes = stack.pop()
      for node in nodes:
        if node in self._members:
          continue
        self._members.add(node)

        if isinstance(node, tf.Tensor):
          stack.append((node.op,))
        elif isinstance(node, tf.Operation):
          stack.append(node.inputs)

  def is_member(self, node):
    """Check if 'node' is in this subgraph."""
    return node in self._members

  def variable_uses(self, var):
    """Computes number of times a variable is used.

    Args:
      var: Variable or ResourceVariable instance.

    Returns:
      Number of times a variable is used within this subgraph.

    Raises:
      ValueError: If 'var' is not a variable type.
    """
    def _add_tensor_consumers_to_set(tensor, consumers_set):
      """Finds consumers of a tensor and add them to the current consumers set.
      """
      for consumer in set(tensor.consumers()):
        # These are the type of ops which relay a tensor to other ops without
        # doing anything to the tensor value, so recursively find the actual
        # consumers.
        if consumer.type in [
            "Identity", "ReadVariableOp", "Enter", "ResourceGather"]:
          for output in consumer.outputs:
            _add_tensor_consumers_to_set(output, consumers_set)
        else:
          consumers_set.add(consumer)

    if resource_variable_ops.is_resource_variable(var):
      var = var.handle
    elif is_reference_variable(var):
      var = var.value()
    else:
      raise ValueError("%s does not appear to be a variable." % str(var))

    consumers = set()
    _add_tensor_consumers_to_set(var, consumers)
    return len(self._members.intersection(consumers))

  def filter_list(self, node_list):
    """Filters 'node_list' to nodes in this subgraph."""
    filtered_list = []
    for node in node_list:
      if self.is_member(node):
        filtered_list.append(node)
    return filtered_list


def generate_random_signs(shape, dtype=tf.float32):
  """Generate a random tensor with {-1, +1} entries."""
  ints = tf.random_uniform(shape, maxval=2, dtype=tf.int32)
  return 2 * tf.cast(ints, dtype=dtype) - 1


# MirroredVariables do not have a hashable op property, which means they cannot
# be used with stop_gradients. This was fixed in the TF-Nightly release, but is
# not in any stable release, so we use the below hack so our fwd_gradients
# function works in the TF 1.14 stable release.
# TODO(b/139376871): Remove this workaround once the bugfix is in a stable release.
DistributedVarOp = collections.namedtuple(
    "DistributedVarOp", ["name", "graph", "traceback", "type"])

class MirroredVariableWrapper(object):

  def __init__(self, var):
    self.__var = var

  def __getattr__(self, name):
    if name == 'op':
      return DistributedVarOp(
          self.__var.op.name,
          self.__var.op.graph,
          # In the updated TF codebase, convert_stack returns tuple instead of
          # list, which makes op.traceback hashable.
          tuple(self.__var.op.traceback),
          self.__var.op.type)
    else:
      return getattr(self.__var, name)

def fwd_gradients(ys, xs, grad_xs=None, stop_gradients=None,
                  colocate_gradients_with_ops=True):
  """Compute forward-mode gradients."""
  # See b/37888268.

  # This version of forward-mode autodiff is based on code by Tim Cooijmans
  # and handles list arguments and certain special cases such as when the
  # ys doesn't depend on one or more of the xs, and when tf.IndexedSlices are
  # generated by the first tf.gradients call.

  us = [tf.zeros_like(y) + float("nan") for y in ys]
  if tf.distribute.has_strategy():
    stop_gradients = [MirroredVariableWrapper(v) for v in stop_gradients]
  dydxs = tf.gradients(ys, xs, grad_ys=us, stop_gradients=stop_gradients,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Deal with strange types that tf.gradients returns but can't
  # deal with.
  dydxs = [
      tf.convert_to_tensor(dydx) if isinstance(dydx, tf.IndexedSlices) else dydx
      for dydx in dydxs
  ]
  dydxs = [
      tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
  ]
  dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)
  return dysdx


def get_tf_replicator():
  return _TF_REPLICATOR


def is_tpu_replicated():
  is_tpu_strategy = (tf.distribute.has_strategy() and
                     tf.distribute.get_replica_context() and
                     isinstance(tf.distribute.get_strategy(),
                                tf.distribute.experimental.TPUStrategy))
  num_shards = tpu_function.get_tpu_context().number_of_shards
  return is_tpu_strategy or num_shards is not None


def is_replicated():
  """Check if we are operating in a supported replicated context."""
  if tf.distribute.has_strategy() and tf.distribute.get_replica_context():
    return tf.distribute.get_strategy().num_replicas_in_sync > 1
  return get_tf_replicator() is not None or is_tpu_replicated()


def get_num_replicas():
  """Returns the number of replicas.

  If not operating in a supported replicated context this function will return
  1.
  """

  tf_replicator = get_tf_replicator()

  if tf_replicator:
    return tf_replicator.num_replicas_in_sync
  elif tf.distribute.has_strategy():
    return tf.distribute.get_strategy().num_replicas_in_sync
  else:
    # I'm assuming replicas and shards are always equal until someone tells me
    # different.
    num_replicas = tpu_function.get_tpu_context().number_of_shards
    if num_replicas:
      return num_replicas
    else:
      return 1


def get_replica_id():
  """Returns an id number for the current replica, counting from 0.

  If not operating in a supported replicated context this function will return
  0.
  """

  tf_replicator = get_tf_replicator()

  if tf_replicator:
    return tf_replicator.current_replica_id
  elif tf.distribute.has_strategy() and tf.distribute.get_replica_context():
    return tf.distribute.get_replica_context().replica_id_in_sync_group

  # This code below this point is based on
  # TensorTracer._add_replica_id_to_graph().
  num_replicas = get_num_replicas()

  if num_replicas <= 1:
    return 0

  with tf.control_dependencies(None):
    # Uses None as dependency to run outside of TPU graph rewrites.
    return tpu_ops.tpu_replicated_input(list(range(num_replicas)),
                                        name="replica_id")


def all_sum(structure, name=None):
  """Sums the contents of a nested structure across all replicas.

  If not operating in a supported replicated context this function acts like
  the identity.

  Args:
    structure: A nested structure of Tensors.
    name: None or string. Optional name of Op. (Default: None)

  Returns:
    A nested structure with the corresponding Tensors being the cross-replica
    summed versions of those in `structure`.
  """
  num_replicas = get_num_replicas()

  if num_replicas <= 1:
    return structure

  tf_replicator = get_tf_replicator()
  if tf_replicator:
    return tf_replicator.all_sum(structure)

  elif tf.distribute.has_strategy() and tf.distribute.get_replica_context():
    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.SUM, structure)

  elif is_tpu_replicated():
    def tpu_all_sum(tensor):
      return tf.contrib.tpu.cross_replica_sum(tensor, name=name)

    return nest.map_structure(tpu_all_sum, structure)

  return structure


def all_average(structure, name=None):
  """Averages the contents of a nested structure across all replicas.

  If not operating in a supported replicated context this function acts like
  the identity.

  Args:
    structure: A nested structure of Tensors.
    name: None or string. Optional name of Op. (Default: None)

  Returns:
    A nested structure with the corresponding Tensors being the cross-replica
    averaged versions of those in `structure`.
  """
  num_replicas = get_num_replicas()

  if num_replicas <= 1:
    return structure

  if tf.distribute.has_strategy() and tf.distribute.get_replica_context():
    return tf.distribute.get_replica_context().all_reduce(
        tf.distribute.ReduceOp.MEAN, structure)

  return nest.map_structure(lambda x: x / num_replicas, all_sum(structure,
                                                                name=name))


def map_gather(thunks, name=None):
  """Distributes the execution of thunks over replicas, then gathers results.

    This method can be used to distribute several expensive computations across
    the replicas, rather than duplicating the computation in all of them.

  Args:
    thunks: A list of thunks that each returns a nested structure of Tensors.
      These should all have statically known shapes.
    name: None or string. Optional name of Op. (Default: None)

  Returns:
    A list of nested structures of Tensors representing the return values of
    the list of thunks.
  """

  num_replicas = get_num_replicas()

  if num_replicas <= 1:
    return tuple(thunk() for thunk in thunks)

  tf_replicator = get_tf_replicator()

  if tf_replicator:
    return tf_replicator.map_gather(thunks, lambda thunk: thunk())

  elif is_tpu_replicated():
    replica_id = get_replica_id()

    def zeros_like(tensor):
      return tf.zeros(dtype=tensor.dtype, shape=tensor.shape)

    results = []
    for idx, thunk in enumerate(thunks):
      # TensorFlow's optimization should eliminate the actual computations
      # done to compute example_structure, using only the (static) shape
      # information.
      def make_zeros_thunk(example_structure):
        def zeros_thunk():
          return nest.map_structure(zeros_like, example_structure)
        return zeros_thunk

      # This trick of using cross_replica_sum with tensors of zeros is
      # obviously wasteful in terms of commmunication. A better solution would
      # involve only communicating the tensors from replicas where `include_me`
      # was True.
      include_me = tf.equal(replica_id, idx % num_replicas)
      results.append(
          all_sum(tf.cond(include_me,
                          thunk,
                          make_zeros_thunk(thunk()),
                          strict=True),
                  name=name))

    return results

  return tuple(thunk() for thunk in thunks)


def ensure_sequence(obj):
  """If `obj` isn't a tuple or list, return a tuple containing `obj`."""
  if isinstance(obj, (tuple, list)):
    return obj
  else:
    return (obj,)


def batch_execute(global_step, thunks, batch_size, name=None):
  """Executes a subset of ops per global step.

  Given a list of thunks, each of which produces a single stateful op,
  ensures that exactly 'batch_size' ops are run per global step. Ops are
  scheduled in a round-robin fashion. For example, with 3 ops

    global_step | op0 | op1 | op2
    ------------+-----+-----+-----
        0       |  x  |  x  |
    ------------+-----+-----+-----
        1       |  x  |     |  x
    ------------+-----+-----+-----
        2       |     |  x  |  x
    ------------+-----+-----+-----
        3       |  x  |  x  |
    ------------+-----+-----+-----
        4       |  x  |     |  x

  Does not guarantee order of op execution within a single global step.

  Args:
    global_step: Tensor indicating time. Determines which ops run.
    thunks: List of thunks. Each thunk encapsulates one op. Return values are
      ignored.
    batch_size: int. Number of ops to execute per global_step.
    name: string or None. Name scope for newly added ops.

  Returns:
    List of ops. Exactly 'batch_size' ops are guaranteed to have an effect
    every global step.
  """

  def true_fn(thunk):
    """Ensures thunk is executed and returns an Op (not a Tensor)."""

    def result():
      with tf.control_dependencies([thunk()]):
        return tf.no_op()

    return result

  def false_fn(_):
    """Executes a no-op."""

    def result():
      return tf.no_op()

    return result

  with tf.name_scope(name, "batch_execute"):
    true_fns = [true_fn(thunk) for thunk in thunks]
    false_fns = [false_fn(thunk) for thunk in thunks]
    num_thunks = len(thunks)
    conditions = [
        tf.less(
            tf.mod(batch_size - 1 + global_step * batch_size - j, num_thunks),
            batch_size) for j in range(num_thunks)
    ]
    result = [
        tf.cond(condition, true_fn, false_fn)
        for (condition, true_fn,
             false_fn) in zip(conditions, true_fns, false_fns)
    ]
    return result


def extract_convolution_patches(inputs,
                                filter_shape,
                                padding,
                                strides=None,
                                dilation_rate=None,
                                name=None,
                                data_format=None):
  """Extracts inputs to each output coordinate in tf.nn.convolution.

  This is a generalization of tf.extract_image_patches() to tf.nn.convolution(),
  where the number of spatial dimensions may be something other than 2.

  Assumes,
  - First dimension of inputs is batch_size
  - Convolution filter is applied to all input channels.

  Args:
    inputs: Tensor of shape [batch_size, ..spatial_image_shape..,
      ..spatial_filter_shape.., in_channels]. Inputs to tf.nn.convolution().
    filter_shape: List of ints. Shape of filter passed to tf.nn.convolution().
    padding: string. Padding method. One of "VALID", "SAME".
    strides: None or list of ints. Strides along spatial dimensions.
    dilation_rate: None or list of ints. Dilation along spatial dimensions.
    name: None or str. Name of Op.
    data_format: None or str. Format of data.

  Returns:
    Tensor of shape [batch_size, ..spatial_image_shape..,
      ..spatial_filter_shape.., in_channels]

  Raises:
    ValueError: If data_format does not put channel last.
    ValueError: If inputs and filter disagree on in_channels.
  """
  if not is_data_format_channel_last(data_format):
    raise ValueError("Channel must be last dimension.")
  with tf.name_scope(name, "extract_convolution_patches",
                     [inputs, filter_shape, padding, strides, dilation_rate]):
    batch_size = inputs.shape.as_list()[0]
    in_channels = inputs.shape.as_list()[-1]

    # filter_shape = spatial_filter_shape + [in_channels, out_channels]
    spatial_filter_shape = filter_shape[:-2]
    if in_channels != filter_shape[-2]:
      raise ValueError("inputs and filter_shape must agree on in_channels.")

    # Map each input feature to a location in the output.
    out_channels = np.prod(spatial_filter_shape) * in_channels
    filters = tf.eye(out_channels, dtype=inputs.dtype)
    filters = tf.reshape(
        filters,
        list(spatial_filter_shape) + [in_channels, out_channels])

    if strides is not None and len(strides) == len(inputs.shape):
      strides = strides[1:-1]  # remove batch and channel dimension

    if dilation_rate is not None and len(dilation_rate) == len(inputs.shape):
      dilation_rate = dilation_rate[1:-1]  # remove batch and channel dimension

    result = tf.nn.convolution(
        inputs,
        filters,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate)
    spatial_output_shape = result.shape.as_list()[1:-1]
    result = tf.reshape(result, [batch_size or -1] + spatial_output_shape +
                        list(spatial_filter_shape) + [in_channels])

    return result


def extract_pointwise_conv2d_patches(inputs,
                                     filter_shape,
                                     name=None,
                                     data_format=None):
  """Extract patches for a 1x1 conv2d.

  Args:
    inputs: 4-D Tensor of shape [batch_size, height, width, in_channels].
    filter_shape: List of 4 ints. Shape of filter to apply with conv2d()
    name: None or str. Name for Op.
    data_format: None or str. Format for data. See 'data_format' in
      tf.nn.conv2d() for details.

  Returns:
    Tensor of shape [batch_size, ..spatial_input_shape..,
    ..spatial_filter_shape.., in_channels]

  Raises:
    ValueError: if inputs is not 4-D.
    ValueError: if filter_shape is not [1, 1, ?, ?]
    ValueError: if data_format is not channels-last.
  """
  if inputs.shape.ndims != 4:
    raise ValueError("inputs must have 4 dims.")
  if len(filter_shape) != 4:
    raise ValueError("filter_shape must have 4 dims.")
  if filter_shape[0] != 1 or filter_shape[1] != 1:
    raise ValueError("filter_shape must have shape 1 along spatial dimensions.")
  if not is_data_format_channel_last(data_format):
    raise ValueError("data_format must be channels last.")
  with tf.name_scope(name, "extract_pointwise_conv2d_patches",
                     [inputs, filter_shape]):
    ksizes = [1, 1, 1, 1]  # Spatial shape is 1x1.
    strides = [1, 1, 1, 1]  # Operate on all pixels.
    rates = [1, 1, 1, 1]  # Dilation has no meaning with spatial shape = 1.
    padding = "VALID"  # Doesn't matter.
    result = tf.extract_image_patches(inputs, ksizes, strides, rates, padding)

    batch_size, input_height, input_width, in_channels = inputs.shape.as_list()
    filter_height, filter_width, in_channels, _ = filter_shape
    return tf.reshape(result, [
        batch_size, input_height, input_width, filter_height, filter_width,
        in_channels
    ])


def is_data_format_channel_last(data_format):
  """True if data_format puts channel last."""
  if data_format is None:
    return True
  return data_format.endswith("C")


def matmul_sparse_dense(A, B, name=None, transpose_a=False, transpose_b=False):  # pylint: disable=invalid-name
  """Computes matmul(A, B) where A is sparse, B is dense.

  Args:
    A: tf.IndexedSlices with dense shape [m, n].
    B: tf.Tensor with shape [n, k].
    name: str. Name of op.
    transpose_a: Bool. If true we transpose A before multiplying it by B.
      (Default: False)
    transpose_b: Bool. If true we transpose B before multiplying it by A.
      (Default: False)

  Returns:
    tf.IndexedSlices resulting from matmul(A, B).

  Raises:
    ValueError: If A doesn't represent a matrix.
    ValueError: If B is not rank-2.
  """
  with tf.name_scope(name, "matmul_sparse_dense", [A, B]):
    if A.indices.shape.ndims != 1 or A.values.shape.ndims != 2:
      raise ValueError("A must represent a matrix. Found: %s." % A)
    if B.shape.ndims != 2:
      raise ValueError("B must be a matrix.")
    new_values = tf.matmul(
        A.values, B, transpose_a=transpose_a, transpose_b=transpose_b)
    return tf.IndexedSlices(
        new_values,
        A.indices,
        dense_shape=tf.stack([A.dense_shape[0], new_values.shape[1]]))


def matmul_diag_sparse(A_diag, B, name=None):  # pylint: disable=invalid-name
  """Computes matmul(A, B) where A is a diagonal matrix, B is sparse.

  Args:
    A_diag: diagonal entries of matrix A of shape [m, m].
    B: tf.IndexedSlices. Represents matrix of shape [m, n].
    name: str. Name of op.

  Returns:
    tf.IndexedSlices resulting from matmul(A, B).

  Raises:
    ValueError: If A_diag is not rank-1.
    ValueError: If B doesn't represent a matrix.
  """
  with tf.name_scope(name, "matmul_diag_sparse", [A_diag, B]):
    A_diag = tf.convert_to_tensor(A_diag)
    if A_diag.shape.ndims != 1:
      raise ValueError("A_diag must be a rank-1 Tensor.")
    if B.indices.shape.ndims != 1 or B.values.shape.ndims != 2:
      raise ValueError("B must represent a matrix. Found: %s." % B)
    a = tf.gather(A_diag, B.indices)
    a = tf.reshape(a, list(a.shape) + [1] * (B.values.shape.ndims - 1))
    return tf.IndexedSlices(a * B.values, B.indices, dense_shape=B.dense_shape)


class AccumulatorVariable(object):
  """A simple abstraction to accumulate data that we want to average.

  Basically this variable accumulates data across multiple inputs, and
  then returns the average of these contributes on command.  This accumulation
  can be reset by the user at any point.
  """

  def __init__(self, name, shape, dtype):
    """Constructs a new `AccumulatorVariable`.

    Args:
      name: `string`. Scope for the variables.
      shape: shape of the variable.
      dtype: dtype of the variable.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self._acc_var = tf.get_variable(
          "acc_var",
          shape=shape,
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          trainable=False,
          use_resource=True)

      # We may be able to make give this a VariableAggregation of
      # ONLY_FIRST_REPLICA, because we only add 1 or reset it to 0 (it does not
      # rely on per-replica values). If we do, we can update this in a per-
      # replica context instead of the cross-replica context. This may improve
      # efficiency when using a VariableSynchronization of ON_READ.
      self._counter = tf.get_variable(
          "counter",
          shape=(),
          dtype=tf.int32,
          initializer=tf.zeros_initializer(),
          trainable=False,
          use_resource=True)

  def accumulate(self, value):
    """Adds `value` to the accumulated data."""
    inc_counter_op = smart_assign(self._counter, 1, assign_fn=tf.assign_add)
    acc_op = smart_assign(self._acc_var, value, assign_fn=tf.assign_add)
    return tf.group(inc_counter_op, acc_op)

  @property
  def value(self):
    """Returns the average of the accumulated values since the last reset."""
    return self._acc_var / tf.cast(self._counter, self._acc_var.dtype)

  def read_value_and_reset(self):
    """Same as `value` property but resets after the data is read."""
    value = self.value
    with tf.control_dependencies([value]):
      with tf.control_dependencies([self.reset()]):
        return tf.identity(value)

  def reset(self):
    """Resets the accumulated data to zero."""
    var_reset_op = smart_assign(
        self._acc_var, tf.zeros(self._acc_var.shape, dtype=self._acc_var.dtype))
    counter_reset_op = smart_assign(self._counter,
                                    tf.constant(0, dtype=tf.int32))

    return tf.group(var_reset_op, counter_reset_op)


class PartitionedTensor(object):
  """A Tensor partitioned across its 0-th dimension."""

  def __init__(self, tensors):
    """Initializes PartitionedTensor.

    Args:
      tensors: List of Tensors. All Tensors must agree on shape (excepting
        batch dimension) and dtype.

    Raises:
      ValueError: If 'tensors' has length zero.
      ValueError: if contents of 'tensors' don't agree on shape or dtype.
    """
    if not tensors:
      raise ValueError("tensors must be a list of 1+ Tensors.")

    dtype = tensors[0].dtype
    if not all(tensor.dtype == dtype for tensor in tensors):
      raise ValueError(
          "all tensors must have the same dtype. The tensors are {}".format(
              tensors))

    shape = tensors[0].shape[1:]
    if not all(tensor.shape[1:] == shape for tensor in tensors):
      raise ValueError("All tensors must have shape = %s (excluding batch "
                       "dimension)." % shape)

    one_hot_depth = getattr(tensors[0], "one_hot_depth", None)
    if not all(
        getattr(tensor, "one_hot_depth", None) == one_hot_depth
        for tensor in tensors):
      raise ValueError(
          "All tensors must have one_hot_depth {}".format(one_hot_depth))

    self.tensors = tensors

  @property
  def shape(self):
    feature_shape = self.tensors[0].shape[1:]
    batch_size = sum([tensor.shape[0] for tensor in self.tensors],
                     tf.Dimension(0))
    return tf.TensorShape([batch_size]).concatenate(feature_shape)

  def get_shape(self):
    return self.shape

  @property
  def dtype(self):
    return self.tensors[0].dtype

  @property
  def one_hot_depth(self):
    return getattr(self.tensors[0], "one_hot_depth", None)

  def __str__(self):
    return "PartitionedTensor([%s, ...], dtype=%s, shape=%s)" % (
        self.tensors[0].name, self.dtype.name, tuple(self.shape.as_list()))

  def __hash__(self):
    return hash(tuple(self.tensors))

  def __eq__(self, other):
    if not isinstance(other, PartitionedTensor):
      return False
    return self.tensors == other.tensors

  def __ne__(self, other):
    return not self == other  # pylint: disable=g-comparison-negation

  def __getitem__(self, key):
    return self.as_tensor()[key]

  def as_tensor(self, dtype=None, name=None, as_ref=False):
    with tf.name_scope(name, "PartitionedTensor.as_tensor", self.tensors):
      assert not as_ref
      assert dtype in [None, self.dtype]
      return tf.concat(self.tensors, axis=0)

  @property
  def device(self):
    # PartitionedTensors in general do not live on a single device.  If the
    # device cannot be determined unambiguously this property will return None.
    device = self.tensors[0].device
    if all(tensor.device == device for tensor in self.tensors):
      return device
    return None


tf.register_tensor_conversion_function(
    PartitionedTensor,
    lambda val, dtype, name, as_ref: val.as_tensor(dtype, name, as_ref))


# TODO(b/69623235): Add a function for finding tensors that share gradients
# to eliminate redundant fisher factor computations.


def _check_match_lists_of_pairs(list1, list2):
  for (_, var1), (_, var2) in zip(list1, list2):
    if var1 is not var2:
      raise ValueError("The variables referenced by the two arguments "
                       "must match.")


def sprod(scalar, list_):
  # Product of scalar with list of items.
  return tuple(scalar*item for item in list_)


def sprod_p(scalar, list_):
  # Product of scalar with list of (item, var) pairs.
  return tuple((scalar*item, var) for (item, var) in list_)


def sum_(list1, list2):
  # Element-wise sum of lists of tensors.
  return tuple(item1 + item2 for item1, item2 in zip(list1, list2))


def sum_p(list1, list2):
  # Element-wise sum of lists of (tensor, var) pairs.
  _check_match_lists_of_pairs(list1, list2)
  return tuple((item1 + item2, var1)
               for (item1, var1), (item2, var2) in zip(list1, list2))


def ip(list1, list2):
  # Inner product of lists of tensors.
  return tf.add_n(tuple(tf.reduce_sum(tensor1 * tensor2)
                        for tensor1, tensor2 in zip(list1, list2)))


def ip_p(list1, list2):
  # Inner product of lists of (tensor, var) pairs.
  _check_match_lists_of_pairs(list1, list2)

  return ip(tuple(tensor for (tensor, _) in list1),
            tuple(tensor for (tensor, _) in list2))


def assert_variables_match_pairs_list(a_and_vars,
                                      b_and_vars,
                                      error_message=None):
  """Assert the variables in two lists of (tensor, var) pairs are the same.

  Args:
    a_and_vars: a list of (tensor, variable) pairs.
    b_and_vars: a list of (tensor, variable) pairs.
    error_message: an optional string prepended to the error message.

  Raises:
    ValueError: if any variables in the input pair lists are not the same.
  """
  _, a_variables = zip(*a_and_vars)
  _, b_variables = zip(*b_and_vars)
  variable_mismatch_indices = []
  for vi, (a_var, b_var) in enumerate(zip(a_variables, b_variables)):
    if a_var is not b_var:
      variable_mismatch_indices.append(vi)

  if variable_mismatch_indices:
    mismatch_indices_str = ", ".join(map(str, variable_mismatch_indices))
    a_variables_str = ", ".join(map(str, a_variables))
    b_variables_str = ", ".join(map(str, b_variables))
    error_str = "Mismatch on variable lists at indices {}.\n{}\n{}".format(
        mismatch_indices_str, a_variables_str, b_variables_str)
    if error_message:
      error_str = "{} {}".format(error_message, error_str)
    raise ValueError(error_str)


def multiline_print(lists):
  """Prints multiple lines of output using tf.print."""

  combined_list = []
  combined_list += lists[0]

  # We prepend newline characters to strings at the start of lines to avoid
  # the ugly space intendations that tf.print's behavior of separating
  # everything with a space would otherwise cause.
  for item in lists[1:]:
    if isinstance(item[0], str):
      combined_list += (("\n" + item[0],) + item[1:])
    else:
      combined_list += (("\n",) + item)

  return tf.print(*combined_list)


def get_shape(tensor):
  """Returns list of dimensions using ints only for statically known ones."""

  if tensor.shape.dims is None:
    raise ValueError("Unknown rank for tensor {}.".format(tensor))

  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.shape(tensor)
  return tuple(elt if elt is not None else dynamic_shape[idx]
               for idx, elt in enumerate(static_shape))


def cls_name(obj):
  return obj.__class__.__name__


def is_reference_variable(x):
  return ((isinstance(x, tf.Variable)
           and not resource_variable_ops.is_resource_variable(x))
          or hasattr(x, "_should_act_as_ref_variable"))


class MovingAverageVariable(object):
  """A variable updated using weighted moving averages.

  Note that to implement a traditional decaying exponential average one should
  use a decay value smaller than 1.0 (e.g. 0.9), and set weight = 1.0 - decay.
  Doing this and setting normalize_value to True will implement "zero-debiased"
  decayed averages.
  """

  def __init__(self, name, shape, dtype, initializer=tf.zeros_initializer(),
               normalize_value=True):
    """Constructs a new `MovingAverageVariable`.

    Args:
      name: `string`. Scope for the variables.
      shape: shape of the variable.
      dtype: dtype of the variable.
      initializer: initializer for the variable (see tf.get_variable). Should
        be tf.zeros_initializer() unless you know what you are doing.
        (Default: tf.zeros_initializer())
      normalize_value: bool. If True we normalize the value property by the
        total weight (which will be subject to decay). (Default: False)
    """
    self._normalize_value = normalize_value

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      self._var = tf.get_variable(
          "var",
          shape=shape,
          dtype=dtype,
          initializer=initializer,
          trainable=False,
          use_resource=True)

      self._total_weight = tf.get_variable(
          "total_weight",
          shape=(),
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          trainable=False,
          use_resource=True)

  @property
  def dtype(self):
    return self._var.dtype.base_dtype

  @property
  def value(self):
    if self._normalize_value:
      return self._var / self._total_weight
    else:
      return tf.identity(self._var)

  def add_to_average(self, value, decay=1.0, weight=1.0):
    """Add a value into the moving average.

    Args:
      value: a Tensor matching the shape and dtype that was passed to the
        constructor.
      decay: float or 0D Tensor. The current value is multiplied by this before
        the value is added, as is the total accumulated weight. (Default: 1.0)
      weight: float or 0D Tensor. The value being added is multiplied by this.
        Also this is added to the total accumulated weight. (Default: 1.0)
    """

    decay = tf.convert_to_tensor(decay, dtype=self.dtype)
    weight = tf.convert_to_tensor(weight, dtype=self.dtype)

    update_var = smart_assign(self._var, decay * self._var + weight * value)

    update_total_weight = smart_assign(self._total_weight,
                                       decay * self._total_weight + weight)

    return tf.group(update_var, update_total_weight)
