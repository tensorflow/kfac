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
"""Utility Functions for using KFAC with Keras Objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from kfac.python.ops import layer_collection as kfac_layer_collection
from kfac.python.ops.tensormatch import tensorflow_graph_util

layers = tf.keras.layers
losses = tf.keras.losses
activations = tf.keras.activations
K = tf.keras.backend

# Added when serializing layer class names to prevent serialized class names
# from clashing with user-defined layer names.
_CLASS_NAME_PREFIX = 'kfac_class_'
_KERAS_LOSS_TO_KFAC_REGISTER_FUNC = {
    'sparsecategoricalcrossentropy':
        kfac_layer_collection.LayerCollection
        .register_softmax_cross_entropy_loss,
    'categoricalcrossentropy':
        kfac_layer_collection.LayerCollection
        .register_softmax_cross_entropy_loss,
    'binarycrossentropy':
        kfac_layer_collection.LayerCollection
        .register_sigmoid_cross_entropy_loss,
}


def get_parent(node):
  """Retrieves the parent tf.Tensor of node in the computation graph.

  Args:
    node: A tf.Tensor.

  Raises:
   ValueError: If the node has more than one input op.
   ValueError: If the node has more than one parent tf.Tensor.

  Returns:
    The parent tensor of the node on the computation graph.
  """
  edge = tensorflow_graph_util.expand_inputs(node)
  if len(edge) != 1:
    raise ValueError('{} has more than one input op.'.format(node))
  parent = tensorflow_graph_util.expand_inputs(edge[0])
  if len(parent) != 1:
    raise ValueError('{} has more than one parent tensor.'.format(node))
  return parent[0]


def serialize_loss(loss):
  """Serialize a valid Keras Kfac loss argument."""
  def serialize(x):
    return x if isinstance(x, six.string_types) else losses.serialize(x)

  if not loss or isinstance(loss, six.string_types):
    return loss
  elif isinstance(loss, dict):
    return {k: serialize(v) for k, v in loss.items()}
  elif isinstance(loss, list):
    return [serialize(v) for v in loss]
  else:
    return losses.serialize(loss)


def serialize_fisher_approx(fisher_approx):
  """Serialize a valid fisher approximation dict or list."""
  def serialize(key):
    return (key if isinstance(key, six.string_types) else _CLASS_NAME_PREFIX +
            key.__name__)

  if isinstance(fisher_approx, dict):
    fisher_approx = {serialize(k): v for k, v in fisher_approx.items()}
  return fisher_approx


def _get_verified_dict(container, container_name, layer_names):
  """Verifies that loss_weights/fisher_approx conform to their specs."""
  if container is None or container == {}:  # pylint: disable=g-explicit-bool-comparison
    # The explicit comparison prevents empty lists from passing.
    return {}
  elif isinstance(container, dict):
    string_keys = {
        str(k) for k in container if isinstance(k, six.string_types) and
        not k.startswith(_CLASS_NAME_PREFIX)
    }
    if string_keys - set(layer_names):
      raise ValueError('There is a {} without a matching layer'
                       .format(container_name))
    return container
  elif isinstance(container, list):
    if len(layer_names) != len(container):
      raise ValueError('Number of {} and layers don\'t match.'
                       .format(container_name))
    return dict(zip(layer_names, container))
  else:
    raise ValueError('{} must be a list or dict'.format(container_name))


def register_layer(layer_collection, layer, fisher_approx=None, **kwargs):
  """Get layer collection with all layers and loss registered.

  Args:
   layer_collection: LayerCollection object on which the layer will be
     registered.
   layer: Keras layer to register with the layer_collection.
   fisher_approx: Option string specifying the fisher approximation type.
   **kwargs: Keyword arguments to be forwarded to the layer registration
     function.

  Raises:
   ValueError: If there is a layer with trainable parameters that isn't Conv1D,
     Conv2D, Dense, BatchNormalization, LayerNormalization or Embedding.
   ValueError: If convolutional layers don't use the "channels_last" format.

  Returns:
    A kfac.LayerCollection with the model's layers and loss registered.
  """
  # The inbound_nodes property is currently deprecated, but appears to be
  # supported in non-eager TF 1.x. This may change.
  # If there are multiple inbound_nodes, it means the model was used as a
  # callable (i.e. y = model(x)). We assume the inputs/outputs from the call
  # need to be registered and not the nodes from the original built model or
  # any other previous calls, since layers can't be used multiple times
  # (RNN-style) with Keras KFAC.
  node = layer.inbound_nodes[-1]
  pre_activation_output = node.output_tensors
  if hasattr(layer, 'activation') and layer.activation != activations.linear:
    pre_activation_output = get_parent(pre_activation_output)

  # This will allow unsupported layers to be in our model as long as KFAC
  # doesn't have to minimize with respect to those parameters.
  if layer.count_params() and layer.trainable:
    if any(isinstance(tensor, (list, tuple))
           for tensor in (node.input_tensors, node.output_tensors)):
      raise ValueError('Individual layers can only have 1 input_tensor and 1 '
                       'output tensor. You are likely using an unsupported '
                       'layer type. Error on layer {}'.format(layer))

    weights = layer.trainable_weights
    kwargs.update({
        'inputs': node.input_tensors,
        'outputs': pre_activation_output,
        'params': weights if len(weights) > 1 else weights[0],
        'approx': fisher_approx,
    })

    # TODO(b/133849249) Support RNNs and other shared weight layers.
    if isinstance(layer, layers.Dense):
      layer_collection.register_fully_connected(**kwargs)
    elif isinstance(layer, layers.Embedding):
      layer_collection.register_fully_connected(dense_inputs=False, **kwargs)
    elif isinstance(layer, (layers.BatchNormalization,
                            layers.LayerNormalization)):
      if not layer.scale:
        # With Batch/Layer Normalization, the user can specify if they want
        # the input to be scaled and/or shifted after it is normalized.
        raise ValueError('Kfac currently does not support batch/layer '
                         'normalization with scale=False. Error on layer {}'
                         .format(layer))
      # Undo batchnorm by subtracting the shift and diving by scale.
      kwargs['inputs'] = ((kwargs['outputs'] - weights[1]) / weights[0]
                          if layer.center else kwargs['outputs'] / weights)
      layer_collection.register_scale_and_shift(**kwargs)

      # A learning_phase of 1 or 0 means it's been set. False means it hasn't.
      is_phase_set = K.get_value(K.learning_phase()) != False  # pylint: disable=g-explicit-bool-comparison
      if hasattr(layer, 'fused') and layer.fused and not is_phase_set:
        # For the fused implementation of the BatchNormalization, there are
        # two ops: one for training and one for inference. When the
        # learning_phase is set, during layer creation, there is a
        # tf_utils.smart_cond that will only create one of the ops. When the
        # learning_phase is not set, it will create a tf.cond with both ops as
        # branches. So, when learning_phase is not set, we must add a "use"
        # for the gamma/beta variables to account for there being two ops that
        # are consumers of the variables. Linked below is the smart_cond in
        # BatchNormalization:
        # https://github.com/tensorflow/tensorflow/blob/59217f581fdef4e5469a98b62e38f851eac88688/tensorflow/python/keras/layers/normalization.py#L513
        # Updated 2019-06-22.
        layer_collection._add_uses(weights, 1)  # pylint: disable=protected-access

    elif all(hasattr(layer, a) for a in
             ('strides', 'padding', 'dilation_rate')):
      if layer.data_format != 'channels_last':
        raise ValueError('KFAC currently only supports the "channels_last" '
                         'data format for convolutional layers. Error on '
                         'layer {}'.format(layer))

      kwargs['padding'] = layer.padding.upper()
      kwargs['strides'] = [1] + list(layer.strides) + [1]
      kwargs['dilations'] = [1] + list(layer.dilation_rate) + [1]

      if isinstance(layer, layers.Conv2D):
        layer_collection.register_conv2d(**kwargs)
      elif isinstance(layer, layers.Conv1D):
        layer_collection.register_conv1d(**kwargs)
      # Depthwise and Separable Conv2D are not supported yet because they are
      # experimental in tensorflow_kfac.
      else:
        raise ValueError('Unsupported convolutional layer type: {}'
                         .format(layer))
        # TODO(b/133849240): Support registering any convolution type.
    else:
      raise ValueError('Unsupported layer type: {}'.format(layer))
      # TODO(b/133849243): Support registering any generic layer type.


def register_loss(layer_collection, layer, loss, **kwargs):
  """Registers the loss with the layer for the layer_collection.

  Args:
   layer_collection: LayerCollection object on which the layer and loss will
     be registered.
   layer: Keras layer whose outputs will be used with the loss function.
   loss: Keras (normal or serialized) loss function. Currently,
     sparse/normal categorical/binary cross entropy and MSE are supported.
   **kwargs: Keyword arguments to be forwarded to the function that
     registers the loss. A couple of notable ones include coeff (the weight of
     the loss) and seed (the seed used when sampling from the output
     distribution).

  Raises:
   ValueError: If a loss function other than MSE and cross entropy
     variants is used.

  Raises:
   ValueError: If a loss function other than MSE and cross entropy
     variants is used.
  """
  node = layer.inbound_nodes[-1]
  pre_activation_output = node.output_tensors
  if hasattr(layer, 'activation') and layer.activation != activations.linear:
    pre_activation_output = get_parent(pre_activation_output)

  # A Keras loss can be a callable class or a function. Their serialized
  # forms differ. The logic below normalizes these difference. This will
  # not work for custom losses (we do not intend to support custom loss
  # functions for now).
  if not isinstance(loss, six.string_types):
    loss = losses.serialize(loss)
  if isinstance(loss, dict):
    loss = loss['class_name']
  loss = loss.replace('_', '').lower()

  if loss in ('meansquarederror', 'mse'):
    # We use the actual output here instead of the pre-activations because
    # MSE is computed with the output. For the logit loss functions,
    # tensorflow_kfac needs the pre-activations.
    layer_collection.register_squared_error_loss(layer.output, **kwargs)
  elif loss in _KERAS_LOSS_TO_KFAC_REGISTER_FUNC:
    _KERAS_LOSS_TO_KFAC_REGISTER_FUNC[loss](
        layer_collection, logits=pre_activation_output, **kwargs)
  else:
    raise ValueError('Unsupported loss function: {}'.format(loss))


def get_layer_collection(model,
                         loss=None,
                         loss_weights=None,
                         fisher_approx=None,
                         layer_collection=None,
                         seed=None):
  """Get layer collection with all layers and loss registered.

  Args:
   model: Keras model whose layers to register. Currently, Conv1D,
     Conv2D, Dense, BatchNormalization, LayerNormalization and Embedding layers
     are supported in a Functional or Sequential model. Other layer types are
     supported as long as they aren't trainable (or don't have weights). Nested
     models are supported.
   loss: Optional Keras (normal or serialized) loss function. Could be a list or
     a dictionary mapping layer names to (normal or serialized) loss functions.
     if there are multiple losses Currently, sparse/normal categorical/binary
     cross entropy and MSE are supported. You must register at least one loss
     with the layer collection before it can be used.
   loss_weights: An optional list of coefficients or a dictionary mapping
     layer names to the coefficient for each loss function. If it is a list,
     there must be a the same number of coefficients as loss functions. If
     it is a dictionary and a coefficient is not given for a loss function,
     a coefficient of 1.0 will be used.
   fisher_approx: An optional list of approximations or a dictionary mapping
     layer name/class to fisher approximation type. If it is a list, there must
     be the same number of approximations as there are layers with trainable
     parameters. For each layer, the approximation is determined as follows:
     if fisher_approx is a dictionary, first we check if the name is in the
     dict, if it isn't found the layer class is checked, if that isn't found
     the default is used. When fisher_approx is a list, the order of the
     approximations must match the order of the layers with trainable parameters
     given by model.layers. None is a valid dict/list entry and indicates to use
     the default approximation for that layer.
   layer_collection: Optional LayerCollection object on which the model and loss
     will be registered.
   seed: Optional integer specifing the TensorFlow random seed. To get
     deterministic behaviour, the seed needs to be set because the targets
     are sampled to approximate the fisher.

  Raises:
   ValueError: If there is a layer with trainable parameters that isn't Conv1D,
     Conv2D, Dense, BatchNormalization, LayerNormalization or Embedding.
   ValueError: If a loss function other than MSE and cross entropy
     variants is used.
   ValueError: If there isn't a one-to-one correspondence between
     loss/loss_weights and output layers, or if loss_weights isn't a list/dict.
   ValueError: If convolutional layers don't use the "channels_last" format.

  Returns:
    A kfac.LayerCollection with the model's layers and loss registered.
  """
  if not layer_collection:
    layer_collection = kfac_layer_collection.LayerCollection()

  if not loss:
    loss = {}
  elif isinstance(loss, dict):
    if set(model.output_names) != set(loss.keys()):
      raise ValueError('Output layer names and loss dict keys don\'t match'
                       ' \nmodel.output_names: {} \nloss dict keys: {}'
                       .format(model.output_names, loss.keys()))
  elif isinstance(loss, list):
    if len(model.output_names) != len(loss):
      raise ValueError('Number of loss dict items doesn\'t match number of '
                       'output layers. \nmodel.output_names: {} \nloss list: '
                       '{}'.format(model.output_names, loss))
    loss = dict(zip(model.output_names, loss))
  else:
    if len(model.output_names) > 1:
      raise ValueError('More output layers than losses. \n'
                       'model.output_names: {} \nloss: {}'
                       .format(model.output_names, loss))
    # When the model is used as a callable, the model's output_names may not
    # match the actual output layer's name. In the one output case, we always
    # want the last layer, so we use the last layer's name.
    loss = {model.layers[-1].name: loss}

  # We want to do a left-to-right depth-first traversal of the model to get the
  # correct flattened order of the layers. The order only matters for the
  # fisher_approx in list form.
  flattened_layers = []
  layer_stack = model.layers[::-1]
  while layer_stack:
    layer = layer_stack.pop()
    if hasattr(layer, 'layers'):
      if layer.name in loss:
        if len(layer.output_names) > 1:
          raise ValueError('Nested models with multiple outputs are '
                           'unsupported.')
        loss[layer.output_names[0]] = loss.pop(layer.name)
      layer_stack += layer.layers[::-1]
    else:
      flattened_layers.append(layer)

  trainable_layer_names = [l.name for l in flattened_layers if
                           l.count_params() and l.trainable]
  fisher_approx = _get_verified_dict(fisher_approx, 'fisher_approx',
                                     trainable_layer_names)
  # The Optimizer class passes in a serialized fisher_approx dictionary, but the
  # user may not. We serialize it so we can use it uniformly.
  fisher_approx = serialize_fisher_approx(fisher_approx)
  loss_weights = _get_verified_dict(loss_weights, 'loss_weights',
                                    model.output_names)

  for layer in flattened_layers:
    if layer.name in fisher_approx:
      approx = fisher_approx[layer.name]
    else:
      approx = fisher_approx.get(
          _CLASS_NAME_PREFIX + layer.__class__.__name__, None)

    register_layer(layer_collection, layer, fisher_approx=approx)

    if layer.name in loss:
      register_loss(layer_collection=layer_collection,
                    layer=layer,
                    loss=loss[layer.name],
                    coeff=loss_weights.get(layer.name, 1.0),
                    seed=seed)

  return layer_collection


def get_loss_fn(
    model, loss, loss_weights=None, reduce_fn=tf.reduce_mean, name='loss'):
  """Creates a loss function to be used for KFAC's adaptive damping.

  This allows Keras KFAC to automatically create the loss function to use
  for adaptive_damping. This function would also be useful for a custom training
  loop that uses adaptive_damping.

  The returned loss function currently does not support masks or sample_weights.

  Currently, if you use a categorical crossentropy loss, due to the
  implementation of tf.keras.losses.*_crossentropy, it will  grab the logits
  whether you use a softmax at the end of your model or not. This is true as of
  August 1, 2019. Code below:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py#L4322

  Args:
    model: tf.keras.Model model that will be used with the inputs to the
      returned loss_fn.
    loss: Potentially serialized tf.keras.losses.* loss function(s)/class(es).
      If the model has multiple outputs, this must be a list of losses that
      matches the order of model.outputs, or a dictionary with names matching
      output_names. Must accept kwargs y_pred and y_true. Note that if your
      model's output are logits, you should pass a callable Keras with
      from_logits=True. This function could be a non-Keras loss, but it is
      untested in this case.
    loss_weights: If you have multiple losses, a list or dictionaryof weights
      for each loss. A default value of 1.0 is given for losses that don't have
      a weight when a dictionary is passed.
    reduce_fn: The function that will be used to aggregate the loss tensor.
      tf.reduce_mean by default. You may replace this with the identity if your
      loss does a reduction by default. Depending on how you compute your loss
      in a distributed setting, you may want to modify this function (for
      example, if you sum across replicas, then the reduce_fn might be
      lambda x: tf.reduce_sum(x) * (1.0 / global_batch_size).
    name: Name scope for the loss_fn ops.

  Raises:
    ValueError: If the loss is a dictionary.

  Returns:
    A function that takes inputs and optionally a prediction and will return
    a loss. This can be used as the KFAC loss_fn for adaptive damping.
  """
  if isinstance(loss, six.string_types):
    loss = losses.deserialize(loss)
  elif isinstance(loss, dict):
    loss = [loss[n] for n in model.output_names]

  if isinstance(loss, list):
    loss = [losses.deserialize(l) if isinstance(l, six.string_types) else l
            for l in loss]

  if isinstance(loss_weights, dict):
    loss_weights = [loss_weights.get(n, 1.0) for n in model.output_names]

  def loss_fn(inputs, prediction=None):
    """Computes loss for a model given inputs.

    This function is meant to be used with K-FAC's adaptive damping, which is
    why the prediction is optional (since K-FAC wants to compute the loss just
    given inputs).

    Args:
      inputs: A tuple with (model_input(s), label(s)), where both elements are
        tensors or lists/tuples of tensors.
      prediction: The output of the model given the inputs. If this isn't,
        provided, the prediction will be computed via
        prediction = model(inputs[0])

    Returns:
      A tensor with the total reduced loss including regularization and other
      layer specific losses.
    """
    with tf.name_scope(name):
      x, y = inputs
      if prediction is None:
        prediction = model(x, training=False)

      if isinstance(prediction, (tuple, list)):
        reduced_losses = [reduce_fn(fn(y_pred=pred_i, y_true=y_i))
                          for fn, pred_i, y_i in zip(loss, prediction, y)]
        if loss_weights:
          reduced_losses = [l * w for l, w in zip(reduced_losses, loss_weights)]
        total_loss = tf.add_n(reduced_losses)
      else:
        total_loss = reduce_fn(loss(y_pred=prediction, y_true=y))

      # Adds regularization penalties and other custom layer specific losses.
      if model.losses:
        total_loss += tf.add_n(model.losses)

    return total_loss

  return loss_fn

