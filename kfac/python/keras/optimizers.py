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
"""KFAC Optimizer for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import numbers
import re
import warnings
import tensorflow as tf

from tensorflow.python.keras import backend
from kfac.python.keras import utils
from kfac.python.ops import optimizer
from kfac.python.ops.kfac_utils import periodic_inv_cov_update_kfac_opt

# TODO(b/135110195): Support letting the user choose the TF KFAC optimizer.
_KFAC_OPT_CLASS = periodic_inv_cov_update_kfac_opt.PeriodicInvCovUpdateKfacOpt

# TODO(b/134945404): Change how default config args are retrieved.
_KFAC_ARGS = inspect.getargspec(optimizer.KfacOptimizer.__init__)
_PERIODIC_KFAC_ARGS = inspect.getargspec(_KFAC_OPT_CLASS.__init__)
_DEFAULT_KWARGS = dict(zip(reversed(_KFAC_ARGS.args),
                           reversed(_KFAC_ARGS.defaults)))
_DEFAULT_KWARGS.update(zip(reversed(_PERIODIC_KFAC_ARGS.args),
                           reversed(_PERIODIC_KFAC_ARGS.defaults)))

_MUTABLE_HYPER_PARAMS = {'learning_rate',
                         'momentum',
                         'damping',
                         'weight_decay_coeff',
                         'norm_constraint'}


def _configure_kfac_kwargs_for_adaptive(kfac_kwargs, adaptive):
  """Checks and fills in some required kwargs to use an adaptive mode.

  This will set up kfac_kwargs for adaptive, adapt_damping, and/or qmodel
  momentum, if needed. It will not check for train_batch or batch_size, as that
  check happens right before the minimize. It will set the following if not set
  by the user:

  If adaptive=True:
    - adapt_damping=True, momentum=None, momentum_type='qmodel'
    - The checks listed below.

  If adapt_damping=True:
    - use_passed_loss=True, and then it will get the loss_tensor from minimize.
    - update_damping_immediately=True
    - damping_adaptation_interval=5 if the user hasn't set this already.
    - invert_every=5 if the user hasn't set this already.

  If momentum_type='qmodel' or momentum_type='qmodel_fixedmu':
    - Ensures learning rate and momentum are None.

  Args:
   kfac_kwargs: dict of keyword arguments to be passed to
     PeriodicInvCovUpdateKfacOpt.
   adaptive: bool indicating the optimizer is in adaptive mode.
  """
  if adaptive:
    kfac_kwargs.update({
        'adapt_damping': True,
        'momentum': None,
        'momentum_type': 'qmodel',
    })

  if kfac_kwargs.get('momentum_type', 'regular').lower().startswith('qmodel'):
    if kfac_kwargs['learning_rate']:
      raise ValueError('learning_rate must be None to use adaptive/qmodel.')
    if kfac_kwargs.get('momentum', None):
      raise ValueError('momentum must be None to use adaptive/qmodel.')

  if kfac_kwargs.get('adapt_damping', False):
    defaults = {'use_passed_loss': True, 'update_damping_immediately': True}
    # This way, we keep the user's preferences and only replace missing items.
    defaults.update(kfac_kwargs)
    kfac_kwargs.update(defaults)

    if not ('invert_every' in kfac_kwargs and
            'damping_adaptation_interval' in kfac_kwargs):
      # damping_adaptation_interval % invert_every must = 0
      kfac_kwargs['invert_every'] = 5
      kfac_kwargs['damping_adaptation_interval'] = 5


class Kfac(tf.keras.optimizers.Optimizer):
  """The KFAC Optimizer for Keras."""

  def __init__(self,  # pylint: disable=invalid-name
               _sentinel=None,
               learning_rate=None,
               damping=None,
               model=None,
               loss=None,
               loss_weights=None,
               fisher_approx=None,
               layer_collection=None,
               adaptive=False,
               train_batch=None,
               name=None,
               seed=None,
               **kfac_kwargs):
    """Construct a new KFAC optimizer.

    If you construct this Optimizer without a model with a loss, model and loss,
    or a layer_collection, you must call register_layers before using the
    optimizer.

    If you use adaptive, adapt_damping, or qmodel_momentum, this class will set
    up the required loss functions and tensors. You must pass the train_batch
    tensors as a tuple (x, y). If the batch_size cannot be inferred from the
    train_batch[0] tensor, you pass in the batch_size in the constructor. You
    may not use numpy arrays as input when using the adaptive mode. If you do
    not use minimize, you must also provide the loss_tensor.

    When using Distribution Strategy, K-FAC expects an unscaled loss tensor
    (i.e. not scaled by 1.0 / global_batch_size), unlike what is commonly
    recommended. This means you cannot use K-FAC with a Distribution Strategy
    and model.fit at the same time, since model.fit does this scaling for you.
    Instead, use a custom training loop with Distribution Strategy (there are
    examples in the Github repo).

    Args:
      _sentinel: Used to prevent positional parameters. Internal, do not use.
      learning_rate: float or 0D Tensor. Required if not using adapt_damping.
        Refer to kfac.KfacOptimizer for a detailed description.
      damping: Required. float or 0D Tensor. Refer to kfac.KfacOptimizer for a
        detailed description.
      model: Keras model which this class will optimize. Currently, dense, Conv
        1D/2D, and embedding are supported as trainable layers.
      loss: Keras (normal or serialized) loss function. Could be a list or a
        dictionary mapping layer names to (normal or serialized) loss functions.
        Currently, sparse/normal categorical/binary cross entropy and MSE are
        supported.
      loss_weights: An optional list of coefficients or a dictionary mapping
        layer names to the coefficient for each loss functions. If it is a list,
        there must be a the same number of coefficients as loss functions. If
        it is a dictionary and a coefficient is not given for a loss function,
        a coefficient of 1.0 will be used.
      fisher_approx: An optional list of approximations or a dictionary mapping
        layer name/class to fisher approximation type. If it is a list, there
        must be the same number of approximations as there are layers with
        trainable parameters. For each layer, the approximation is determined as
        follows. If fisher_approx is a dictionary, first we check if the name is
        in the dict, if it isn't found the layer class is checked, if it isn't
        found the default is used. When fisher_approx is a list, the order of
        the approximations must match the order of the layers with trainable
        parameters given by model.layers. None is a valid dict/list entry and
        indicates to use the default approximation for that layer.
      layer_collection: Only use this argument when you have an unsupported
        model architecture and so manually register the layers. Refer to
        kfac.KfacOptimizer for a detailed description.
      adaptive: Whether this optimizer is in adaptive mode or not. In adaptive
        mode, we set momentum_type='qmodel' and adapt_damping=True, so you must
        provide the damping (used as the initial value). learning_rate and
        momentum must be None. You must provide a train_batch and potentially
        a batch_size if we cannot infer the batch_size from the train_batch.
      train_batch: A tuple (input, label). The input must be a tensor or a list
        of tensors that you can call the model on. The label must be a tensor
        or list of tensors compatible with the loss_fn. See utils.get_loss_fn
        for the standard loss_fn we create, or you can provide a custom loss_fn.
      name: Optional name for operations created when applying gradients.
        Defaults to "kfac".
      seed: Optional integer specifying the TensorFlow random seed. To get
        deterministic behaviour, the seed needs to be set because the targets
        are sampled to approximate the fisher.
      **kfac_kwargs: Additional arguments to be passed to
        kfac.PeriodicInvCovUpdateKfacOpt (and then to kfac.KfacOptimizer). Note
        the "loss" argument for kfac.KfacOptimizer should be passed as
        "loss_tensor".

    Raises:
      ValueError: If clipvalue or clipnorm arguments are used.
      ValueError: If positional arguments are used (or _sentinel is used).
      ValueError: If damping is not provided.
      ValueError: If learning_rate or momentum are set with adaptive=True.
    """
    if tf.executing_eagerly():
      warnings.warn('Eager mode appears to be enabled. Kfac is untested in '
                    'eager mode.')
    if _sentinel:
      raise ValueError('Do not pass positional arguments, only use keyword '
                       'arguments.')
    if damping is None:
      raise ValueError('Please provide a value for damping.')

    if 'clipvalue' in kfac_kwargs:
      raise ValueError('Argument "clipvalue" is not support.')
    if 'clipnorm' in kfac_kwargs:
      raise ValueError('Argument "clipnorm" is not supported. Use '
                       '"norm_constraint" instead.')

    super(Kfac, self).__init__(name=name)

    kfac_kwargs.update({'name': self._name,
                        'learning_rate': learning_rate,
                        'damping': damping})

    _configure_kfac_kwargs_for_adaptive(kfac_kwargs, adaptive)

    self._optimizer = None
    self._layer_collection = None
    self._model = model
    self._loss = loss
    self._have_tracked_vars = False
    self._tf_var_scope = self._name + '/tf_vars'
    # We use _kfac_kwargs and _config in various parts in the code below.
    # _kfac_kwargs is checked when we want to know only what the user passed.
    # _config is used when we want user selections with the default kwargs as a
    # fallback.
    self._kfac_kwargs = kfac_kwargs
    self._layer_collection_kwargs = {
        'loss_weights': loss_weights,
        'fisher_approx': utils.serialize_fisher_approx(fisher_approx),
        'seed': seed,
    }
    self._config = _DEFAULT_KWARGS.copy()
    self._config.update(kfac_kwargs)
    self._config.update(self._layer_collection_kwargs)
    self._config['loss'] = utils.serialize_loss(loss)

    if 'loss_tensor' in self._kfac_kwargs:
      self._kfac_kwargs['loss'] = self._kfac_kwargs.pop('loss_tensor')

    self._mutable_hypers = _MUTABLE_HYPER_PARAMS.copy()
    if self._config['adapt_damping']:
      self._mutable_hypers.remove('damping')
    if self._config['momentum_type'].lower().startswith('qmodel'):
      self._mutable_hypers -= {'learning_rate', 'momentum'}
    for hp in self._mutable_hypers.copy():
      if self._config[hp] is None:
        self._mutable_hypers.remove(hp)
      else:
        self._set_hyper(hp, self._config[hp])

    if layer_collection:
      self.register_layers(layer_collection=layer_collection)
    if train_batch and self._kfac_kwargs.get('adapt_damping', False):
      self.register_train_batch(train_batch=train_batch)

  @property
  def name(self):
    # This settable property exists to avoid variable name scope conflicts.
    return self._name

  @name.setter
  def name(self, value):
    if self._optimizer:
      raise ValueError('Can\'t change the optimizer\'s name after the variables'
                       ' are created')
    self._name = value
    self._config['name'] = value
    self._kfac_kwargs['name'] = value
    self._tf_var_scope = value + '/tf_vars'

  @property
  def optimizer(self):
    # We defer the creation of the optimizer for a few reasons. First, if the
    # user decides to use the model as a callable, we want to capture the latest
    # inbound node of the model. Also, this mimics the behaviour of existing
    # Keras optimizers, as all the variables are created on the first
    # apply_gradients call (unless the user tries to access this property).
    # Second, this reduces code duplication as we can use the super class's
    # _set_hypers and _create_hypers methods. Finally, if the user restores an
    # optimizer, this allows them to control the variable scope before the
    # variables are created (to avoid scope conflicts).
    if not self._optimizer:
      self._create_optimizer()
    return self._optimizer

  @property
  def layers(self):
    return self._layer_collection

  @property
  def mutable_hyperparameters(self):
    return self._mutable_hypers

  def register_layers(self, model=None, loss=None, layer_collection=None):
    if not layer_collection:
      if not loss and hasattr(model, 'loss'):
        loss = model.loss
      if not (model and loss):
        raise ValueError('Please provide a model with a loss, a model and loss,'
                         ' or a LayerCollection')
      layer_collection = utils.get_layer_collection(
          model, loss, **self._layer_collection_kwargs)
    self._layer_collection = layer_collection
    self._kfac_kwargs['var_list'] = layer_collection.registered_variables

  def register_train_batch(self, train_batch, batch_size=None):
    """Configures the train_batch tuple and batch_size for adaptive damping."""
    if not isinstance(train_batch, tuple):
      raise ValueError('You must provide the train_batch tuple of inputs to '
                       'use adaptive/adapt_damping mode.')
    elif not all(isinstance(inp, tf.Tensor) for inp in train_batch):
      raise ValueError('You must use TF tensors as input.')
    self._kfac_kwargs['train_batch'] = train_batch

    if batch_size:
      self._kfac_kwargs['batch_size'] = batch_size
    elif 'batch_size' not in self._kfac_kwargs:
      inferred_batch_size = train_batch[0].shape.as_list()[0]
      if inferred_batch_size:
        self._kfac_kwargs['batch_size'] = inferred_batch_size
      else:
        raise ValueError('Could not infer batch_size from the train_batch. '
                         'Please provide it in the optimizer constructor or '
                         'through register_train_batch.')

  def minimize(self, loss, var_list, grad_loss=None, name=None):
    if (self._config['use_passed_loss'] and 'loss' not in self._kfac_kwargs):
      self._kfac_kwargs['loss'] = loss

    return self._call_and_track_vars(
        'minimize', loss, var_list=var_list, grad_loss=grad_loss, name=name)

  def apply_gradients(self, grads_and_vars, name=None):
    return self._call_and_track_vars(
        'apply_gradients', grads_and_vars, name=name)

  def get_updates(self, loss, params):
    return [self.minimize(loss, params)]

  def get_config(self):
    config = self._config.copy()
    for param in self._hyper:
      config[param] = self._serialize_hyperparameter(param)
    return config

  def _create_optimizer(self):
    """Initializes the hyperparameters and sets the self._optimizer property."""
    if self._optimizer:
      return
    if not self._layer_collection:
      self.register_layers(self._model, self._loss)

    if self._config['adapt_damping']:
      if 'train_batch' not in self._kfac_kwargs:
        raise ValueError('Must provide a train_batch tuple to use adaptive '
                         'damping. Use register_train_batch or pass it in '
                         'during optimizer construction.')
      if 'loss_fn' not in self._kfac_kwargs:
        self._kfac_kwargs['loss_fn'] = utils.get_loss_fn(
            self._model, self._loss, loss_weights=self._config['loss_weights'])

    with tf.name_scope(self._name):
      with tf.init_scope():
        # "iterations" property will create iterations if necessary.
        _ = self.iterations
        self._create_hypers()

    self._kfac_kwargs.update(self._hyper)
    try:
      # We use the TF 1 variable_scope instead of the TF 2 recommended
      # name_scope because we need to recover the variables created in this
      # scope, which is not possible with name_scope.
      with tf.variable_scope(self._tf_var_scope):
        self._optimizer = _KFAC_OPT_CLASS(
            layer_collection=self._layer_collection, **self._kfac_kwargs)
    except ValueError as e:
      msg = str(e)
      if re.search('Variable .* already exists', msg):
        raise ValueError(
            'You may have instantiated a KFAC Optimizer with the same name as '
            'an existing one. Try resetting the default graph, instantiating '
            'the optimizer with a different name, or changing the optimizer\'s '
            'name.\nHere is the original ValueError:\n ' + msg)
      elif re.search('Found the following errors with variable registration'
                     '.*gamma.*registered with wrong number of uses.*', msg):
        # We don't regex the name batch_normalization because the user could
        # have renamed the layer. We don't regex beta because they could have
        # used BatchNorm without the shift.
        raise ValueError(
            'There may have been an issue registering BatchNormalization. Try '
            'using tf.keras.backend.set_learning_phase before model '
            'construction. An alternative solution is to use the unfused '
            'batchnorm implementation (pass the argument fused=False to '
            'BatchNormalization).\nHere is the original ValueError:\n ' + msg)
      else:
        raise e

  def _call_and_track_vars(self, method_name, *args, **kwargs):
    # We call _create_optimizer outside of the var_scope because
    # _create_optimizer also opens the same variable_scope.
    self._create_optimizer()
    with tf.variable_scope(self._tf_var_scope):
      kwargs['global_step'] = self.iterations
      update_op = getattr(self._optimizer, method_name)(*args, **kwargs)

    if not self._have_tracked_vars:
      # We rely on the variables created in a deterministic order for get and
      # set weights. Sorting the variables by name is not a reliable way to
      # get a deterministic order due to the way TF KFAC assigns variable names.
      for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                   scope=self._tf_var_scope):
        backend.track_variable(var)
        self.weights.append(var)
      self._have_tracked_vars = True

    return update_op

  def _set_hyper(self, name, value):
    """Set hyper `name` to value. value must be numeric."""
    if self._hypers_created:
      if not isinstance(self._hyper[name], tf.Variable):
        raise AttributeError("Can't set attribute: {}".format(name))
      if not isinstance(value, numbers.Number):
        raise ValueError('Dynamic reassignment only supports setting with a '
                         'number. tf.Tensors and tf.Variables can only be used '
                         'before the internal kfac optimizer is created.')
      backend.set_value(self._hyper[name], value)
    else:
      super(Kfac, self)._set_hyper(name, value)
