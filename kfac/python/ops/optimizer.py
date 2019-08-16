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
"""The KFAC optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf

from kfac.python.ops import curvature_matrix_vector_products as cmvp
from kfac.python.ops import estimator as est
from kfac.python.ops import fisher_factors as ff
from kfac.python.ops import utils as utils

ip = utils.ip
ip_p = utils.ip_p
sprod = utils.sprod
sprod_p = utils.sprod_p

# If True we the damping contribution is included in the quadratic model for
# the purposes of computing qmodel_change in rho (the reduction ratio used in
# the LM damping adjustment method).
_INCLUDE_DAMPING_IN_QMODEL_CHANGE = False


def set_global_constants(include_damping_in_qmodel_change=None):
  """Sets various global constants used by the classes in this module."""
  global _INCLUDE_DAMPING_IN_QMODEL_CHANGE

  if include_damping_in_qmodel_change is not None:
    _INCLUDE_DAMPING_IN_QMODEL_CHANGE = include_damping_in_qmodel_change


class KfacOptimizer(tf.train.GradientDescentOptimizer):
  """The KFAC Optimizer (https://arxiv.org/abs/1503.05671)."""

  def __init__(self,
               learning_rate,
               damping,
               layer_collection,
               cov_ema_decay=0.95,
               var_list=None,
               momentum=0.9,
               momentum_type="adam",
               use_weight_decay=False,
               weight_decay_coeff=0.1,
               qmodel_update_rescale=None,
               norm_constraint=None,
               name="KFAC",
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               batch_size=None,
               placement_strategy=None,
               compute_params_stats=False,
               adapt_damping=False,
               update_damping_immediately=False,
               is_chief=True,
               prev_train_batch=None,
               loss=None,
               loss_fn=None,
               min_damping=1e-8,
               damping_adaptation_decay=0.95,
               damping_adaptation_interval=5,
               use_passed_loss=True,
               train_batch=None,
               print_logs=False,
               tf_replicator=None,
               **kwargs):
    """Initializes the K-FAC optimizer with the given settings.

      NOTE: this is a base class for K-FAC optimizers that offers full control
      over the execution of K-FAC's various ops.  For a more fool-proof /
      automated version see for example PeriodicInvCovUpdateKfacOpt.

      Also, please keep in mind that while the K-FAC code loosely conforms to
      TensorFlow's Optimizer API it can't be used naively as a "drop in
      replacement" for basic classes like MomentumOptimizer.  Using it
      properly with SyncReplicasOptimizer, for example, requires special care.
      When using it with Distribution Strategy, unlike common practice, K-FAC
      expects an unscaled loss tensor (i.e. not scaled by
      1.0 / global_batch_size like you may see in TF Distribution Strategy
      tutorials). Regardles of whether you are using estimator, strategy, or
      a normal custom training loop, you should pass in the same loss.

      See the various examples in the "examples" directory for a guide about
      how to use K-FAC in various contexts and various systems, like
      TF-Estimator. See in particular the "convnet" example.  google/examples
      also contains an example using TPUEstimator.

    Args:
      learning_rate: float or 0D Tensor. The base learning rate for the
          optimizer. Must be set to None if using one of the 'qmodel'
          momentum_type values.
      damping: float or 0D Tensor. This quantity times the identity matrix is
          (approximately) added to the curvature matrix (i.e. the Fisher or GGN)
          before it is inverted multiplied by the gradient when computing the
          (raw) update. This quantity should match the scale of the objective,
          so that if you put a multiplier on your loss you should apply the
          same multiplier to the damping. Roughly speaking, larger values
          constrain the update vector to a smaller region around zero, which
          we want to do when our local quadratic model is a less trustworthy
          local approximation of the true objective.  The damping value is
          closely related to the trust region radius and to the classical
          Tikhonov regularization method. If the `adapt_damping` argument is
          True then this value is used only as an initial value for the
          adaptation method.
      layer_collection: The layer collection object, which holds the Fisher
          blocks, Kronecker factors, and losses associated with the
          graph.  The layer_collection cannot be modified after KfacOptimizer's
          initialization.
      cov_ema_decay: The decay factor used when calculating the
          covariance estimate moving averages. (Default: 0.95)
      var_list: Optional list or tuple of variables to train. Defaults to
          tf.trainable_variables.
      momentum: The momentum decay constant to use. Only applies when
          momentum_type is 'regular' or 'adam'. (Default: 0.9)
      momentum_type: The type of momentum to use in this optimizer, one of
          'regular', 'adam', 'qmodel', or 'qmodel_fixedmu'. 'regular' gives
          standard momentum. 'adam' gives a style of momentum reminisent
          of the Adam method, which seems to work better in practice.
          'qmodel' makes the optimizer perform automatic control of both the
          learning rate and momentum using a quadratic model based method
          (see _compute_qmodel_hyperparams for more details). 'qmodel_fixedmu'
          is similar to 'qmodel' but only controls the learning rate.
          (Default: 'adam')
      use_weight_decay: If True, explicit "weight decay" is performed by K-FAC.
          Note that this is distinct from L2 regularization, and corresponds to
          optimizing a regularized version of the loss passed to minimize(),
          where the regularization term added is related to the "Fisher-Rao
          norm". See https://openreview.net/pdf?id=B1lz-3Rct7 for more details.
          Note that using this feature won't change the loss function you pass
          to minimize(), and thus the loss you report will not correspond
          precisely to what K-FAC is optimizing. (Default: False)
      weight_decay_coeff: The coefficient to use for weight decay (see above).
          (Default: 0.1)
      qmodel_update_rescale: float or None.  An additional multiplier to apply
          to the update computed by the quadratic model based adjustment
          methods. If None it will behave like a value of 1.0. (Default: None)
      norm_constraint: float or Tensor. If specified, the update is scaled down
          so that its approximate squared Fisher norm v^T F v is at most the
          specified value. May only be used with momentum type 'regular'.  See
          the docstring for the method _clip_updates() for a more detailed
          explanation of this feature. (Default: None)
      name: The name for this optimizer. (Default: 'KFAC')
      estimation_mode: The type of estimator to use for the Fishers/GGNs. Can be
          'gradients', 'empirical', 'curvature_prop', 'curvature_prop_GGN',
          'exact', or 'exact_GGN'. See the doc-string for FisherEstimator
          (in estimator.py) for more a more detailed description of these
          options. (Default: 'gradients').
      colocate_gradients_with_ops: Whether we should request gradients we
          compute in the estimator be colocated with their respective ops.
          (Default: True)
      batch_size: The size of the mini-batch. Only needed when `momentum_type`
          == 'qmodel' or when `compute_params_stats` is True. Note that when
          using data parallelism where the model graph and optimizer are
          replicated across multiple devices, this should be the per-replica
          batch size. An example of this is sharded data on the TPU, where
          batch_size should be set to the total batch size divided by the
          number of shards. (Default: None)
      placement_strategy: string or None. Device placement strategy used when
          creating variables, and various ops. Can be None, 'round_robin', or
          'replica_round_robin'. 'round_robin' supports round-robin placement of
          various ops on lists of provided devices. 'replica_round_robin' does
          something similar but over shards/replicas instead, and only works
          in certain 'replicated' contexts (e.g. TPUEstimator).  The details of
          the different placement strategies are controlled by additional
          keyword arguments that can be passed to this class, and which are
          described in the different placement mixin classes in placement.py.
          (Default: None)
      compute_params_stats: Bool. If True, we compute the first order version
          of the statistics computed to estimate the Fisher/GGN. These
          correspond to the `variables` method in a one-to-one fashion.  They
          are available via the `params_stats` property.  When estimation_mode
          is 'empirical', this will correspond to the standard parameter
          gradient on the loss. (Default: False)
      adapt_damping: `Boolean`. If True we adapt the damping according to the
          Levenberg-Marquardt rule described in Section 6.5 of the original
          K-FAC paper. The details of this scheme are controlled by various
          additional arguments below. Also some of these arguments are extra
          pieces of information, such as the loss, needed by the method. Note
          that unless using a convenience subclass like
          PeriodicInvCovUpdateKfacOpt the damping adaptation op must be
          executed by the user (like the cov and inv ops). This op is returned
          by the maybe_pre_update_adapt_damping() method. (Default: False)
      update_damping_immediately: Damping adjustment strategy. If True then the
          damping is updated in the same optimizer minimize call as
          `(step+1) % damping_adaptation_interval == 0`, immediately after the
          parameter update is performed. If False then the damping is updated
          in the next step. If True then it is assumed that the apply_gradients
          op will safely update the model before returning; it is recommended
          to only resource variables in this case. Note that there is odd
          behavior currently being investigated when this is true and running on
          TPUs and TPUConfig.iterations_per_loop > 1. (Default: False)
      is_chief: `Boolean`, `True` if the worker is chief. (Default: True)
      prev_train_batch: Training mini-batch used in the previous step. This
          will be used to evaluate loss by calling `loss_fn(prev_train_batch)`
          when damping adaptation is used. (Default: None)
      loss: `Tensor` the model loss, used as the pre-update loss in adaptive
          damping. Also used for the built-in log printing. When using
          Distribution Strategy, unlike common Distribution Strategy practice,
          this loss tensor should NOT be scaled by 1.0 / global_batch_size.
          (Default: None)
      loss_fn: `function` that takes as input training data tensor and returns
          a scalar loss. Only needed if using damping adaptation. When using
          Distribution Strategy, unlike common Distribution Strategy practice,
          this loss function's output should NOT be scaled by
          1.0 / global_batch_size. (Default: None)
      min_damping: `float`, Minimum value the damping parameter
          can take. This should be at least as big as the L2 regularization
          coefficient. (Default: 1e-8)
      damping_adaptation_decay: `float`, The `damping` parameter is
          multiplied by the `damping_adaptation_decay` every
          `damping_adaptation_interval` number of iterations. (Default: 0.99)
      damping_adaptation_interval: `int`, Number of steps in between
          updating the `damping` parameter. Note that damping is adapted at
          the step where (step+1) % damping_adaptation_interval == 0,
          (or immediately at the start of the next step by
          maybe_pre_update_adapt_damping() if update_damping_immediately is
          False). (Default: 5)
      use_passed_loss: `Boolean`. If True we use the loss tensor passed in by
          the user (via minimze() or compute_gradients() or the set_loss()
          method) in damping adaptation scheme, instead of calling loss_fn()
          a second time for this. This is more efficient but may not always be
          desired. (Default: True)
      train_batch: Training mini-batch used in the current step. This
          will be used to evaluate loss by calling `loss_fn(train_batch)`
          when damping adaptation is used and `use_passed_loss` is False.
          (Default: None)
      print_logs: `Boolean`. If True, we print some logging info using
          tf.print after each iteration. This is done in the method
          _maybe_print_logging_info, which we encourage you to modify in order
          to add whatever you want. (Default: False)
      tf_replicator: A Replicator object or None. If not None, K-FAC will set
          itself up to work inside of the provided TF-Replicator object.
          (Default: None)
      **kwargs: Arguments to be passed to specific placement strategy mixin.
          Check `placement.RoundRobinPlacementMixin` for example.

    Raises:
      ValueError: If the momentum type is unsupported.
      ValueError: If clipping is used with momentum type other than 'regular'.
      ValueError: If no losses have been registered with layer_collection.
      ValueError: If momentum is non-zero and momentum_type is not 'regular'
          or 'adam'.
    """
    self._layers = layer_collection

    self._colocate_gradients_with_ops = colocate_gradients_with_ops

    momentum_type = momentum_type.lower()
    legal_momentum_types = ["regular", "adam", "qmodel", "qmodel_fixedmu"]

    if momentum_type not in legal_momentum_types:
      raise ValueError("Unsupported momentum type {}. Must be one of {}."
                       .format(momentum_type, legal_momentum_types))
    if momentum_type not in ["regular", "adam"] and norm_constraint is not None:
      raise ValueError("Update clipping is only supported with momentum "
                       "type 'regular' and 'adam'.")
    if momentum_type == "qmodel" and momentum is not None:
      raise ValueError("Momentum must be None if using a momentum_type "
                       "'qmodel'.")
    self._momentum_type = momentum_type
    self._momentum = momentum

    self._use_weight_decay = use_weight_decay
    self._weight_decay_coeff = weight_decay_coeff

    self._norm_constraint = norm_constraint
    self._batch_size = batch_size
    self._placement_strategy = placement_strategy

    # Damping adaptation parameters
    self._adapt_damping = adapt_damping

    if self._adapt_damping:
      with tf.variable_scope(name):
        self._damping = tf.get_variable(
            "damping", initializer=damping, trainable=False,
            use_resource=True)
    else:
      self._damping = tf.convert_to_tensor(damping)

    self._update_damping_immediately = update_damping_immediately
    self._is_chief = is_chief
    self._prev_train_batch = prev_train_batch
    self._loss_tensor = loss
    self._loss_fn = loss_fn
    self._damping_adaptation_decay = damping_adaptation_decay
    self._damping_adaptation_interval = damping_adaptation_interval
    self._omega = (
        self._damping_adaptation_decay**self._damping_adaptation_interval)
    self._min_damping = min_damping
    self._use_passed_loss = use_passed_loss
    if not use_passed_loss and train_batch is None:
      raise ValueError("Must pass in train_batch if used_passed_loss is false.")

    self._train_batch = train_batch

    self._print_logs = print_logs

    if self._momentum_type.startswith("qmodel"):
      if learning_rate is not None:
        raise ValueError("'learning_rate' must be set to None if using one of "
                         "the 'qmodel' momentum types.")
      if qmodel_update_rescale is not None:
        learning_rate = qmodel_update_rescale
      else:
        learning_rate = 1.0
    else:
      if learning_rate is None:
        raise ValueError("'learning_rate' must *not* be set to None unless "
                         "using one of the 'qmodel' momentum types.")
      if qmodel_update_rescale is not None:
        raise ValueError("'qmodel_update_rescale' must be None unless using "
                         "one of the 'qmodel' momentum types.")
    self._qmodel_update_rescale = qmodel_update_rescale

    with tf.variable_scope(name):
      # We store rho only for possible logging purposes.
      self._rho = tf.get_variable(
          "rho", initializer=float("nan"), dtype=tf.float32, trainable=False,
          use_resource=True)
      self._prev_loss = tf.get_variable(
          "prev_loss", initializer=float("nan"), dtype=tf.float32,
          trainable=False, use_resource=True)
      self._qmodel_learning_rate = tf.get_variable(
          "qmodel_learning_rate", initializer=float("nan"), dtype=tf.float32,
          trainable=False, use_resource=True)
      self._qmodel_momentum = tf.get_variable(
          "qmodel_momentum", initializer=float("nan"), dtype=tf.float32,
          trainable=False, use_resource=True)
      self._qmodel_change = tf.get_variable(
          "qmodel_change", initializer=float("nan"), dtype=tf.float32,
          trainable=False, use_resource=True)

      self._counter = tf.get_variable(
          "counter", dtype=tf.int64, shape=(), trainable=False,
          initializer=tf.zeros_initializer, use_resource=True,
          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

      variables = var_list or tf.trainable_variables()

      if tf_replicator is not None:
        def _get_sanitized_name(var_name):
          return re.sub(r"replica_\d+", "", var_name)

        # This tells K-FAC's libraries that we are using TF-Replicator with this
        # particular Replicator object.
        utils.set_global_constants(tf_replicator=tf_replicator)

        # We need to sanitize the names of the variables that K-FAC creates
        # so they are the same between replicas.
        ff.set_global_constants(get_sanitized_name_fn=_get_sanitized_name)

      self._fisher_est = est.make_fisher_estimator(
          placement_strategy=placement_strategy,
          variables=variables,
          cov_ema_decay=cov_ema_decay,
          damping=self._damping,
          layer_collection=self.layers,
          exps=(-1,),
          estimation_mode=estimation_mode,
          colocate_gradients_with_ops=self._colocate_gradients_with_ops,
          compute_params_stats=compute_params_stats,
          batch_size=batch_size,
          **kwargs)

    super(KfacOptimizer, self).__init__(learning_rate, name=name)

  def get_cov_vars(self):
    """Returns all covaraiance varaiables."""
    return self._fisher_est.get_cov_vars()

  def get_inv_vars(self):
    """Returns all inverse computation related varaiables."""
    return self._fisher_est.get_inv_vars()

  @property
  def factors(self):
    return self._fisher_est.factors

  @property
  def registered_variables(self):
    return self._fisher_est.variables

  @property
  def layers(self):
    return self._layers

  @property
  def mat_type(self):
    return self._fisher_est.mat_type

  @property
  def damping(self):
    return tf.identity(self._damping)

  @property
  def damping_adaptation_interval(self):
    return self._damping_adaptation_interval

  @property
  def learning_rate(self):
    if self._momentum_type.startswith("qmodel"):
      return self._learning_rate * tf.identity(self._qmodel_learning_rate)
    else:
      return tf.convert_to_tensor(self._learning_rate)

  @property
  def momentum(self):
    if self._momentum_type.startswith("qmodel"):
      return tf.identity(self._qmodel_momentum)
    else:
      return tf.convert_to_tensor(self._momentum)

  @property
  def rho(self):
    return tf.identity(self._rho)

  @property
  def qmodel_change(self):
    return tf.identity(self._qmodel_change)

  @property
  def counter(self):
    return tf.identity(self._counter)

  @property
  def params_stats(self):
    return self._fisher_est.params_stats

  def set_loss(self, loss):
    # Use this method if you have overridden both the minimize method and
    # compute_gradients method but still want K-FAC to know the loss value
    # (which is required for damping adaptation).
    self._loss_tensor = loss

  def _maybe_print_logging_info(self):
    if not self._print_logs:
      return tf.no_op()

    p = []
    p.append(("=========================================================",))
    p.append(("Iteration:", self.counter))
    p.append(("mini-batch loss =", self._loss_tensor))
    p.append(("learning_rate =", self.learning_rate, "| momentum =",
              self.momentum))
    p.append(("damping =", self.damping, "| rho =", self.rho,
              "| qmodel_change =", self.qmodel_change))
    p.append(("=========================================================",))

    return utils.multiline_print(p)

  def make_vars_and_create_op_thunks(self):
    """Make vars and create op thunks.

    Returns:
      cov_update_thunks: List of cov update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
      inv_update_thunks: List of inv update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
    """
    scope = self.get_name() + "/" + self._fisher_est.name
    return self._fisher_est.make_vars_and_create_op_thunks(scope=scope)

  def create_ops_and_vars_thunks(self):
    """Create thunks that make the ops and vars on demand.

    This function returns 4 lists of thunks: cov_variable_thunks,
    cov_update_thunks, inv_variable_thunks, and inv_update_thunks.

    The length of each list is the number of factors and the i-th element of
    each list corresponds to the i-th factor (given by the "factors" property).

    Note that the execution of these thunks must happen in a certain
    partial order.  The i-th element of cov_variable_thunks must execute
    before the i-th element of cov_update_thunks (and also the i-th element
    of inv_update_thunks).  Similarly, the i-th element of inv_variable_thunks
    must execute before the i-th element of inv_update_thunks.

    TL;DR (oversimplified): Execute the thunks according to the order that
    they are returned.

    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """
    scope = self.get_name() + "/" + self._fisher_est.name
    return self._fisher_est.create_ops_and_vars_thunks(scope=scope)

  def check_var_list(self, var_list):
    if set(var_list) != set(self.registered_variables):
      raise ValueError("var_list doesn't match with set of Fisher-estimating "
                       "variables (i.e. those that were registered).")

  @staticmethod
  def _scale_loss(loss_value):
    # tf.train.Optimizer uses this method to account for the Estimator +
    # Distribution Strategy (DS) case. DS wants a scaled loss and to aggregate
    # gradients via a sum. Estimator provides an unscaled loss by default. So,
    # this method would divide the loss by num_replicas. For our optimizer, we
    # require users to pass in an unscaled loss, so we do not want this method
    # to alter Estimator's input when it's used with DS.
    return loss_value

  def minimize(self,
               loss,
               global_step=None,
               var_list=None,
               gate_gradients=tf.train.Optimizer.GATE_OP,
               aggregation_method=None,
               colocate_gradients_with_ops=True,
               name=None,
               grad_loss=None,
               **kwargs):
    # This method has the same general arguments as the minimize methods in
    # standard optimizers do.
    # With most optimizers used with Distribution Strategy (DS), the user is
    # expected to scale their loss by 1.0 / global_batch_size, then DS
    # aggregates the gradients via a sum. We expect users to pass in a loss that
    # is NOT scaled. This is so we can handle the Estimator and DS cases in a
    # consistent way. As a side effet, this means each replica must have the
    # same per-replica batch size.

    if var_list is None:
      var_list = self.registered_variables
    else:
      self.check_var_list(var_list)

    return super(KfacOptimizer, self).minimize(
        loss,
        global_step=global_step,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        name=name,
        grad_loss=grad_loss,
        **kwargs)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=True,
                        grad_loss=None,
                        **kwargs):
    # This method has the same general arguments as the minimize methods in
    # standard optimizers do. Unlike the compute_gradient method for typical
    # optimizer implementations, this one performs cross-replica syncronization
    # automatically when under one the supported replicated contexts, and so
    # use of things like CrossShardOptimizer is unessesary (and wasteful).

    if var_list is not None:
      self.check_var_list(var_list)

    grads_and_vars = super(KfacOptimizer, self).compute_gradients(
        loss=loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss,
        **kwargs)

    # When using the TF Keras fused BatchNormalization implementation, in some
    # cases the gradient shape is ?. KFAC needs the gradient shape in at least
    # two cases: when registering a layer as generic, and when computing the
    # qmodel. The gradient should have the same shape as the variable, so when
    # any dimension is None we set the shape ourselves.
    for grad, var in grads_and_vars:
      if len(grad.shape) and not all(grad.shape.as_list()):
        grad.set_shape(var.shape)

    grads, vars_ = list(zip(*grads_and_vars))
    grads = utils.all_average(grads)

    return tuple(zip(grads, vars_))

  def _is_damping_adaptation_time(self):
    # Note that we update damping at the step right before the end of the
    # interval, instead of at the beginning of the next interval. This is
    # so it properly lines up with the periodic inverse updates (i.e. happens
    # immediately before them.)
    return tf.equal(tf.mod(self.counter + 1,
                           self._damping_adaptation_interval),
                    0)

  def _is_just_after_damping_adaptation_time(self):
    return tf.equal(tf.mod(self.counter,
                           self._damping_adaptation_interval),
                    0)

  def _maybe_update_prev_loss(self):
    if self._adapt_damping:
      should_update_prev_loss = self._is_damping_adaptation_time()

      def update_prev_loss():
        loss = self._loss_tensor if self._use_passed_loss else self._loss_fn(
            self._train_batch)
        loss = utils.all_average(loss)
        return tf.group(utils.smart_assign(self._prev_loss, loss))

      maybe_update_prev_loss_op = tf.cond(
          should_update_prev_loss,
          update_prev_loss,
          tf.no_op)

      return maybe_update_prev_loss_op
    else:
      return tf.no_op()

  def maybe_pre_update_adapt_damping(self):
    """Maybe adapt the damping according to the built-in scheme.

    Unless using a convenience class like PeriodicInvCovUpdateKfacOpt the op
    returned by this function should be run every sess.run call, preferably
    before the inv ops (using a control dependency).

    Returns:
      An op that applies the specified gradients, and also updates the counter
      variable.
    """
    if (not self._adapt_damping or not self._is_chief or
        self._update_damping_immediately):
      return tf.no_op()

    # We update the damping on the iteration that is technically after
    # where we compute qmodel_change.  However, it should happen before
    # anything else does, so it's as if we computed it on the previous
    # iteration.  The only reason we do it this way and not on the
    # actual iteration is due to weirdness related to parameter servers
    # or possibly just non-resource variables. Essentially, the model
    # variables won't be updated and so we can't properly compute
    # prev_batch_loss until the next sess.run() call.
    should_update_damping = self._is_just_after_damping_adaptation_time()

    maybe_update_damping = tf.cond(
        should_update_damping,
        self._update_damping,
        tf.no_op)
    return maybe_update_damping

  def _maybe_post_update_adapt_damping(self):
    if not self._update_damping_immediately or not self._adapt_damping:
      return tf.no_op()

    should_update_damping = self._is_damping_adaptation_time()

    maybe_update_damping = tf.cond(
        should_update_damping,
        self._update_damping,
        tf.no_op)
    return maybe_update_damping

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    """Applies gradients to variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      *args: Additional arguments for super.apply_gradients.
      **kwargs: Additional keyword arguments for super.apply_gradients.

    Returns:
      An op that applies the specified gradients, and also updates the counter
      variable.
    """
    maybe_update_prev_loss = self._maybe_update_prev_loss()

    with tf.control_dependencies([maybe_update_prev_loss]):
      # In Python 3, grads_and_vars can be a zip() object which can only be
      # iterated over once. By converting it to a list, we ensure that it can be
      # iterated over more than once.
      grads_and_vars = list(grads_and_vars)

      with tf.variable_scope(self.get_name()):
        # Compute raw update step (self._learning_rate not yet applied).
        # Note that this function also updates the velocity vectors.
        raw_updates_and_vars = self._compute_raw_update_steps(grads_and_vars)

      if self._use_weight_decay:
        raw_updates_and_vars = self._add_weight_decay(raw_updates_and_vars)

      if tf.distribute.has_strategy():
        # Distribution Strategy (DS) expects users to pass in loss /
        # global_batch_size to minimize. We require users not to do this, so our
        # code can consistently deal with input in the single device, Estimator,
        # and DS cases. However, the _distributed_apply call in
        # super(...).apply_gradients(...) will perform a sum over replicas to
        # aggregate the gradients. Therefore, we divide by the number of
        # replicas so the gradient applied to the variables is correct.
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        raw_updates_and_vars = [(update/num_replicas, var)
                                for update, var in raw_updates_and_vars]

      # Update trainable variables with this step, applying self._learning_rate.
      apply_op = super(KfacOptimizer, self).apply_gradients(
          raw_updates_and_vars, *args, **kwargs)

      with tf.control_dependencies([apply_op]):
        maybe_post_update_damping_op = self._maybe_post_update_adapt_damping()

        with tf.control_dependencies([maybe_post_update_damping_op]):
          maybe_print_logging_info = self._maybe_print_logging_info()
          with tf.control_dependencies([maybe_print_logging_info]):
            # Update the main counter
            return tf.group(self._counter.assign(self._counter + 1))

  def _add_weight_decay(self, grads_and_vars):
    """Applies weight decay.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      List of (gradient, variable) pairs.
    """
    return [(grad + self._weight_decay_coeff * tf.stop_gradient(var), var)
            for grad, var in grads_and_vars]

  def _squared_fisher_norm(self, grads_and_vars, precon_grads_and_vars):
    """Computes the squared (approximate) Fisher norm of the updates.

    This is defined as v^T F v, where F is the approximate Fisher matrix
    as computed by the estimator, and v = F^{-1} g, where g is the gradient.
    This is computed efficiently as v^T g.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._multiply_preconditioner`
        on `grads_and_vars`.

    Returns:
      Scalar representing the squared norm.

    Raises:
      ValueError: if the two list arguments do not contain the same variables,
        in the same order.
    """
    return ip_p(grads_and_vars, precon_grads_and_vars)

  def _update_clip_coeff(self, grads_and_vars, precon_grads_and_vars):
    """Computes the scale factor for the update to satisfy the norm constraint.

    Defined as min(1, sqrt(c / r^T F r)), where c is the norm constraint,
    F is the approximate Fisher matrix, and r is the update vector, i.e.
    -alpha * v, where alpha is the learning rate, and v is the preconditioned
    gradient.

    This is based on Section 5 of Ba et al., Distributed Second-Order
    Optimization using Kronecker-Factored Approximations. Note that they
    absorb the learning rate alpha (which they denote eta_max) into the formula
    for the coefficient, while in our implementation, the rescaling is done
    before multiplying by alpha. Hence, our formula differs from theirs by a
    factor of alpha.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._multiply_preconditioner`
        on `grads_and_vars`.

    Returns:
      Scalar representing the coefficient which should be applied to the
      preconditioned gradients to satisfy the norm constraint.
    """
    sq_norm_grad = self._squared_fisher_norm(grads_and_vars,
                                             precon_grads_and_vars)
    sq_norm_up = sq_norm_grad * self._learning_rate**2
    return tf.minimum(
        tf.ones(shape=(), dtype=sq_norm_up.dtype),
        tf.sqrt(self._norm_constraint / sq_norm_up))

  def _clip_updates(self, grads_and_vars, precon_grads_and_vars):
    """Rescales the preconditioned gradients to satisfy the norm constraint.

    Rescales the preconditioned gradients such that the resulting update r
    (after multiplying by the learning rate) will satisfy the norm constraint.
    This constraint is that r^T F r <= C, where F is the approximate Fisher
    matrix, and C is the norm_constraint attribute. See Section 5 of
    Ba et al., Distributed Second-Order Optimization using Kronecker-Factored
    Approximations.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      precon_grads_and_vars: List of (preconditioned gradient, variable) pairs.
        Must be the result of calling `self._multiply_preconditioner`
        on `grads_and_vars`.

    Returns:
      List of (rescaled preconditioned gradient, variable) pairs.
    """
    coeff = self._update_clip_coeff(grads_and_vars, precon_grads_and_vars)
    return sprod_p(coeff, precon_grads_and_vars)

  def _compute_prev_updates(self, variables):
    """Returns the previous update vector computed using the quadratic model.

    Note that this vector does not include any additional scaling that may have
    been applied after the quadratic model optimization (i.e. the quantity
    returned by self.learning_rate).

    Note that this may not actually be the previous update if
    momentum_type="adam".

    Args:
      variables: List of variables for which to compute the previous update.

    Returns:
      List of (previous_update, variable) pairs in the same order as
      `variables`.
    """
    # What guarantee do we have that this is the old value and not the
    # new value?  Remember that control flow doesn't work in TF whenever
    # non-resource variables are involved.
    # TODO(b/121245468): Figure out if this is a problem and if not explain why
    # Or fix it by somehow forcing the slots to use resource variables instead.

    prev_updates = sprod(
        -1., tuple(self._zeros_slot(var, "velocity", self.get_name())
                   for var in variables))
    return tuple(zip(prev_updates, variables))

  def _compute_qmodel(self,
                      raw_updates_and_vars,
                      prev_updates_and_vars,
                      grads_and_vars):
    """Computes the 2 dimensional version of the (exact) quadratic model.

       The two dimesions are the update and the previous update vectors.

       The arguments are all lists of (Tensor, Variable) pairs where the
       variables are the same and in the same order.

    Args:
      raw_updates_and_vars: a list of (precond grad, variable) pairs. Raw update
        proposal to apply to the variables (before scaling by learning rate and
        addition of velocity/momentum).
      prev_updates_and_vars: a list of (previos update, variable) pairs.
        Previous update applied to the variables (includes learning rate and
        velocity/momentum).
      grads_and_vars: a list of (gradient, variable) pairs. Gradients for the
        parameters and the variables that the updates are being applied to. The
        order of this list must correspond to the order of the other arguments.
        (Note that this function doesn't actually apply the update.)

    Returns:
      m, c, and b. m is the 2 by 2 matrix representing the quadratic term,
      c is a 2 by 1 vector representing the linear term, and b is the 2 by 2
      matrix representing only the contribution of the damping to the quadratic
      term. These are all multi-dimensional lists (lists of lists) of Tensors.
    """

    # Raw update proposal to apply to the variables (before scaling by learning
    # rate and addition of velocity/momentum).
    raw_updates, _ = zip(*raw_updates_and_vars)
    prev_updates, _ = zip(*prev_updates_and_vars)
    grads, variables = zip(*grads_and_vars)

    utils.assert_variables_match_pairs_list(
        raw_updates_and_vars, prev_updates_and_vars,
        error_message="_compute_qmodel raw_updates_and_vars and "
        "prev_updates_and_vars differ.")
    utils.assert_variables_match_pairs_list(
        prev_updates_and_vars, grads_and_vars,
        error_message="_compute_qmodel prev_updates_and_vars and "
        "grads_and_vars differ.")

    cmvpc = cmvp.CurvatureMatrixVectorProductComputer(
        self.layers,
        variables,
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)

    # Compute the matrix-vector products with the transposed Fisher factor
    # (or GGN factor)
    if self.mat_type == "Fisher":
      mft_updates = cmvpc.multiply_fisher_factor_transpose(raw_updates)
      mft_prev_updates = cmvpc.multiply_fisher_factor_transpose(prev_updates)
    elif self.mat_type == "GGN" or self.mat_type == "Empirical_Fisher":
      mft_updates = cmvpc.multiply_ggn_factor_transpose(raw_updates)
      mft_prev_updates = cmvpc.multiply_ggn_factor_transpose(prev_updates)

    batch_size = tf.cast(self._batch_size, dtype=mft_updates[0].dtype)

    b_11 = self.damping * ip(raw_updates, raw_updates)
    b_21 = self.damping * ip(prev_updates, raw_updates)
    b_22 = self.damping * ip(prev_updates, prev_updates)
    b = [[b_11, b_21], [b_21, b_22]]

    # Compute the entries of the 2x2 matrix
    m_11 = ip(mft_updates, mft_updates) / batch_size
    m_21 = ip(mft_prev_updates, mft_updates) / batch_size
    m_22 = (ip(mft_prev_updates, mft_prev_updates)
            / batch_size)
    m = [[m_11 + b_11, m_21 + b_21],
         [m_21 + b_21, m_22 + b_22]]

    m = utils.all_average(m)

    c_1 = ip(grads, raw_updates)
    c_2 = ip(grads, prev_updates)

    c = [[c_1], [c_2]]

    return m, c, b

  def _compute_qmodel_hyperparams(self, m, c, b, fixed_mu=None):
    """Compute optimal update hyperparameters from the quadratic model.

    More specifically, if L is the loss we minimize a quadratic approximation
    of L(theta + d) which we denote by qmodel(d) with
    d = alpha*precon_grad + mu*prev_update with respect to alpha and mu, where

      qmodel(d) = (1/2) * d^T * C * d + grad^T*d + L(theta) .

    Unlike in the KL clipping approach we use the non-approximated quadratic
    model where the curvature matrix C is the true Fisher (or GGN) on the
    current mini-batch (computed without any approximations beyond mini-batch
    sampling), with the usual Tikhonov damping/regularization applied,

      C = F + damping * I

    See Section 7 of https://arxiv.org/abs/1503.05671 for a derivation of
    the formula.  See Appendix C for a discussion of the trick of using
    a factorized Fisher matrix to more efficiently compute the required
    vector-matrix-vector products.

    Args:
      m: 2 by 2 matrix representing the quadratic term (a list of list of
        0D Tensors)
      c: a 2 by 1 vector representing the linear term (a list of 0D Tensors)
      b: 2 by 2 matrix representing only the contribution of the damping to the
        quadratic term
      fixed_mu: A fixed value of mu to use instead of the optimal one.
        (Default: None)
    Returns:
      (alpha, mu, qmodel_change), where alpha and mu are chosen to optimize the
      quadratic model, and
      qmodel_change = qmodel(alpha*precon_grad + mu*prev_update) - qmodel(0)
                    = qmodel(alpha*precon_grad + mu*prev_update) - L(theta).
    """

    def non_zero_prevupd_case():
      r"""Computes optimal (alpha, mu) given non-zero previous update.

      We solve the full 2x2 linear system. See Martens & Grosse (2015),
      Section 7, definition of $\alpha^*$ and $\mu^*$.

      Returns:
        (alpha, mu, qmodel_change), where alpha and mu are chosen to optimize
        the quadratic model, and
        qmodel_change = qmodel(alpha*precon_grad + mu*prev_update) - qmodel(0).
      """
      if fixed_mu is None:
        sol = -1. * _two_by_two_solve(m, c)
        alpha = sol[0, 0]
        mu = sol[1, 0]

        if self._qmodel_update_rescale is None:
          # This is a special formula that takes advantage of the particular
          # relationship of sol to m and c. It should be equivalent to
          # _eval_quadratic(m, c, sol) if everything is working properly.
          qmodel_change = 0.5 * tf.reduce_sum(sol * c)
        else:
          sol = self._qmodel_update_rescale * sol
          qmodel_change = _eval_quadratic(m, c, sol)

        # Subtract out the damping-related penalty
        if not _INCLUDE_DAMPING_IN_QMODEL_CHANGE:
          qmodel_change -= _eval_quadratic_no_c(b, sol)

      else:
        alpha = -1. * (fixed_mu * m[0][1] + c[0][0]) / (m[0][0])
        mu = fixed_mu

        sol = [[alpha], [mu]]

        if self._qmodel_update_rescale is not None:
          sol = self._qmodel_update_rescale * tf.convert_to_tensor(sol)

        qmodel_change = _eval_quadratic(m, c, sol)

        # Subtract out the damping-related penalty
        if not _INCLUDE_DAMPING_IN_QMODEL_CHANGE:
          qmodel_change -= _eval_quadratic_no_c(b, sol)

      return tf.squeeze(alpha), tf.squeeze(mu), tf.squeeze(qmodel_change)

    def zero_prevupd_case():
      r"""Computes optimal (alpha, mu) given all-zero previous update.

      The linear system reduces to 1x1. See Martens & Grosse (2015),
      Section 6.4, definition of $\alpha^*$.

      Returns:
        (alpha, 0.0, qmodel_change), where alpha is chosen to optimize the
        quadratic model, and
        qmodel_change = qmodel(alpha*precon_grad) - qmodel(0)
      """
      alpha = -c[0][0] / m[0][0]
      if fixed_mu is None:
        mu = 0.0
      else:
        mu = fixed_mu

      if self._qmodel_update_rescale is None:
        # This is a special formula that takes advantage of the particular
        # relationship of sol to m and c.
        qmodel_change = 0.5 * alpha * c[0][0]

        # Subtract out the damping-related penalty
        if not _INCLUDE_DAMPING_IN_QMODEL_CHANGE:
          qmodel_change -= 0.5 * tf.square(alpha) * b[0][0]
      else:
        sol = self._qmodel_update_rescale * alpha
        qmodel_change = 0.5 * m[0][0] * tf.square(sol) + c[0][0] * sol
        # Subtract out the damping-related penalty
        if not _INCLUDE_DAMPING_IN_QMODEL_CHANGE:
          qmodel_change -= 0.5 * tf.square(sol) * b[0][0]

      return alpha, mu, qmodel_change

    return tf.cond(
        tf.equal(c[1][0], 0.0),
        zero_prevupd_case,
        non_zero_prevupd_case)

  def _compute_approx_qmodel_change(self, updates_and_vars, grads_and_vars):
    """Computes the change in the approximate quadratic model.

    'Approximate' means the quadratic model which uses the approximate
    Fisher/GGN as the curvature matrix, instead of the exact Fisher/GGN which
    is used by _compute_qmodel and its dependent methods.

    Args:
      updates_and_vars: List of (update, variable) pairs.
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      A 0D Tensor which is the change in the approximate quadratic model.
    """

    quad_term = 0.5*ip_p(updates_and_vars,
                         self._fisher_est.multiply(updates_and_vars))

    if not _INCLUDE_DAMPING_IN_QMODEL_CHANGE:
      # This isn't quite right, but doing it properly is too awkward.
      quad_term = quad_term - 0.5*self.damping*ip_p(updates_and_vars,
                                                    updates_and_vars)
    linear_term = ip_p(updates_and_vars, grads_and_vars)

    return quad_term + linear_term

  def _maybe_update_qmodel_change(self, qmodel_change_thunk):
    """Returns an op which updates the qmodel_change variable if it is time to.

    Args:
      qmodel_change_thunk: A callable which when evaluated returns the qmodel
        change.

    Returns:
      An op.
    """
    def update_qmodel_change():
      # The tf.group is needed to strip away the value so it can be used
      # in the cond later.
      return tf.group(utils.smart_assign(self._qmodel_change,
                                         tf.squeeze(qmodel_change_thunk())))

    # Note that we compute the qmodel change and store it in a variable so
    # it can be used at the next sess.run call (where rho will actually be
    # computed).
    return tf.cond(self._is_damping_adaptation_time(),
                   update_qmodel_change, tf.no_op)

  def _multiply_preconditioner(self, vecs_and_vars):
    return self._fisher_est.multiply_inverse(vecs_and_vars)

  def _get_qmodel_quantities(self, grads_and_vars):

    # Compute "preconditioned gradient".
    precon_grads_and_vars = self._multiply_preconditioner(grads_and_vars)

    var_list = tuple(var for (_, var) in grads_and_vars)
    prev_updates_and_vars = self._compute_prev_updates(var_list)

    # While it might seem like this call performs needless computations
    # involving prev_updates_and_vars in the case where it is zero, because
    # we extract out only the part of the solution that is not zero the rest
    # of it will not actually be computed by TensorFlow (I think).
    m, c, b = self._compute_qmodel(
        precon_grads_and_vars, prev_updates_and_vars, grads_and_vars)

    return precon_grads_and_vars, m, c, b

  def _compute_raw_update_steps(self, grads_and_vars):
    """Computes the raw update steps for the variables given the gradients.

    Note that these "raw updates" are further multiplied by
    -1*self._learning_rate when the update is eventually applied in the
    superclass (which is GradientDescentOptimizer).

    Args:
      grads_and_vars: List of (gradient, variable) pairs.

    Returns:
      A list of tuples (raw_update, var) where raw_update is the update to
      the parameter. These updates must be actually used since they carry
      with them certain control dependencies that need to happen.
    """

    if self._momentum_type == "regular":
      # Compute "preconditioned" gradient.
      precon_grads_and_vars = self._multiply_preconditioner(grads_and_vars)

      # Apply "KL clipping" if asked for.
      if self._norm_constraint is not None:
        precon_grads_and_vars = self._clip_updates(grads_and_vars,
                                                   precon_grads_and_vars)

      # Update the velocities and get their values as the "raw" updates
      raw_updates_and_vars = self._update_velocities(precon_grads_and_vars,
                                                     self._momentum)

      if self._adapt_damping and self._is_chief:

        def compute_qmodel_change():
          updates_and_vars = sprod_p(-1. * self._learning_rate,
                                     raw_updates_and_vars)
          return self._compute_approx_qmodel_change(updates_and_vars,
                                                    grads_and_vars)

        maybe_update_qmodel_change = self._maybe_update_qmodel_change(
            compute_qmodel_change)

        with tf.control_dependencies([maybe_update_qmodel_change]):
          # Making this a tuple is important so that it actually gets evaluated
          # in the context.
          return tuple((tf.identity(vec), var)
                       for (vec, var) in raw_updates_and_vars)
      else:
        return raw_updates_and_vars

    elif self._momentum_type == "adam":
      velocities_and_vars = self._update_velocities(grads_and_vars,
                                                    self._momentum)
      # The "preconditioned" velocity vector is the raw update step.
      raw_updates_and_vars = self._multiply_preconditioner(velocities_and_vars)

      # Apply "KL clipping" if asked for. Note that we are applying this to
      # the combined preconditioned gradient + velocity, unlike for the
      # momentum_type = 'regular' case.
      if self._norm_constraint is not None:
        raw_updates_and_vars = self._clip_updates(velocities_and_vars,
                                                  raw_updates_and_vars)

      if self._adapt_damping and self._is_chief:
        def compute_qmodel_change():
          # This is a special formula that exploits the structure of the
          # particular update we are using.  Note that this is using the approx
          # Fisher as defined by the inverses, which might be stale (perhaps so
          # stale that they are using an old damping value, which may mess up
          # the damping adaptation method).
          return (0.5 * (self._learning_rate**2) *
                  ip_p(raw_updates_and_vars, velocities_and_vars) -
                  self._learning_rate * ip_p(raw_updates_and_vars,
                                             grads_and_vars))

        maybe_update_qmodel_change = self._maybe_update_qmodel_change(
            compute_qmodel_change)

        with tf.control_dependencies([maybe_update_qmodel_change]):
          # Making this a tuple is important so that it actually gets evaluated
          # in the context.
          return tuple((tf.identity(vec), var)
                       for (vec, var) in raw_updates_and_vars)
      else:
        return raw_updates_and_vars

    elif (self._momentum_type == "qmodel"
          or self._momentum_type == "qmodel_fixedmu"):

      precon_grads_and_vars, m, c, b = self._get_qmodel_quantities(
          grads_and_vars)

      if self._momentum_type == "qmodel_fixedmu":
        fixed_mu = self._momentum
      else:
        fixed_mu = None

      # Compute optimal velocity update parameters according to quadratic
      # model
      alpha, mu, qmodel_change = self._compute_qmodel_hyperparams(
          m, c, b, fixed_mu=fixed_mu)

      qmodel_assign_op = tf.group(
          utils.smart_assign(self._qmodel_change, qmodel_change),
          utils.smart_assign(self._qmodel_learning_rate, -alpha),
          utils.smart_assign(self._qmodel_momentum, mu))

      with tf.control_dependencies([qmodel_assign_op]):
        return self._update_velocities(
            precon_grads_and_vars, mu, vec_coeff=-alpha)

  # NOTE: the very particular way this function is written is probably important
  # for it to work correctly with non-resource variables, which are very
  # unpredictable with regards to control flow.
  def _update_velocities(self, vecs_and_vars, decay, vec_coeff=1.0):
    """Updates the velocities of the variables with the given vectors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      decay: How much to decay the old velocity by.  This is often referred to
        as the 'momentum constant'.
      vec_coeff: Coefficient to apply to the vectors before adding them to the
        velocity.

    Returns:
      A list of (velocity, var) indicating the new velocity for each var.
    """
    def _update_velocity(vec, var):
      velocity = self._zeros_slot(var, "velocity", self.get_name())
      with tf.colocate_with(velocity):
        # NOTE(mattjj): read/modify/write race condition not suitable for async.

        # Compute the new velocity for this variable.
        new_velocity = decay * velocity + vec_coeff * vec

        # Save the updated velocity.
        return (tf.identity(utils.smart_assign(velocity, new_velocity)), var)

    # Go through variable and update its associated part of the velocity vector.
    return [_update_velocity(vec, var) for vec, var in vecs_and_vars]

  def _get_current_loss(self):
    if self._update_damping_immediately:
      return utils.all_average(self._loss_fn(self._train_batch))

    return utils.all_average(self._loss_fn(self._prev_train_batch))

  def _get_prev_loss(self):
    return self._prev_loss

  def _update_damping(self):
    """Adapts damping parameter. Check KFAC paper (Section 6.5) for the details.

    The damping parameter is updated according to the Levenberg-Marquardt rule
    every `self._damping_adaptation_interval` iterations.

    Essentially, the rule computes the reduction ratio "rho" and depending on
    the value either increases lambda, decreases it, or leaves it as is.

    The reduction ratio captures how closely the quadratic approximation to the
    loss function approximates the actual loss within a trust region. The
    damping update tries to make the damping as small as possible while
    maintaining the property that the quadratic model remains a good local
    approximation to the loss function.

    Returns:
      An Op to assign newly computed damping value to `self._damping`, and also
      updates the _rho member.
    """
    prev_loss = self._get_prev_loss()
    current_loss = self._get_current_loss()

    loss_change = current_loss - prev_loss
    rho = loss_change / self._qmodel_change

    should_decrease = tf.math.logical_or(
        tf.math.logical_and(loss_change < 0, self._qmodel_change > 0),
        rho > 0.75)
    should_increase = rho < 0.25

    new_damping = tf.case(
        [(should_decrease, lambda: self.damping * self._omega),
         (should_increase, lambda: self.damping / self._omega)],
        default=lambda: self.damping)

    new_damping = tf.maximum(new_damping, self._min_damping)

    return tf.group(utils.smart_assign(self._damping, new_damping),
                    utils.smart_assign(self._rho, rho))


def _two_by_two_solve(m, vec):
  """Solve a 2x2 system by direct inversion.

  Args:
    m: A length 2 list of length 2 lists, is a 2x2 matrix of [[a, b], [c, d]].
    vec: The length 2 list of length 1 lists, a vector of [e, f].

  Returns:
    matmul(m^{-1}, vec).
  """
  a = m[0][0]
  b = m[0][1]
  c = m[1][0]
  d = m[1][1]
  inv_m_det = 1.0 / (a * d - b * c)
  m_inverse = [
      [d * inv_m_det, -b * inv_m_det],
      [-c * inv_m_det, a * inv_m_det]
  ]
  return tf.matmul(m_inverse, vec)


def _eval_quadratic_no_c(m, vec):
  return 0.5*tf.matmul(tf.matmul(vec, m, transpose_a=True), vec)


def _eval_quadratic(m, c, vec):
  return _eval_quadratic_no_c(m, vec) + tf.matmul(c, vec, transpose_a=True)
