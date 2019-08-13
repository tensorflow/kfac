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
"""Implementation of KFAC which runs covariance and inverse ops periodically.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
# Dependency imports

import tensorflow as tf

from kfac.python.ops import optimizer


class PeriodicInvCovUpdateKfacOpt(optimizer.KfacOptimizer):
  """Provides functionality to run covariance and inverse ops periodically.

  Creates KFAC optimizer with a `placement strategy`.
  Also runs the covariance and inverse ops periodically. The base class
  does not provide a mechanism to automatically construct and run the covariance
  and inverse ops, they must be created and run manually using
  make_vars_and_create_op_thunks or create_ops_and_vars_thunks. This class
  provides functionality to create these ops and runs them periodically whenever
  optimizer.minimize op is run.

  The inverse ops are run `invert_every` iterations and covariance statistics
  are updated `cov_update_every` iterations. Ideally set
  the `invert_every` to a multiple of `cov_update_every` so that the
  inverses are computed after the covariance is updated. The higher the multiple
  more the delay in using the computed covariance estimates in the KFAC update
  step. Also computing the statistics and inverses periodically saves on
  computation cost and a "reasonable" value often does not show any perforamnce
  degradation compared to computing these quantitites every iteration.
  """

  def __init__(self,
               invert_every=10,
               cov_update_every=1,
               num_burnin_steps=0,
               **kwargs):
    """Initializes a PeriodicInvCovUpdateKfacOptimizer object.

    See the docstring for `KfacOptimizer` class (in optimizer.py) for
    complete list of arguments (there are many!).

    Please keep in mind that while the K-FAC code loosely conforms to
    TensorFlow's Optimizer API, it can't be used naively as a "drop in
    replacement" for basic classes like MomentumOptimizer.  Using it
    properly with SyncReplicasOptimizer, for example, requires special care.

    See the various examples in the "examples" directory for a guide about
    how to use K-FAC in various contexts and various systems, like
    TF-Estimator. See in particular the convnet example.  google/examples
    also contains an example using TPUEstimator.

    Note that not all use cases will work with
    PeriodicInvCovUpdateKfacOptimizer. Sometimes you will have to use the base
    KfacOptimizer which provides more fine-grained control over ops.  Other
    times you might want to use one of the other subclassed optimizers like
    AsyncInvCovUpdateKfacOpt.

    Args:
      invert_every: int. The inversion ops are run once every `invert_every`
        executions of the training op. (Default: 10)
      cov_update_every: int. The 'covariance update ops' are run once every
        `covariance_update_every` executions of the training op. (Default: 1)
      num_burnin_steps: int. For the first `num_burnin_steps` steps the
        optimizer will only perform cov updates. Note: this doesn't work with
        CrossShardOptimizer since the custom minimize method implementation
        will be ignored. (Default: 0)
      **kwargs: Arguments to `KfacOptimizer` class.
    """

    if "cov_ema_decay" in kwargs:
      kwargs["cov_ema_decay"] = kwargs["cov_ema_decay"]**cov_update_every

    super(PeriodicInvCovUpdateKfacOpt, self).__init__(**kwargs)

    self._invert_every = invert_every
    self._cov_update_every = cov_update_every
    self._num_burnin_steps = num_burnin_steps

    self._made_vars_already = False

    if self._adapt_damping:
      if self._damping_adaptation_interval % self._invert_every != 0:
        raise ValueError("damping_adaptation_interval must be divisible by "
                         "invert_every.")

    with tf.variable_scope(self.get_name()):
      self._burnin_counter = tf.get_variable(
          "burnin_counter", dtype=tf.int64, shape=(), trainable=False,
          initializer=tf.zeros_initializer, use_resource=True,
          aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

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

    cov_update_thunks, _ = self.make_vars_and_create_op_thunks()
    self._made_vars_already = True

    def update_cov_and_burnin_counter():
      cov_update = tf.group(*(thunk(ema_decay=1.0,
                                    should_write=True)
                              for thunk in cov_update_thunks))

      burnin_counter_update = self._burnin_counter.assign(
          self._burnin_counter + 1)

      return tf.group(cov_update, burnin_counter_update)

    def super_minimize():
      return super(PeriodicInvCovUpdateKfacOpt, self).minimize(
          loss,
          global_step=global_step,
          var_list=var_list,
          gate_gradients=gate_gradients,
          aggregation_method=aggregation_method,
          colocate_gradients_with_ops=colocate_gradients_with_ops,
          name=name,
          grad_loss=grad_loss,
          **kwargs)

    return tf.cond(self._burnin_counter < self._num_burnin_steps,
                   update_cov_and_burnin_counter, super_minimize)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    with tf.control_dependencies([self.kfac_update_ops()]):
      return super(PeriodicInvCovUpdateKfacOpt, self).apply_gradients(
          grads_and_vars=grads_and_vars,
          global_step=global_step,
          name=name)

  def kfac_update_ops(self):
    """Sets up the KFAC factor update ops.

    Returns:
      An op that when run will run the update ops at their update frequencies.
    """
    # This if-statement is a trick/hack to maintain compatibility with
    # CrossShardOptimizer or other optimizers that might not call our
    # custom minimize() method (that would otherwise always make the variables).
    if not self._made_vars_already:
      (cov_update_thunks,
       inv_update_thunks) = self.make_vars_and_create_op_thunks()
      warnings.warn("It looks like apply_gradients() was called before "
                    "minimze() was called. This is not recommended, and you "
                    "should avoid using optimizer wrappers like "
                    "CrossShardOptimizer with K-FAC that try to bypass the "
                    "minimize() method. The burn-in feature won't work when "
                    "the class is used this way, for example.")
    else:
      (_, cov_update_thunks,
       _, inv_update_thunks) = self.create_ops_and_vars_thunks()

    should_do_cov_updates = tf.equal(tf.mod(self.counter,
                                            self._cov_update_every), 0)
    maybe_cov_updates = tf.cond(
        should_do_cov_updates,
        lambda: tf.group(*(thunk() for thunk in cov_update_thunks)),
        tf.no_op)

    maybe_pre_update_adapt_damping = self.maybe_pre_update_adapt_damping()
    with tf.control_dependencies([maybe_cov_updates,
                                  maybe_pre_update_adapt_damping]):
      should_do_inv_updates = tf.equal(tf.mod(self.counter,
                                              self._invert_every), 0)
      maybe_inv_updates = tf.cond(
          should_do_inv_updates,
          lambda: tf.group(*(thunk() for thunk in inv_update_thunks)),
          tf.no_op)
      return maybe_inv_updates

