# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
               **kwargs):
    """Initializes PeriodicInvCovUpdateKfacOptimizer.

    Args:
      invert_every: int, The inversion ops are run once every `invert_every`
        calls to optimizer.minimize, (Default: 10)
      cov_update_every: int, The covariance update ops are run once every
        `covariance_update_every` calls to optimizer.minimize. (Default: 1)
      **kwargs: Arguments to `tensorflow_kfac.KfacOptimizer` class.
    """
    self._invert_every = invert_every
    self._cov_update_every = cov_update_every
    # kwargs contains argument for a specifc placement strategy.
    super(PeriodicInvCovUpdateKfacOpt, self).__init__(**kwargs)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=True,
                        grad_loss=None):
    return super(PeriodicInvCovUpdateKfacOpt, self).compute_gradients(
        loss=loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    with tf.control_dependencies([self.kfac_update_ops()]):
      return super(PeriodicInvCovUpdateKfacOpt, self).apply_gradients(
          grads_and_vars=grads_and_vars,
          global_step=global_step,
          name=name)

  def kfac_update_ops(self):
    """Sets up the KFAC factor update ops.

    Constructs the covariance and inverse ops, builds counter variables for
    them, and then sets them up to run only every self._cov_update_every and
    self._invert_every calls.

    Returns:
      An op that when run will run the update ops at their update frequencies.
    """
    with tf.variable_scope(self.get_name()):
      (cov_update_thunks,
       inv_update_thunks) = self.make_vars_and_create_op_thunks()
      counter = self.counter
      prev_counter = tf.assign(
          tf.get_variable(
              "prev_counter", dtype=tf.int64, shape=(), trainable=False,
              initializer=tf.zeros_initializer, use_resource=True),
          counter)
      with tf.control_dependencies([prev_counter]):
        update_counter = tf.assign_add(counter, 1, name="update_counter")
        # GPU doesn't support mod so we allow TF to allocate this op
        # automatically.
        with tf.device(None):
          should_do_cov_updates = tf.equal(tf.mod(prev_counter,
                                                  self._cov_update_every), 0)
        maybe_cov_updates = tf.cond(
            should_do_cov_updates,
            lambda: tf.group([thunk() for thunk in cov_update_thunks]),
            tf.no_op)
      with tf.control_dependencies([maybe_cov_updates, update_counter]):
        # GPU doesn't support mod so we allow TF to allocate this op
        # automatically.
        with tf.device(None):
          should_do_inv_updates = tf.equal(tf.mod(prev_counter,
                                                  self._invert_every), 0)
        maybe_inv_updates = tf.cond(
            should_do_inv_updates,
            lambda: tf.group([thunk() for thunk in inv_update_thunks]),
            tf.no_op)
        return maybe_inv_updates

  @property
  def counter(self):
    if not hasattr(self, "_counter"):
      with tf.variable_scope("periodic_counter", reuse=tf.AUTO_REUSE):
        self._counter = tf.get_variable(
            "counter", dtype=tf.int64, shape=(), trainable=False,
            initializer=tf.zeros_initializer, use_resource=True)

    return self._counter
