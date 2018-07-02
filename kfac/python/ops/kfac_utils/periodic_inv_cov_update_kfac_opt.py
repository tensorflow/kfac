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

import tensorflow as tf

from kfac.python.ops import optimizer


class PeriodicInvCovUpdateKfacOpt(optimizer.KfacOptimizer):
  """Provides functionality to run covariance and inverse ops periodically.

  Creates KFAC optimizer with a `placement strategy`.
  Also runs the covariance and inverse ops periodically.
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
    cov_update_thunks, inv_update_thunks = self.make_vars_and_create_op_thunks()
    counter = self.counter
    prev_counter = tf.assign(
        tf.get_variable(
            "prev_counter", shape=(), initializer=tf.zeros_initializer),
        counter)
    with tf.control_dependencies([prev_counter]):
      update_counter = tf.assign_add(counter, 1, name="update_counter")
      do_cov_updates = tf.cond(
          tf.equal(tf.mod(prev_counter, self._cov_update_every), 0),
          lambda: tf.group([thunk() for thunk in cov_update_thunks]),
          tf.no_op)
    with tf.control_dependencies([do_cov_updates, update_counter]):
      do_inv_updates = tf.cond(
          tf.equal(tf.mod(prev_counter, self._invert_every), 0),
          lambda: tf.group([thunk() for thunk in inv_update_thunks]), tf.no_op)
      with tf.control_dependencies([do_inv_updates]):
        return super(PeriodicInvCovUpdateKfacOpt, self).apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=global_step,
            name=name)

  @property
  def counter(self):
    if not hasattr(self, "_counter"):
      with tf.variable_scope("periodic_counter", reuse=tf.AUTO_REUSE):
        self._counter = tf.get_variable(
            "counter", shape=(), initializer=tf.zeros_initializer)

    return self._counter
