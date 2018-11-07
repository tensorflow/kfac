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
"""Implementation of KFAC to aggreagte cov statistics over multiple runs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

from kfac.python.ops import optimizer
from kfac.python.ops import utils


class KfacMultiRunOpt(optimizer.KfacOptimizer):
  """Accumulates data across multiple sessions and applies updates periodically.

  The class provides functionality to process large minibatch for training. The
  statistics and gradients for the large batch are collected over multiple
  optimizer.minimize runs and applied periodically.

  More specifically, this class aggregates covariance statistics over
  `num_steps_per_cov_update` runs. The aggregated covariance estimate is then
  assigned to KFAC covariance variable. Also the gradients are accumulated over
  `num_steps_per_update` runs and then applied. The inverses will be computed
  every `invert_every` steps. Ideally set the `invert_every` to a multiple of
  `num_steps_per_cov_update` so that the inverses are computed after the
  covariance is updated. The higher the multiple more the delay in using the
  computed covariance estimates in the KFAC update step.
  """

  def __init__(self,
               invert_every=10,
               **kwargs):
    """Initializes KfacMultiRunOpt.

    Args:
      invert_every: int, The inversion ops are run once every `invert_every`
        calls to optimizer.minimize, (Default: 10)
      **kwargs: Arguments to `tensorflow_kfac.KfacOptimizer` class.
    """
    self._invert_every = invert_every
    super(KfacMultiRunOpt, self).__init__(**kwargs)

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=True,
                        grad_loss=None):
    return super(KfacMultiRunOpt, self).compute_gradients(
        loss=loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    cov_update_thunks, inv_update_thunks = self.make_vars_and_create_op_thunks()

    # Create Accumulator variables for adding gradients over multiple sess.run.
    self._acc_grads = [
        utils.AccumulatorVariable(
            name="grad_acc_var_{}".format(i),
            acc_var_shape=grads.shape,
            acc_var_dtype=grads.dtype)
        for i, (grads, _) in enumerate(grads_and_vars)
    ]

    counter = self.counter
    prev_counter = tf.assign(
        tf.get_variable(
            "prev_counter",
            shape=(),
            dtype=tf.int64,
            initializer=tf.zeros_initializer,
            trainable=False), counter)
    with tf.control_dependencies([prev_counter]):
      update_counter = tf.assign_add(counter, 1, name="update_counter")

    # Covarainces are computed every run and stored in accumuator variables.
    cov_updates = tf.group([thunk() for thunk in cov_update_thunks])
    with tf.control_dependencies([cov_updates, update_counter]):
      # GPU doesn't support mod so we allow TF to allocate this op
      # automatically.
      with tf.device(None):
        should_do_inv_updates = tf.logical_and(
            prev_counter > 0,
            tf.equal(tf.mod(prev_counter, self._invert_every), 0))
      maybe_inv_updates = tf.cond(
          should_do_inv_updates,
          lambda: tf.group([thunk() for thunk in inv_update_thunks]), tf.no_op)
      with tf.control_dependencies([maybe_inv_updates]):
        with tf.device(None):
          maybe_apply_grads = tf.logical_and(
              prev_counter > 0,
              tf.equal(tf.mod(prev_counter, self._num_steps_per_cov_update), 0))

        def apply_grads():
          acc_vars = [var for _, var in grads_and_vars]
          acc_grads_and_vars = [
              (acc_grad.accumulated_value, var)
              for acc_grad, var in zip(self._acc_grads, acc_vars)
          ]
          return super(KfacMultiRunOpt, self).apply_gradients(
              grads_and_vars=acc_grads_and_vars,
              global_step=global_step,
              name=name)

        def update_acc_grads():
          grads_list = [grads for grads, _ in grads_and_vars]
          return [
              acc_grad.accumulate(
                  grad, num_steps_for_update=self._num_steps_per_cov_update)
              for acc_grad, grad in zip(self._acc_grads, grads_list)
          ]

        with tf.control_dependencies(update_acc_grads()):
          return tf.cond(maybe_apply_grads, apply_grads, tf.no_op)

  @property
  def counter(self):
    if not hasattr(self, "_counter"):
      with tf.variable_scope("periodic_counter", reuse=tf.AUTO_REUSE):
        self._counter = tf.get_variable(
            "counter",
            shape=(),
            dtype=tf.int64,
            initializer=tf.zeros_initializer,
            trainable=False)

    return self._counter
