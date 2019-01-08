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
  parameter gradients, and the second order statistics (aka 'cov' statistics)
  needed to compute the preconditioner, are collected over multiple executions
  (steps) of the training op (returned by optimizer.minimize). Periodically,
  these accumulated statistics and gradients are then used to actually update
  the preconditioner, and apply an update to the parameters.

  The best way to think about it is that the only the executions that update
  the parameters are the real ``iterations".  All other executions are just
  doing the prep-work for those iterations.

  In more detail, each execution of the training op (returned by minimize) will
  only accumulate gradient information and cov information from the current
  mini-batch, and not actually apply any of this to the params or cov variables.
  This is except for the k-th, 2k-th, 3k-th etc execution of the
  training op, where k = `num_steps_per_update`.

  Inversion ops are executed periodically every j-th update to the parameters /
  cov variables, where j = `invert_every`. So this means that they will execute
  on the jk-th, 2jk-th, 3jk-th, etc execution of the training op.
  """

  def __init__(self,
               num_steps_per_update=1,
               invert_every=10,
               **kwargs):
    """Initializes KfacMultiRunOpt.

    See the docstring for `KfacOptimizer` class (in optimizer.py) for
    complete list of arguments (there are many!).  Note that the argument
    `num_steps_per_cov_update` will be overridden by the value of
    `num_steps_per_update`.

    Args:
      num_steps_per_update: int. The number of steps per update of the second
        order statistics and parameters. (Default: 1)
      invert_every: int. The frequency (as measured by the number of parameter
        updates -- not the number of training op executions) at which the
        inversion ops executed. (Default: 10)
      **kwargs: Arguments to `KfacOptimizer` class.
    """
    self._num_steps_per_update = num_steps_per_update
    self._invert_every = invert_every

    kwargs["num_steps_per_cov_update"] = self._num_steps_per_update
    super(KfacMultiRunOpt, self).__init__(**kwargs)

    with tf.variable_scope(self.get_name()):
      self._inner_counter = tf.get_variable(
          "inner_counter", dtype=tf.int64, shape=(), trainable=False,
          initializer=tf.zeros_initializer, use_resource=True)

    if self._adapt_damping:
      raise ValueError("Damping adaptation currently not supported with "
                       "KfacMultiRunOpt.")

  @property
  def inner_counter(self):
    return tf.identity(self._inner_counter)

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

    should_update = tf.equal(tf.mod(self.inner_counter + 1,
                                    self._num_steps_per_update),
                             0)

    # Covariances are computed every run and stored in accumuator variables.
    cov_updates = tf.group(*(thunk() for thunk in cov_update_thunks))

    with tf.control_dependencies([cov_updates]):

      should_do_inv_updates = tf.logical_and(
          should_update,
          tf.equal(tf.mod(self.counter, self._invert_every), 0))

      maybe_inv_updates = tf.cond(
          should_do_inv_updates,
          lambda: tf.group(*(thunk() for thunk in inv_update_thunks)),
          tf.no_op)

      with tf.control_dependencies([maybe_inv_updates]):

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

        grads_list = [grads for grads, _ in grads_and_vars]
        update_acc_grads = [
            acc_grad.accumulate(
                grad, num_steps_for_update=self._num_steps_per_update)
            for acc_grad, grad in zip(self._acc_grads, grads_list)
        ]

        with tf.control_dependencies(update_acc_grads):

          maybe_apply_grads = tf.cond(should_update, apply_grads, tf.no_op)

          with tf.control_dependencies([maybe_apply_grads]):
            return self._inner_counter.assign(self._inner_counter + 1)
