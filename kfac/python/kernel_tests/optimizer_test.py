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
"""Tests for kfac.optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from kfac.python.ops import fisher_factors as ff
from kfac.python.ops import layer_collection as lc
from kfac.python.ops import optimizer

# We need to set these constants since the numerical values used in the tests
# were chosen when these used to be the defaults.
ff.set_global_constants(init_covariances_at_zero=False,
                        zero_debias=False,
                        init_inverses_at_zero=False)


def dummy_layer_collection():
  lcoll = lc.LayerCollection()
  dummy = tf.constant([1., 2.])
  lcoll.register_categorical_predictive_distribution(logits=dummy)
  return lcoll


class OptimizerTest(tf.test.TestCase):

  def testOptimizerInitInvalidMomentumRegistration(self):
    with self.assertRaises(ValueError):
      optimizer.KfacOptimizer(
          0.1, 0.2, lc.LayerCollection(), 0.3, momentum_type='foo')

  def testOptimizerInit(self):
    with tf.Graph().as_default():
      layer_collection = lc.LayerCollection()

      inputs = tf.ones((2, 1)) * 2
      weights_val = np.ones((1, 1), dtype=np.float32) * 3.
      weights = tf.get_variable('w', initializer=tf.constant(weights_val))
      bias = tf.get_variable(
          'b', initializer=tf.zeros_initializer(), shape=(1, 1))
      output = tf.matmul(inputs, weights) + bias

      layer_collection.register_fully_connected((weights, bias), inputs, output)

      logits = tf.tanh(output)
      targets = tf.constant([[0.], [1.]])
      output = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=targets))

      layer_collection.register_categorical_predictive_distribution(logits)

      optimizer.KfacOptimizer(
          0.1,
          0.2,
          layer_collection,
          0.3,
          momentum=0.5,
          momentum_type='regular')

  def testSquaredFisherNorm(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      grads_and_vars = [(tf.constant([[1., 2.], [3., 4.]]), None),
                        (tf.constant([[2., 3.], [4., 5.]]), None)]
      pgrads_and_vars = [(tf.constant([[3., 4.], [5., 6.]]), None),
                         (tf.constant([[7., 8.], [9., 10.]]), None)]
      opt = optimizer.KfacOptimizer(0.1, 0.2, dummy_layer_collection(), 0.3)
      sq_norm = opt._squared_fisher_norm(grads_and_vars, pgrads_and_vars)
      self.assertAlmostEqual(174., sess.run(sq_norm), places=5)

  def testUpdateClipCoeff(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      grads_and_vars = [(tf.constant([[1., 2.], [3., 4.]]), None),
                        (tf.constant([[2., 3.], [4., 5.]]), None)]
      pgrads_and_vars = [(tf.constant([[3., 4.], [5., 6.]]), None),
                         (tf.constant([[7., 8.], [9., 10.]]), None)]
      lrate = 0.1

      # Note: without rescaling, the squared Fisher norm of the update
      # is 1.74

      # If the update already satisfies the norm constraint, there should
      # be no rescaling.
      opt = optimizer.KfacOptimizer(
          lrate, 0.2, dummy_layer_collection(), 0.3, norm_constraint=10.,
          name='KFAC_1')
      coeff = opt._update_clip_coeff(grads_and_vars, pgrads_and_vars)
      self.assertAlmostEqual(1., sess.run(coeff), places=5)

      # If the update violates the constraint, it should be rescaled to
      # be on the constraint boundary.
      opt = optimizer.KfacOptimizer(
          lrate, 0.2, dummy_layer_collection(), 0.3, norm_constraint=0.5,
          name='KFAC_2')
      coeff = opt._update_clip_coeff(grads_and_vars, pgrads_and_vars)
      sq_norm_pgrad = opt._squared_fisher_norm(grads_and_vars, pgrads_and_vars)
      sq_norm_update = lrate**2 * coeff**2 * sq_norm_pgrad
      self.assertAlmostEqual(0.5, sess.run(sq_norm_update), places=5)

  def testUpdateVelocities(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      layers = lc.LayerCollection()
      layers.register_categorical_predictive_distribution(tf.constant([1.0]))
      opt = optimizer.KfacOptimizer(
          0.1, 0.2, layers, 0.3, momentum=0.5, momentum_type='regular')
      x = tf.get_variable('x', initializer=tf.ones((2, 2)))
      y = tf.get_variable('y', initializer=tf.ones((2, 2)) * 2)
      vec1 = tf.ones((2, 2)) * 3
      vec2 = tf.ones((2, 2)) * 4

      model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      update_op = opt._update_velocities([(vec1, x), (vec2, y)], 0.5)
      opt_vars = [
          v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
          if v not in model_vars
      ]

      sess.run(tf.global_variables_initializer())
      old_opt_vars = sess.run(opt_vars)

      # Optimizer vars start out at 0.
      for opt_var in old_opt_vars:
        self.assertAllEqual(sess.run(tf.zeros_like(opt_var)), opt_var)

      sess.run(update_op)
      new_opt_vars = sess.run(opt_vars)
      # After one update, the velocities are equal to the vectors.
      for vec, opt_var in zip([vec1, vec2], new_opt_vars):
        self.assertAllEqual(sess.run(vec), opt_var)

      sess.run(update_op)
      final_opt_vars = sess.run(opt_vars)
      for first, second in zip(new_opt_vars, final_opt_vars):
        self.assertFalse(np.equal(first, second).all())

  def testApplyGradients(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      layer_collection = lc.LayerCollection()

      inputs = tf.ones((2, 1)) * 2
      weights_val = np.ones((1, 1), dtype=np.float32) * 3.
      weights = tf.get_variable('w', initializer=tf.constant(weights_val))
      bias = tf.get_variable(
          'b', initializer=tf.zeros_initializer(), shape=(1, 1))
      output = tf.matmul(inputs, weights) + bias

      layer_collection.register_fully_connected((weights, bias), inputs, output)

      logits = tf.tanh(output)
      targets = tf.constant([[0.], [1.]])
      output = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=targets))

      layer_collection.register_categorical_predictive_distribution(logits)

      opt = optimizer.KfacOptimizer(
          0.1,
          0.2,
          layer_collection,
          0.3,
          momentum=0.5,
          momentum_type='regular')
      (cov_update_thunks,
       inv_update_thunks) = opt.make_vars_and_create_op_thunks()
      cov_update_ops = tuple(thunk() for thunk in cov_update_thunks)
      inv_update_ops = tuple(thunk() for thunk in inv_update_thunks)

      grads_and_vars = opt.compute_gradients(output, [weights, bias])
      all_vars = [grad_and_var[1] for grad_and_var in grads_and_vars]

      op = opt.apply_gradients(grads_and_vars)

      sess.run(tf.global_variables_initializer())
      old_vars = sess.run(all_vars)
      sess.run(cov_update_ops)
      sess.run(inv_update_ops)
      sess.run(op)
      new_vars = sess.run(all_vars)

      for old_var, new_var in zip(old_vars, new_vars):
        self.assertNotEqual(old_var, new_var)

if __name__ == '__main__':
  tf.test.main()
