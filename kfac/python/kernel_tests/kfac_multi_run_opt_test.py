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
"""Tests for l.d.tf.optimizers.python.KfacMultiRunOpt class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import sonnet as snt
import tensorflow as tf

from kfac.python.ops import layer_collection
from kfac.python.ops.kfac_utils import kfac_multi_run_opt

_BATCH_SIZE = 128


def _construct_layer_collection(layers, logits):
  layers.register_categorical_predictive_distribution(
      logits, name="register_logits")
  layers.auto_register_layers()


class KfacMultiRunOptTest(tf.test.TestCase):

  def test_train(self):
    image = tf.random_uniform(shape=(_BATCH_SIZE, 784), maxval=1.)
    labels = tf.random_uniform(shape=(_BATCH_SIZE,), maxval=10, dtype=tf.int32)
    labels_one_hot = tf.one_hot(labels, 10)

    model = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP([128, 128, 10])])
    logits = model(image)
    all_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_one_hot)
    loss = tf.reduce_mean(all_losses)
    layers = layer_collection.LayerCollection()
    _construct_layer_collection(layers, logits)

    num_steps_per_update = 5
    optimizer = kfac_multi_run_opt.KfacMultiRunOpt(
        invert_every=10,
        num_steps_per_update=num_steps_per_update,
        learning_rate=1e-4,
        cov_ema_decay=0.95,
        damping=1e+2,
        layer_collection=layers,
        momentum=0.9,
        placement_strategy="round_robin")

    train_step = optimizer.minimize(loss)
    inner_counter = optimizer.inner_counter
    counter = optimizer.counter
    num_inner_steps = num_steps_per_update
    num_outer_steps = 100

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)

      for outer_step in range(num_outer_steps):
        for inner_step in range(num_inner_steps):

          counter_ = sess.run(counter)
          inner_counter_ = sess.run(inner_counter)

          self.assertEqual(counter_, outer_step)
          self.assertEqual(inner_counter_,
                           inner_step + outer_step*num_inner_steps)

          sess.run([loss, train_step])


if __name__ == "__main__":
  tf.test.main()
