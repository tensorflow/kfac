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
"""Tests for l.d.tf.optimizers.python.AsyncInvCovUpdateKfacOpt class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import sonnet as snt
import tensorflow as tf

from kfac.python.ops import layer_collection as lc
from kfac.python.ops.kfac_utils import async_inv_cov_update_kfac_opt as ak
from kfac.python.ops.tensormatch import graph_search as gs


_BATCH_SIZE = 256


def _construct_layer_collection(layers, all_logits, var_list):
  for idx, logits in enumerate(all_logits):
    tf.logging.info("Registering logits: %s", logits)
    with tf.variable_scope(tf.get_variable_scope(), reuse=(idx > 0)):
      layers.register_categorical_predictive_distribution(
          logits, name="register_logits")
  batch_size = all_logits[0].shape.as_list()[0]
  vars_to_register = var_list if var_list else tf.trainable_variables()
  gs.register_layers(layers, vars_to_register, batch_size)


class AsyncInvCovUpdateKfacOpt(tf.test.TestCase):

  def test_train(self):
    image = tf.random_uniform(shape=(_BATCH_SIZE, 784), maxval=1.)
    labels = tf.random_uniform(shape=(_BATCH_SIZE,), maxval=10, dtype=tf.int32)
    labels_one_hot = tf.one_hot(labels, 10)

    model = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP([128, 128, 10])])
    logits = model(image)
    all_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels_one_hot)
    loss = tf.reduce_mean(all_losses)
    layers = lc.LayerCollection()
    optimizer = ak.AsyncInvCovUpdateKfacOpt(
        inv_devices=["/cpu:0"],
        cov_devices=["/cpu:0"],
        learning_rate=1e-4,
        cov_ema_decay=0.95,
        damping=1e+3,
        layer_collection=layers,
        momentum=0.9)
    _construct_layer_collection(layers, [logits], tf.trainable_variables())
    train_step = optimizer.minimize(loss)
    target_loss = 0.05
    max_iterations = 500

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      optimizer.run_cov_inv_ops(sess)
      for _ in xrange(max_iterations):
        loss_, _ = sess.run([loss, train_step])
        if loss_ < target_loss:
          break
      optimizer.stop_cov_inv_ops(sess)

if __name__ == "__main__":
  tf.test.main()
