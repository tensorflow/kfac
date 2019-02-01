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
"""Tests for autoencoder_mnist.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from kfac.examples import autoencoder_mnist


class ConvNetTest(tf.test.TestCase):

  def testAutoEncoder(self):
    self.assertTrue(tf.test.is_built_with_cuda())
    (train_op, _, batch_loss, batch_error, batch_size_schedule,
     batch_size) = autoencoder_mnist.construct_train_quants()
    config = tf.ConfigProto(allow_soft_placement=True)
    init_op = tf.global_variables_initializer()
    with self.test_session(config=config, use_gpu=True) as sess:
      sess.run(init_op)
      for step in range(700):
        batch_size_ = batch_size_schedule[min(step,
                                              len(batch_size_schedule) - 1)]
        _, batch_loss_, batch_error_ = sess.run(
            [train_op, batch_loss, batch_error],
            feed_dict={batch_size: batch_size_})

      self.assertLessEqual(batch_loss_, 50.)
      self.assertLessEqual(batch_error_, 0.75)


if __name__ == '__main__':
  tf.test.main()
