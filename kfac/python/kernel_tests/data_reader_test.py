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
"""Tests for CachedDataReader class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
import tensorflow as tf

from kfac.python.ops.kfac_utils import data_reader


class DataReaderTest(tf.test.TestCase):

  def test_read_batch(self):
    max_batch_size = 10
    batch_size_schedule = [2, 4, 6, 8]
    data_set = tf.random_uniform(shape=(max_batch_size, 784), maxval=1.)
    var_data = data_reader.CachedDataReader(
        (data_set,), max_batch_size)
    cur_batch_size = tf.placeholder(
        shape=(), dtype=tf.int32, name='cur_batch_size')
    # Force create the ops
    data = var_data(cur_batch_size)[0]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      for batch_size in batch_size_schedule:
        data_ = sess.run(
            data, feed_dict={cur_batch_size: batch_size})
        self.assertEqual(len(data_), batch_size)
        self.assertEqual(len(data_[0]), 784)

  def test_cached_batch(self):
    max_batch_size = 100
    data_set = tf.random_uniform(shape=(max_batch_size, 784), maxval=1.)
    var_data = data_reader.CachedDataReader(
        (data_set,), max_batch_size)
    cur_batch_size = tf.placeholder(
        shape=(), dtype=tf.int32, name='cur_batch_size')
    # Force create the ops
    data = var_data(cur_batch_size)[0]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord)
      data_ = sess.run(data, feed_dict={cur_batch_size: 25})
      stored_data_ = sess.run(var_data.cached_batch)[0]
      self.assertListEqual(list(data_[1]), list(stored_data_[1]))


if __name__ == '__main__':
  tf.test.main()
