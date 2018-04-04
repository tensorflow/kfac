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
"""Tests for kfac.op_queue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from kfac.python.ops import op_queue


class OpQueueTest(tf.test.TestCase):

  def testNextOp(self):
    """Ensures all ops get selected eventually."""
    with tf.Graph().as_default():
      ops = [
          tf.add(1, 2),
          tf.subtract(1, 2),
          tf.reduce_mean([1, 2]),
      ]
      queue = op_queue.OpQueue(ops, seed=0)

      with self.test_session() as sess:
        # Ensure every inv update op gets selected.
        selected_ops = set([queue.next_op(sess) for _ in ops])
        self.assertEqual(set(ops), set(selected_ops))

        # Ensure additional calls don't create any new ops.
        selected_ops.add(queue.next_op(sess))
        self.assertEqual(set(ops), set(selected_ops))


if __name__ == "__main__":
  tf.test.main()
