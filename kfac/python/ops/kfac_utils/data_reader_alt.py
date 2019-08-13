# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Reads variable size batches of data from a data set and stores read data.

`VariableBatchReader` reads variable size data from a dataset.
`CachedDataReader` on top of `VariableBatchReader` adds functionality to store
the read batch for use in the next session.run() call.

This file is similar to data_reader.py but uses an alternative implementation
that requires the whole dataset to be passed in. This will often be faster than
using the original implementation with a very large max_batch_size.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def _extract_data(tensor_list, indices):
  return [tf.gather(tensor, indices, axis=0) for tensor in tensor_list]


class VariableBatchReader(object):
  """Read data of varying batch sizes from a data set."""

  def __init__(self, dataset, num_examples):
    """Initializes class.

    Args:
      dataset: List of Tensors. These must remain constant across session.run
        calls, unlike the version of VariableBatchReader in data_reader.py.
      num_examples: The number of examples in the data set (i.e. dimension 0
        of the elements of `dataset`).
    """
    self._dataset = dataset
    self._num_examples = num_examples
    self._indices = None

  def __call__(self, batch_size):
    """Reads `batch_size` data.

    Args:
      batch_size: Tensor of type `int32`. Batch size of the data to be
        retrieved from the dataset. `batch_size` should be less than or
        equal to the number of examples in the dataset.

    Returns:
       Read data, a list of Tensors with batch size equal to `batch_size`.
    """
    check_size = tf.assert_less_equal(
        batch_size,
        tf.convert_to_tensor(self._num_examples, dtype=tf.int32),
        message='Data set read failure, batch_size > num_examples.'
    )
    with tf.control_dependencies([check_size]):
      self._indices = tf.random.shuffle(
          tf.range(self._num_examples, dtype=tf.int32))
      return _extract_data(self._dataset, self._indices[:batch_size])


class CachedDataReader(VariableBatchReader):
  """Provides functionality to store variable batch size data."""

  def __init__(self, dataset, num_examples):
    """Initializes class and creates variables for storing previous batch.

    Args:
      dataset: List of Tensors. These must remain constant across session.run
        calls, unlike the version of VariableBatchReader in data_reader.py.
      num_examples: The number of examples in the data set (i.e. dimension 0
        of the elements of `dataset`).
    """
    super(CachedDataReader, self).__init__(dataset, num_examples)

    self._cached_batch_indices = tf.get_variable(
        name='cached_batch_indices',
        shape=[self._num_examples],
        dtype=tf.int32,
        trainable=False,
        use_resource=True)

    self._cached_batch_size = tf.get_variable(
        name='cached_batch_size', shape=(), dtype=tf.int32, trainable=False,
        use_resource=True)

    self._cached_batch = _extract_data(
        self._dataset,
        self._cached_batch_indices[:self._cached_batch_size])

  def __call__(self, batch_size):
    """Reads `batch_size` data and stores the read batch.

    Args:
      batch_size: Tensor of type `int32`, batch size of the data to be
        retrieved from the dataset. `batch_size` should be less than or
        equal to `max_batch_size`.

    Returns:
       Read data, An iterable of tensors with batch size equal to `batch_size`.
    """
    tensor_list = super(CachedDataReader, self).__call__(batch_size)

    with tf.control_dependencies(self._cached_batch):
      indices_assign_op = self._cached_batch_indices.assign(self._indices)
      batch_size_assign_op = tf.assign(self._cached_batch_size, batch_size)

      with tf.control_dependencies([indices_assign_op, batch_size_assign_op]):
        return [tf.identity(tensor) for tensor in tensor_list]

  @property
  def cached_batch(self):
    return self._cached_batch

