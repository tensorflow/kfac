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
"""Reads variable size batches of data from a data set and stores read data.

`VariableBatchReader` reads variable size data from a dataset.
`CachedDataReader` on top of `VariableBatchReader` adds functionality to store
the read batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def _slice_data(stored_data, size):
  return [data[:size] for data in stored_data]


class VariableBatchReader(object):
  """Read data of varying batch sizes from a data set."""

  def __init__(self, dataset, max_batch_size):
    """Initializes class.

    Args:
      dataset: Iterable of tensors.
      max_batch_size: `int32` scalar tensor, Maximum batch size of the data that
        can be retrieved from the data set.
    """
    self._dataset = dataset
    self._max_batch_size = max_batch_size

  def __call__(self, batch_size):
    """Reads `batch_size` data.

    Args:
      batch_size: Tensor of type `int32`, batch size of the data to be
        retrieved from the dataset. `batch_size` should be less than or
        equal to `max_batch_size`.

    Returns:
       Read data, An iterable of tensors with batch size equal to `batch_size`.
    """
    check_size = tf.assert_less_equal(
        batch_size,
        self._max_batch_size,
        message='Data set read failure, Batch size greater than max allowed.'
    )
    with tf.control_dependencies([check_size]):
      return _slice_data(self._dataset, batch_size)


class CachedDataReader(VariableBatchReader):
  """Provides functionality to store variable batch size data."""

  def __init__(self, dataset, max_batch_size):
    """Initializes class and creates variables for storing previous batch.

    Args:
      dataset: Iterable of tensors.
      max_batch_size: `int32` scalar tensor, Maximum batch size of the data that
        can be retrieved from the data set.
    """
    super(CachedDataReader, self).__init__(dataset, max_batch_size)
    with tf.variable_scope('data_loader'):
      self._cached_batch = [
          tf.get_variable(
              name='{}{}'.format('cached_batch_', i),
              shape=[max_batch_size]+ var.shape.as_list()[1:],
              dtype=var.dtype,
              trainable=False) for i, var in enumerate(self._dataset)
      ]
      self._cached_batch_size = tf.get_variable(
          name='cached_batch_size', shape=(), dtype=tf.int32, trainable=False)
      self._cached_batch_op = _slice_data(self._cached_batch,
                                          self._cached_batch_size)

  def __call__(self, batch_size):
    """Reads `batch_size` data and stores the read batch.

    Args:
      batch_size: Tensor of type `int32`, batch size of the data to be
        retrieved from the dataset. `batch_size` should be less than or
        equal to `max_batch_size`.

    Returns:
       Read data, An iterable of tensors with batch size equal to `batch_size`.
    """
    sliced_data = super(CachedDataReader, self).__call__(batch_size)
    with tf.control_dependencies(sliced_data):
      batch_size_assign = [tf.assign(self._cached_batch_size, batch_size)]
      data_assign_op = [
          tf.scatter_update(prev, tf.range(batch_size), cur)
          for prev, cur in zip(self._cached_batch, sliced_data)
      ]
      with tf.control_dependencies(data_assign_op + batch_size_assign):
        return [tf.identity(sdata) for sdata in sliced_data]

  @property
  def cached_batch(self):
    return self._cached_batch_op

  @property
  def cached_batch_size(self):
    return self._cached_batch_size
