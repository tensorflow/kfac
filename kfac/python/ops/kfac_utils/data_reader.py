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
      dataset: List of Tensors representing the dataset, shuffled, repeated,
        and batched into mini-batches of size at least `max_batch_size`.  In
        other words it should be reshuffled at each session.run call.  This can
        be done with the tf.data package using the construction demonstrated in
        load_mnist() function in examples/autoencoder_auto_damping.py.
      max_batch_size: `int`. Maximum batch size of the data that can be
        retrieved from the data set.
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
        tf.convert_to_tensor(self._max_batch_size, dtype=tf.int32),
        message='Data set read failure, Batch size greater than max allowed.'
    )
    with tf.control_dependencies([check_size]):
      return _slice_data(self._dataset, batch_size)


class CachedDataReader(VariableBatchReader):
  """Provides functionality to store variable batch size data."""

  def __init__(self, dataset, max_batch_size):
    """Initializes class and creates variables for storing previous batch.

    Args:
      dataset: List of Tensors representing the dataset, shuffled, repeated,
        and batched into mini-batches of size at least `max_batch_size`.  In
        other words it should be reshuffled at each session.run call.  This can
        be done with the tf.data package using the construction demonstrated in
        load_mnist() function in examples/autoencoder_auto_damping.py.
      max_batch_size: `int`. Maximum batch size of the data that can be
        retrieved from the data set.
    """
    super(CachedDataReader, self).__init__(dataset, max_batch_size)
    with tf.variable_scope('cached_data_reader'):
      self._cached_batch_storage = [
          tf.get_variable(
              name='{}{}'.format('cached_batch_storage_', i),
              shape=[max_batch_size]+ var.shape.as_list()[1:],
              dtype=var.dtype,
              trainable=False,
              use_resource=True) for i, var in enumerate(self._dataset)
      ]
      self._cached_batch_size = tf.get_variable(
          name='cached_batch_size', shape=(), dtype=tf.int32, trainable=False,
          use_resource=True)

      self._cached_batch = _slice_data(self._cached_batch_storage,
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

    # We need to make sure we read the cached batch before we update it!
    with tf.control_dependencies(self._cached_batch):
      batch_size_assign_op = self._cached_batch_size.assign(batch_size)
      data_assign_ops = [
          prev[:batch_size].assign(cur)  # yes, this actually works
          for prev, cur in zip(self._cached_batch_storage, sliced_data)
      ]
      with tf.control_dependencies(data_assign_ops + [batch_size_assign_op]):
        return [tf.identity(sdata) for sdata in sliced_data]

  @property
  def cached_batch(self):
    return self._cached_batch
