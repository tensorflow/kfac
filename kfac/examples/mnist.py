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
"""Utilities for loading MNIST into TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

__all__ = [
    'load_mnist_as_tensors',
    'load_mnist_as_dataset',
]


def load_mnist_as_tensors(data_dir, flatten_images=True):
  """Loads MNIST as Tensors.

  Args:
    data_dir: string. Directory to read MNIST examples from.
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.

  Returns:
    images, labels, num_examples
  """
  mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(
      data_dir, reshape=flatten_images)
  num_examples = len(mnist_data.train.labels)
  images = mnist_data.train.images
  labels = mnist_data.train.labels

  images = tf.constant(np.asarray(images, dtype=np.float32))
  labels = tf.constant(np.asarray(labels, dtype=np.int64))

  return images, labels, num_examples


def load_mnist_as_dataset(data_dir, flatten_images=True):
  """Loads MNIST as a Dataset object.

  Args:
    data_dir: string. Directory to read MNIST examples from.
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.

  Returns:
    dataset, num_examples, where dataset is a Dataset object containing the
    whole MNIST training dataset and num_examples is the number of examples
    in the MNIST dataset (should be 55000).
  """
  images, labels, num_examples = load_mnist_as_tensors(
      data_dir, flatten_images=flatten_images)
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  return dataset, num_examples
