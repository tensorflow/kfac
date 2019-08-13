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
    'load_mnist_as_iterator',
]


def load_mnist_as_tensors(flatten_images=True):
  """Loads MNIST as Tensors.

  Args:
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.

  Returns:
    images, labels, num_examples
  """

#   mnist_data = tf.contrib.learn.datasets.mnist.read_data_sets(
#       '/tmp/mnist', reshape=flatten_images)
#   num_examples = len(mnist_data.train.labels)
#   images = mnist_data.train.images
#   labels = mnist_data.train.labels
#
#   images = tf.constant(np.asarray(images, dtype=np.float32))
#   labels = tf.constant(np.asarray(labels, dtype=np.int64))
#
#   return images, labels, num_examples

  (images, labels), _ = tf.keras.datasets.mnist.load_data()
  num_examples = images.shape[0]

  if flatten_images:
    images = images.reshape(images.shape[0], 28**2)
  else:
    images = images.reshape(images.shape[0], 28, 28, 1)

  images = images.astype('float32')
  labels = labels.astype('int32')

  images /= 255.

  images = tf.constant(images)
  labels = tf.constant(labels)

  return images, labels, num_examples


def load_mnist_as_dataset(flatten_images=True):
  """Loads MNIST as a Dataset object.

  Args:
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.

  Returns:
    dataset, num_examples, where dataset is a Dataset object containing the
    whole MNIST training dataset and num_examples is the number of examples
    in the MNIST dataset (should be 60000).
  """
  images, labels, num_examples = load_mnist_as_tensors(
      flatten_images=flatten_images)
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  return dataset, num_examples


def load_mnist_as_iterator(num_epochs, batch_size,
                           use_fake_data=False,
                           flatten_images=True):
  """Loads MNIST dataset as an iterator Tensor.

  Args:
    num_epochs: int. Number of passes to make over the dataset.
    batch_size: int. Number of examples per minibatch.
    use_fake_data: bool. If True, generate a synthetic dataset rather than
      reading MNIST in.
    flatten_images: bool. If True, [28, 28, 1]-shaped images are flattened into
      [784]-shaped vectors.

  Returns:
    examples: Tensor of shape [batch_size, 784] if 'flatten_images' is
      True, else [batch_size, 28, 28, 1]. Each row is one example.
      Values in [0, 1].
    labels: Tensor of shape [batch_size]. Indices of integer corresponding to
      each example. Values in {0...9}.
  """

  if use_fake_data:
    rng = np.random.RandomState(42)
    num_examples = batch_size * 4
    images = rng.rand(num_examples, 28 * 28)
    if not flatten_images:
      images = np.reshape(images, [num_examples, 28, 28, 1])
    labels = rng.randint(10, size=num_examples)
    dataset = tf.data.Dataset.from_tensor_slices((np.asarray(
        images, dtype=np.float32), np.asarray(labels, dtype=np.int64)))
  else:
    dataset, num_examples = load_mnist_as_dataset(flatten_images=flatten_images)

  dataset = (dataset.shuffle(num_examples).repeat(num_epochs)
             .batch(batch_size).prefetch(5))
  return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
