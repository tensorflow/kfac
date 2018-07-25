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
"""Implementation of Deep AutoEncoder from Martens & Grosse (2015).

This script demonstrates training using KFAC optimizer and updating the
damping parameter according to the Levenberg-Marquardt rule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import kfac
import tensorflow as tf

from kfac.python.ops.kfac_utils import data_reader
from kfac.python.ops.kfac_utils import mnist

# Training batch size.
_BATCH_SIZE = 128

# Auto Encoder layer size.
_ENCODER_LAYER_SIZE = [1000, 500, 250, 30]

flags.DEFINE_integer(
    'inverse_update_period', 10,
    '#Steps between computing inverse of fisher factor matrices.')
flags.DEFINE_integer(
    'cov_update_period', 1,
    '#Steps between computing covaraiance matrices.')
flags.DEFINE_integer('damping_adaptation_interval', 5,
                     '#Steps between updating the damping parameter.')
flags.DEFINE_float('weight_decay', 1e-5,
                   'L2 regularization applied to weight matrices.')
flags.DEFINE_string('data_dir', '/tmp/mnist', 'local mnist dir')
flags.DEFINE_integer('num_epochs', 128,
                     'Number of passes to make over the dataset.')
FLAGS = flags.FLAGS


def minimize_loss(batch_size,
                  batch_loss,
                  layer_collection,
                  loss_fn,
                  cached_reader):
  """Constructs optimizer and train op.

  Args:
    batch_size: Tensor of shape (), Size of the training batch.
    batch_loss: Tensor of shape (), Loss with respect to minibatch to be
      minimzed.
    layer_collection: LayerCollection or None. Registry for model parameters.
      Required when using a K-FAC optimizer.
    loss_fn: Function which takes as input training data and returns loss.
    cached_reader: `data_reader.CachedReader` instance.

  Returns:
    train_op: Op that can be used to update model parameters.
    optimizer: Optimizer used to produce train_op.

  Raises:
    ValueError: If layer_collection is None when K-FAC is selected as an
      optimization method.
  """
  global_step = tf.train.get_or_create_global_step()

  if layer_collection is None:
    raise ValueError('layer_collection must be defined to use K-FAC.')

  optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=FLAGS.inverse_update_period,
      cov_update_every=FLAGS.cov_update_period,
      learning_rate=1e-3,
      damping=100.,
      cov_ema_decay=0.95,
      momentum=0.95,
      layer_collection=layer_collection,
      batch_size=batch_size
      )
  # Set the damping parameters required to adapt damping.
  optimizer.set_damping_adaptation_params(
      prev_train_batch=cached_reader.cached_batch,
      is_chief=True,
      loss_fn=loss_fn,
      damping_adaptation_decay=0.95,
      damping_adaptation_interval=FLAGS.damping_adaptation_interval,
  )
  return optimizer.minimize(batch_loss, global_step=global_step)


class AutoEncoder(object):
  """Deep AutoEncoder as described in Hinton & Salakhutdinov, 2006."""

  def __init__(self, encoder_layer_sizes, weight_decay=0.0, variables=None):
    """Initializes AutoEncoder.

    Args:
      encoder_layer_sizes: list of ints. Number of dims output by each layer of
        the encoder portion of the autoencoder. The reverse, sans the last, will
        be used to construct the decoder.
      weight_decay: float or Tensor. Multiplier for L2 regularization added to
        the loss.
      variables: None or dict. If dict, contains string keys of the form
        "linear_x/w" and "linear_x/b" and values corresponding to each layer's
        weight and bias variables. (Default: None)
    """
    self._encoder_layer_sizes = encoder_layer_sizes
    self._weight_decay = weight_decay
    self._variables = variables or {}

  @property
  def variables(self):
    """Dictionary mapping variable names to variables.

    Returns:
      dict with variable names as keys and tf.Variables as values. Can be used
      as the 'variables' argument to __init__().
    """
    return self._variables

  def _layer_output_sizes_with_activations(self, num_pixels):
    """Produces output size and activation for each layer.

    Args:
      num_pixels: int. Number of pixels in each input example.

    Yields:
      (output_size, activation function) for each layer in the autoencoder.
      Output size is the number of output dimensions for the layer (input size
      is inferred from input).
    """
    # Apply sigmoid to all but last encoder layer.
    for output_size in self._encoder_layer_sizes[:-1]:
      yield (output_size, tf.sigmoid)

    # Last encoder layer has no activation.
    yield (self._encoder_layer_sizes[-1], tf.identity)

    # Apply sigmoid to all decoder layers.
    decoder_layer_sizes = reversed(self._encoder_layer_sizes[:-1])
    for output_size in decoder_layer_sizes:
      yield (output_size, tf.sigmoid)

    # Apply sigmoid to reconstruct input.
    yield (num_pixels, tf.sigmoid)

  def build(self, layer_collection=None):
    """Adds AutoEncoder inference and loss computation to graph.

    Reuses variables from self.variables if available, else instantiates new
    variables in their place.

    Args:
      layer_collection: LayerCollection or None. If non-null, each layer's
        parameters and per-pixel logits are registered with this LayerCollection
        instance.

    Returns:
      A loss function which takes training batch and returns loss, error.
    """
    def loss_fn(training_batch):
      """Computes loss on the training batch.

      Args:
        training_batch: Tensor of shape [num_examples, num_pixels]. Mini-batch
          to calculate loss with respect to.

      Returns:
        loss: Tensor of shape [] representing regularized loss.
        error: Tensor of shape [] representing squared reconstruction error.
      """
      batch = training_batch[0]
      num_pixels = batch.get_shape()[1].value
      if len(batch.get_shape()) > 2:
        batch = tf.reshape(batch, [-1, 784])
      # Build layers. Keep track of them along the way.
      with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        inputs = batch
        for i, (output_size, activation_fn) in enumerate(
            self._layer_output_sizes_with_activations(num_pixels)):
          # Build variables for layer i
          linear_mod_name = 'linear_%d' % i
          weights_name = '%s/w' % linear_mod_name
          biases_name = '%s/b' % linear_mod_name

          # Initialize variables for linear module.
          if weights_name not in self._variables:
            self._variables[weights_name] = tf.get_variable(
                weights_name,
                shape=[int(inputs.get_shape()[1]), output_size])

          weights = self._variables[weights_name]
          if biases_name not in self._variables:
            self._variables[biases_name] = tf.get_variable(
                biases_name,
                shape=[output_size],
                initializer=tf.zeros_initializer())
          biases = self._variables[biases_name]

          # Apply layer.
          preactivations = tf.matmul(inputs, weights) + biases
          activations = activation_fn(preactivations)

          # Prepare for next layer.
          inputs = activations

        # Register outputs.
        logits = preactivations
      if layer_collection is not None:
        layer_collection.register_multi_bernoulli_predictive_distribution(
            logits)
        layer_collection.auto_register_layers()

      with tf.variable_scope('model'):
        # Calculate logistic loss, per-pixel. Sum across pixels in an image.
        # Average across images.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=batch, logits=logits)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)

        # Calculate squared error, per-pixel. Sum across pixels in an image.
        # Average across images.
        error = tf.square(batch - tf.sigmoid(logits))
        error = tf.reduce_mean(tf.reduce_sum(error, axis=1), axis=0)
        # Calculate l2 regularization across all weights.
        regularization = 0.5 * self._weight_decay * tf.add_n([
            tf.nn.l2_loss(var)
            for k, var in self._variables.items()
            if k.endswith('/w')
        ])

      return loss + regularization, error
    return loss_fn


def load_mnist(batch_size):
  """Creates MNIST dataset and wraps it inside cached data reader.

  Args:
    batch_size: Scalar placeholder variable which needs to fed to read variable
      sized training data.

  Returns:
    cached_reader: `data_reader.CachedReader` instance which wraps MNIST
      dataset.
    training_batch: Tensor of shape `[batch_size, 784]`, MNIST training images.
  """
  # Create a MNIST data batch with max training batch size.
  # data_set = datasets.Mnist(batch_size=_BATCH_SIZE, mode='train')()
  data_set = mnist.load_mnist(
      FLAGS.data_dir,
      num_epochs=FLAGS.num_epochs,
      batch_size=_BATCH_SIZE,
      flatten_images=True)
  # Wrap the data set into cached_reader which provides variable sized training
  # and caches the read train batch.
  cached_reader = data_reader.CachedDataReader(data_set, _BATCH_SIZE)
  return cached_reader, cached_reader(batch_size)[0]


def main(_):
  batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')
  cached_reader, training_batch = load_mnist(batch_size)

  # Create autoencoder model.
  training_model = AutoEncoder(
      _ENCODER_LAYER_SIZE,
      weight_decay=FLAGS.weight_decay)
  layer_collection = kfac.LayerCollection()
  batch_loss, batch_error = training_model.build(layer_collection)(
      (training_batch,))

  # Minimize loss.
  train_op = minimize_loss(
      batch_size,
      batch_loss,
      layer_collection,
      loss_fn=lambda prev_batch: training_model.build()(prev_batch)[0],
      cached_reader=cached_reader)

  # Fit model.
  global_step = tf.train.get_or_create_global_step()
  with tf.train.MonitoredTrainingSession(save_checkpoint_secs=30) as sess:
    while not sess.should_stop():
      i = sess.run(global_step)
      # Update the covariance matrices. Also updates the damping parameter
      # every damping adaptation interval. Note that the
      # `cached_reader.cached_batch` is paased to `opt.KfacOptimizer`.
      _, batch_loss_, batch_error_ = sess.run(
          [train_op, batch_loss, batch_error],
          feed_dict={batch_size: _BATCH_SIZE})

      # Print training stats.
      tf.logging.info('%d steps/batch_loss = %f, batch_error = %f', i,
                      batch_loss_, batch_error_)


if __name__ == '__main__':
  tf.app.run(main)
