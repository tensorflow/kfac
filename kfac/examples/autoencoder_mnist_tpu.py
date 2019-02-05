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
"""Implementation of Deep AutoEncoder from Martens & Grosse (2015).

This script demonstrates training using KFAC optimizer on TPUs.  Will not work
as well as adaptive learning rate, momentum, and damping because it has fixed
hyper parameters, adaptive tuning coming soon.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import kfac
import sonnet as snt
import tensorflow as tf

from kfac.examples import mnist

# Model parameters
_ENCODER_SIZES = [1000, 500, 250, 30]
_DECODER_SIZES = [250, 500, 1000]


flags.DEFINE_integer('inverse_update_period', 10,
                     '# of steps between computing inverse of fisher factor '
                     'matrices.')
flags.DEFINE_integer('cov_update_period', 1,
                     '# of steps between computing covaraiance matrices.')
flags.DEFINE_integer('train_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('seed', 12345, 'Random seed')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations in a TPU training loop.')
flags.DEFINE_integer('num_shards', 8, 'Number of shards (TPU cores).')
flags.DEFINE_string('master', None,
                    'GRPC URL of the master '
                    '(e.g. grpc://ip.address.of.tpu:8470). You must specify '
                    'either this flag or --tpu_name.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('momentum', 0.95, 'Momentum decay value.')

flags.DEFINE_float('weight_decay', 1e-5,
                   'L2 regularization applied to weight matrices.')

flags.DEFINE_string('model_dir', '', 'Model dir.')

flags.DEFINE_integer('batch_size', 1024,
                     'The size of the mini-batches to use if not using the '
                     'schedule.')


FLAGS = flags.FLAGS


class AutoEncoder(snt.AbstractModule):
  """Simple autoencoder module."""

  def __init__(self,
               input_size,
               regularizers=None,
               initializers=None,
               custom_getter=None,
               nonlinearity=tf.tanh,  # sigmoid cannot be used with default init
               name='AutoEncoder'):
    super(AutoEncoder, self).__init__(name=name)

    if initializers is None:
      initializers = {'w': tf.glorot_uniform_initializer(),
                      'b': tf.zeros_initializer()}
    if regularizers is None:
      regularizers = {'w': lambda w: FLAGS.weight_decay * tf.nn.l2_loss(w),
                      'b': lambda w: FLAGS.weight_decay * tf.nn.l2_loss(w),}

    with self._enter_variable_scope():
      self._encoder = snt.nets.MLP(
          output_sizes=_ENCODER_SIZES,
          regularizers=regularizers,
          initializers=initializers,
          custom_getter=custom_getter,
          activation=nonlinearity,
          activate_final=False)
      self._decoder = snt.nets.MLP(
          output_sizes=_DECODER_SIZES + [input_size],
          regularizers=regularizers,
          initializers=initializers,
          custom_getter=custom_getter,
          activation=nonlinearity,
          activate_final=False)

  def _build(self, inputs):
    code = self._encoder(inputs)
    output = self._decoder(code)

    return output


def make_train_op(batch_loss,
                  layer_collection,
                  global_step):
  """Constructs optimizer and train op.

  Args:
    batch_loss: Tensor of shape (), Loss with respect to minibatch to be
      minimzed.
    layer_collection: LayerCollection or None. Registry for model parameters.
      Required when using a K-FAC optimizer.
    global_step: The global training step.

  Returns:
    train_op: Op that can be used to update model parameters.
    optimizer: KFAC optimizer, not the tpu.CrossShardOptimizer used to produce
      train_op.

  Raises:
    ValueError: If layer_collection is None when K-FAC is selected as an
      optimization method.
  """

  if layer_collection is None:
    raise ValueError('layer_collection must be defined to use K-FAC.')

  kfac_optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=FLAGS.inverse_update_period,
      cov_update_every=FLAGS.cov_update_period,
      learning_rate=FLAGS.learning_rate,
      damping=1e-2,
      cov_ema_decay=0.95,
      momentum=FLAGS.momentum,
      momentum_type='regular',
      layer_collection=layer_collection,
      batch_size=FLAGS.batch_size // FLAGS.num_shards,
      num_burnin_steps=5,
      )
  optimizer = tf.contrib.tpu.CrossShardOptimizer(kfac_optimizer)
  return optimizer.minimize(batch_loss, global_step=global_step), kfac_optimizer


def compute_loss(logits=None,
                 labels=None,
                 layer_collection=None,
                 return_acc=False):
  """Compute loss value."""
  graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_regularization_loss = tf.reduce_sum(graph_regularizers)
  loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
  loss = tf.reduce_sum(tf.reduce_mean(loss_matrix, axis=0))
  regularized_loss = loss + total_regularization_loss

  if layer_collection is not None:
    layer_collection.register_multi_bernoulli_predictive_distribution(logits)
    layer_collection.auto_register_layers()

  if return_acc:
    squared_error = tf.reduce_sum(
        tf.reduce_mean(
            tf.square(labels - tf.nn.sigmoid(logits)),
            axis=0))
    return regularized_loss, squared_error

  return regularized_loss


def mnist_input_fn(params):
  """Creates MNIST tf.data.Dataset."""
  dataset, num_examples = mnist.load_mnist_as_dataset(flatten_images=True)

  # Shuffle before repeat is correct unless you want repeat cases in the
  # same batch.
  dataset = (dataset.shuffle(num_examples)
             .repeat()
             .batch(params['batch_size'], drop_remainder=True)
             .prefetch(tf.contrib.data.AUTOTUNE))
  return dataset


def print_tensors(**tensors):
  print_op = tf.no_op()
  for name in sorted(tensors):
    with tf.control_dependencies([print_op]):
      tensor = tensors[name]
      print_op = tf.Print(tensor, [tensor], message=name + '=')
  with tf.control_dependencies([print_op]):
    print_op = tf.Print(0., [0.], message='------')
  return print_op


def model_fn(features, labels, mode, params):
  """The model_fn for TPUEstimator."""
  del labels
  del params
  # Create autoencoder model using Soham's code instead
  training_model = AutoEncoder(784)

  logits = training_model(features)

  layer_collection = kfac.LayerCollection()
  batch_loss = compute_loss(
      logits=logits,
      labels=features,
      layer_collection=layer_collection)

  global_step = tf.train.get_or_create_global_step()
  train_op, kfac_optimizer = make_train_op(
      batch_loss, layer_collection, global_step)
  tensors_to_print = {
      'learning_rate': tf.expand_dims(kfac_optimizer.learning_rate, 0),
      'qmodel_change': tf.expand_dims(kfac_optimizer.qmodel_change, 0),
      'momentum': tf.expand_dims(kfac_optimizer.momentum, 0),
      'damping': tf.expand_dims(kfac_optimizer.damping, 0),
      'global_step': tf.expand_dims(global_step, 0),
      'rho': tf.expand_dims(kfac_optimizer.rho, 0),
  }
  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=batch_loss,
      train_op=train_op,
      host_call=(print_tensors, tensors_to_print),
      eval_metrics=None)


def main(_):
  tf.set_random_seed(FLAGS.seed)
  # Invert using cholesky decomposition + triangular solve.  This is the only
  # code path for matrix inversion supported on TPU right now.
  kfac.utils.set_global_constants(posdef_inv_method='cholesky')
  kfac.fisher_factors.set_global_constants(
      eigenvalue_decomposition_threshold=10000)

  config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.iterations_per_loop,
      cluster=None,
      tf_random_seed=FLAGS.seed,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards,
          per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tf.contrib.tpu.TPUEstimator(
      use_tpu=True,
      model_fn=model_fn,
      config=config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=1024)

  resnet_classifier.train(
      input_fn=mnist_input_fn,
      max_steps=FLAGS.train_steps,
      hooks=[])


if __name__ == '__main__':
  tf.app.run(main)

