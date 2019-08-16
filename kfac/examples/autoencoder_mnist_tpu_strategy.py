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

This script demonstrates training on TPUs with TPUStrategy using KFAC optimizer
and updating the damping parameter according to the Levenberg-Marquardt rule.

See third_party/tensorflow_kfac/google/examples/ae_tpu_xm_launcher.py
for an example Borg launch script.  If you can't access this launch script,
some important things to know about running K-FAC on TPUs (at least for this
example) are that you must use high-precision matrix multiplications.
iterations_per_loop is not relevant when using TPU Strategy, but you must set
it to 1 when using TPU Estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import kfac
import tensorflow as tf

from kfac.examples import autoencoder_mnist
from kfac.examples import mnist


# TODO(znado): figure out the bug with this and update_damping_immediately=True.
# TODO(znado): Add checkpointing code to the training loop.
flags.DEFINE_integer('save_checkpoints_steps', 500,
                     'Number of iterations between model checkpoints.')
flags.DEFINE_string('model_dir', '', 'Model dir.')

# iterations_per_loop is not used with TPU Strategy. We keep the flag so the
# Estimator launching script can be used.
flags.DEFINE_integer('iterations_per_loop', 1,
                     'Number of iterations in a TPU training loop.')

flags.DEFINE_string('master', None,
                    'GRPC URL of the master '
                    '(e.g. grpc://ip.address.of.tpu:8470).')


FLAGS = flags.FLAGS


def make_train_op(minibatch,
                  batch_loss,
                  layer_collection,
                  loss_fn):
  """Constructs optimizer and train op.

  Args:
    minibatch: Tuple[Tensor, Tensor] representing the current batch of input
      images and labels.
    batch_loss: Tensor of shape (), Loss with respect to minibatch to be
      minimzed.
    layer_collection: LayerCollection object. Registry for model parameters.
      Required when using a K-FAC optimizer.
    loss_fn: A function that when called constructs the graph to compute the
      model loss on the current minibatch.  Returns a Tensor of the loss scalar.

  Returns:
    train_op: Op that can be used to update model parameters.
    optimizer: The KFAC optimizer used to produce train_op.

  Raises:
    ValueError: If layer_collection is None when K-FAC is selected as an
      optimization method.
  """
  # Do not use CrossShardOptimizer with K-FAC. K-FAC now handles its own
  # cross-replica syncronization automatically!

  return autoencoder_mnist.make_train_op(
      minibatch=minibatch,
      batch_size=minibatch[0].get_shape().as_list()[0],
      batch_loss=batch_loss,
      layer_collection=layer_collection,
      loss_fn=loss_fn,
      prev_train_batch=None,
      placement_strategy='replica_round_robin',
      )


def compute_squared_error(logits, targets):
  """Compute mean squared error."""
  return tf.reduce_sum(
      tf.reduce_mean(tf.square(targets - tf.nn.sigmoid(logits)), axis=0))


def compute_loss(logits, labels, model):
  """Compute loss value."""
  loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
  regularization_loss = tf.reduce_sum(model.losses)
  crossentropy_loss = tf.reduce_sum(tf.reduce_mean(loss_matrix, axis=0))
  return crossentropy_loss + regularization_loss


def mnist_input_fn(batch_size):
  dataset, num_examples = mnist.load_mnist_as_dataset(flatten_images=True)

  # Shuffle before repeat is correct unless you want repeat cases in the
  # same batch.
  dataset = (dataset.shuffle(num_examples)
             .repeat()
             .batch(batch_size, drop_remainder=True)
             .prefetch(tf.data.experimental.AUTOTUNE))
  return dataset


def _train_step(batch):
  """Estimator model_fn for an autoencoder with adaptive damping."""
  features, labels = batch
  model = autoencoder_mnist.get_keras_autoencoder(tensor=features)

  def loss_fn(minibatch, logits=None):
    """Compute the model loss given a batch of inputs.

    Args:
      minibatch: `Tuple[Tensor, Tensor]` for the current batch of input images
        and labels.
      logits: `Tensor` for the current batch of logits. If None then reuses the
        AutoEncoder to compute them.

    Returns:
      `Tensor` for the batch loss.
    """
    features, labels = minibatch
    del labels
    if logits is None:
      logits = model(features)
    batch_loss = compute_loss(logits=logits, labels=features, model=model)
    return batch_loss

  logits = model.output
  pre_update_batch_loss = loss_fn((features, labels), logits)
  pre_update_batch_error = compute_squared_error(logits, features)

  # binary_crossentropy corresponds to sigmoid_crossentropy.
  layer_collection = kfac.keras.utils.get_layer_collection(
      model, 'binary_crossentropy', seed=FLAGS.seed + 1)

  global_step = tf.train.get_or_create_global_step()
  train_op, kfac_optimizer = make_train_op(
      (features, labels),
      pre_update_batch_loss,
      layer_collection,
      loss_fn)
  tensors_to_print = {
      'learning_rate': kfac_optimizer.learning_rate,
      'momentum': kfac_optimizer.momentum,
      'damping': kfac_optimizer.damping,
      'global_step': global_step,
      'loss': pre_update_batch_loss,
      'error': pre_update_batch_error,
  }
  if FLAGS.adapt_damping:
    tensors_to_print['qmodel_change'] = kfac_optimizer.qmodel_change
    tensors_to_print['rho'] = kfac_optimizer.rho

  with tf.control_dependencies([train_op]):
    return {k: tf.identity(v) for k, v in tensors_to_print.items()}


def train():
  """Trains the Autoencoder using TPU Strategy."""
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.master)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

  with tpu_strategy.scope():
    data = mnist_input_fn(batch_size=FLAGS.batch_size)
    train_iterator = tpu_strategy.make_dataset_iterator(data)
    tensor_dict = tpu_strategy.experimental_run(_train_step, train_iterator)
    for k, v in tensor_dict.items():
      if k in ('loss', 'error'):   # Losses are NOT scaled for num replicas.
        tensor_dict[k] = tpu_strategy.reduce(tf.distribute.ReduceOp.MEAN, v)
      else:  # Other tensors (hyperparameters) are identical across replicas.
        # experimental_local_results gives you a tuple of per-replica values.
        tensor_dict[k] = tpu_strategy.experimental_local_results(v)

  config = tf.ConfigProto(allow_soft_placement=True)
  with tf.Session(cluster_resolver.master(), config=config) as session:
    session.run(tf.global_variables_initializer())
    session.run(train_iterator.initializer)
    print('Starting training.')
    for step in range(FLAGS.train_steps):
      values_dict = session.run(tensor_dict)
      print('Training Step: {}'.format(step))
      for k, v in values_dict.items():
        print('{}: {}'.format(k, v))
    print('Done training.')


def main(argv):
  del argv  # Unused.
  tf.set_random_seed(FLAGS.seed)
  # Invert using cholesky decomposition + triangular solve.  This is the only
  # code path for matrix inversion supported on TPU right now.
  kfac.utils.set_global_constants(posdef_inv_method='cholesky')
  kfac.fisher_factors.set_global_constants(
      eigenvalue_decomposition_threshold=10000)

  train()


if __name__ == '__main__':
  tf.app.run(main)
