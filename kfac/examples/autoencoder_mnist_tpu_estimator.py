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

This script demonstrates training on TPUs with TPU Estimator using the KFAC
optimizer, updating the damping parameter according to the
Levenberg-Marquardt rule, and using the quadratic model method for adapting
the learning rate and momentum parameters.

See third_party/tensorflow_kfac/google/examples/ae_tpu_xm_launcher.py
for an example Borg launch script.  If you can't access this launch script,
some important things to know about running K-FAC on TPUs (at least for this
example) are that you must use higher-precision matrix multiplications.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import kfac
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu

from kfac.examples import autoencoder_mnist
from kfac.examples import mnist


flags.DEFINE_integer('save_checkpoints_steps', 500,
                     'Number of iterations between model checkpoints.')

flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations in a TPU training loop.')

flags.DEFINE_string('model_dir', '', 'Model dir.')

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


def compute_loss(logits, labels):
  """Compute loss value."""
  graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_regularization_loss = tf.add_n(graph_regularizers)
  loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                        labels=labels)
  loss = tf.reduce_sum(tf.reduce_mean(loss_matrix, axis=0))
  regularized_loss = loss + total_regularization_loss
  return regularized_loss


def mnist_input_fn(params):
  dataset, num_examples = mnist.load_mnist_as_dataset(flatten_images=True)

  # Shuffle before repeat is correct unless you want repeat cases in the
  # same batch.
  dataset = (
      dataset.shuffle(num_examples).repeat().batch(
          params['batch_size'],
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
  return dataset


def print_tensors(**tensors):
  """Host call function to print Tensors from the TPU during training."""
  print_op = tf.no_op()
  for name in sorted(tensors):
    with tf.control_dependencies([print_op]):
      tensor = tensors[name]
      if name in ['error', 'loss']:
        tensor = tf.reduce_mean(tensor)
      print_op = tf.Print(tensor, [tensor], message=name + '=')
  with tf.control_dependencies([print_op]):
    return tf.Print(0., [0.], message='------')


def _model_fn(features, labels, mode, params):
  """Estimator model_fn for an autoencoder with adaptive damping."""
  del params
  layer_collection = kfac.LayerCollection()
  training_model_fn = autoencoder_mnist.AutoEncoder(784)

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
      # Note we do not need to do anything like
      # `with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):`
      # here because Sonnet takes care of variable reuse for us as long as we
      # call the same `training_model_fn` module.  Otherwise we would need to
      # use variable reusing here.
      logits = training_model_fn(features)
    batch_loss = compute_loss(logits=logits, labels=features)
    return batch_loss

  logits = training_model_fn(features)
  pre_update_batch_loss = loss_fn((features, labels), logits=logits)
  pre_update_batch_error = compute_squared_error(logits, features)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Make sure never to confuse this with register_softmax_cross_entropy_loss!
    layer_collection.register_sigmoid_cross_entropy_loss(logits,
                                                         seed=FLAGS.seed + 1)
    layer_collection.auto_register_layers()

    global_step = tf.train.get_or_create_global_step()
    train_op, kfac_optimizer = make_train_op(
        (features, labels),
        pre_update_batch_loss,
        layer_collection,
        loss_fn)

    tensors_to_print = {
        'learning_rate': tf.expand_dims(kfac_optimizer.learning_rate, 0),
        'momentum': tf.expand_dims(kfac_optimizer.momentum, 0),
        'damping': tf.expand_dims(kfac_optimizer.damping, 0),
        'global_step': tf.expand_dims(global_step, 0),
        'loss': tf.expand_dims(pre_update_batch_loss, 0),
        'error': tf.expand_dims(pre_update_batch_error, 0),
    }
    if FLAGS.adapt_damping:
      tensors_to_print['qmodel_change'] = tf.expand_dims(
          kfac_optimizer.qmodel_change, 0)
      tensors_to_print['rho'] = tf.expand_dims(kfac_optimizer.rho, 0)

    return contrib_tpu.TPUEstimatorSpec(
        mode=mode,
        loss=pre_update_batch_loss,
        train_op=train_op,
        host_call=(print_tensors, tensors_to_print),
        eval_metrics=None)

  else:  # mode == tf.estimator.ModeKeys.{EVAL, PREDICT}:
    return contrib_tpu.TPUEstimatorSpec(
        mode=mode,
        loss=pre_update_batch_loss,
        eval_metrics=None)


def make_tpu_run_config(master, seed, model_dir, iterations_per_loop,
                        save_checkpoints_steps):
  return contrib_tpu.RunConfig(
      master=master,
      evaluation_master=master,
      model_dir=model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      cluster=None,
      tf_random_seed=seed,
      tpu_config=contrib_tpu.TPUConfig(iterations_per_loop=iterations_per_loop))


def main(argv):

  if FLAGS.use_control_flow_v2:
    tf.enable_control_flow_v2()

  del argv  # Unused.
  tf.set_random_seed(FLAGS.seed)
  # Invert using cholesky decomposition + triangular solve.  This is the only
  # code path for matrix inversion supported on TPU right now.
  kfac.utils.set_global_constants(posdef_inv_method='cholesky')
  kfac.fisher_factors.set_global_constants(
      eigenvalue_decomposition_threshold=10000)

  config = make_tpu_run_config(
      FLAGS.master, FLAGS.seed, FLAGS.model_dir, FLAGS.iterations_per_loop,
      FLAGS.save_checkpoints_steps)

  estimator = contrib_tpu.TPUEstimator(
      use_tpu=True,
      model_fn=_model_fn,
      config=config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=1024)

  estimator.train(
      input_fn=mnist_input_fn,
      max_steps=FLAGS.train_steps,
      hooks=[])


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.app.run(main)
