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
"""Full implementation of deep autoencoder experiment from original K-FAC paper.

This script demonstrates training using KFAC optimizer and updating the
damping parameter according to the Levenberg-Marquardt rule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# Dependency imports
from absl import flags
import kfac
import sonnet as snt
import tensorflow as tf

from kfac.examples import mnist
from kfac.python.ops.kfac_utils import data_reader
from kfac.python.ops.kfac_utils import data_reader_alt


# Model parameters
_ENCODER_SIZES = [1000, 500, 250, 30]
_DECODER_SIZES = [250, 500, 1000]
_NONLINEARITY = tf.tanh  # Note: sigmoid cannot be used with the default init.
_WEIGHTS_INITIALIZER = None  # Default init


flags.DEFINE_integer('train_steps', 10000, 'Number of training steps.')
flags.DEFINE_integer('inverse_update_period', 5,
                     '# of steps between computing inverse of Fisher factor '
                     'matrices.')
flags.DEFINE_integer('cov_update_period', 1,
                     '# of steps between computing covaraiance matrices.')
flags.DEFINE_integer('damping_adaptation_interval', 5,
                     '# of steps between updating the damping parameter.')

flags.DEFINE_integer('num_burnin_steps', 5, 'Number of steps the at the '
                     'start of training where the optimizer will only perform '
                     'cov updates. Will not work on CrossShardOptimizer. See '
                     'PeriodicInvCovUpdateKfacOpt for details.')
flags.DEFINE_integer('seed', 12345, 'Random seed')
flags.DEFINE_float('learning_rate', 3e-3,
                   'Learning rate to use when adaptation="off".')
flags.DEFINE_float('momentum', 0.9,
                   'Momentum decay value to use when '
                   'lrmu_adaptation="off" or "only_lr".')
flags.DEFINE_float('damping', 1e-2, 'The fixed damping value to use. This is '
                   'ignored if adapt_damping is True.')

flags.DEFINE_float('l2_reg', 1e-5,
                   'L2 regularization applied to weight matrices.')

flags.DEFINE_boolean('update_damping_immediately', True, 'Adapt the damping '
                     'immediately after the parameter update (i.e. in the same '
                     'sess.run() call).  Only safe if everything is a resource '
                     'variable.')

flags.DEFINE_boolean('use_batch_size_schedule', True,
                     'If True then we use the growing mini-batch schedule from '
                     'the original K-FAC paper.')
flags.DEFINE_integer('batch_size', 1024,
                     'The size of the mini-batches to use if not using the '
                     'schedule.')

flags.DEFINE_string('lrmu_adaptation', 'on',
                    'If set to "on" then we use the quadratic model '
                    'based learning-rate and momentum adaptation method from '
                    'the original paper. Note that this only works well in '
                    'practice when use_batch_size_schedule=True. Can also '
                    'be set to "off" and "only_lr", which turns '
                    'it off, or uses a version where the momentum parameter '
                    'is fixed (resp.).')


flags.DEFINE_boolean('use_alt_data_reader', True,
                     'If True we use the alternative data reader for MNIST '
                     'that is faster for small datasets.')

flags.DEFINE_string('device', '/gpu:0',
                    'The device to run the major ops on.')

flags.DEFINE_boolean('adapt_damping', True,
                     'If True we use the LM rule for damping adaptation as '
                     'described in the original K-FAC paper.')

# When using damping adaptation it is advisable to start with a high
# value. This value is probably far too high to use for most neural nets
# if you aren't using damping adaptation. (Although it always depends on
# the scale of the loss.)
flags.DEFINE_float('initial_damping', 0.1,
                   'The initial damping value to use when adapt_damping is '
                   'True.')

flags.DEFINE_string('optimizer', 'kfac',
                    'The optimizer to use. Can be kfac or adam. If adam is '
                    'used the various kfac hyperparameter map roughly on to '
                    'their Adam equivalents.')

flags.DEFINE_float('polyak_decay', 0.995, 'Rate of decay for Polyak averaging.')

flags.DEFINE_integer('eval_every', 50,
                     'Interval to print total training loss.')

flags.DEFINE_boolean('use_sua_approx', False,
                     'If True we use the SUA approximation for conv layers.')


FLAGS = flags.FLAGS


class Model(snt.AbstractModule):
  """CNN model for MNIST data."""

  def _build(self, inputs):

    if FLAGS.l2_reg:
      regularizers = {'w': lambda w: FLAGS.l2_reg*tf.nn.l2_loss(w),
                      'b': lambda w: FLAGS.l2_reg*tf.nn.l2_loss(w),}
    else:
      regularizers = None

    reshape = snt.BatchReshape([28, 28, 1])

    conv = snt.Conv2D(2, 5, padding=snt.SAME, regularizers=regularizers)
    relu = tf.nn.relu(conv(reshape(inputs)))

    max_pool = tf.nn.max_pool(relu, (2, 2), (2, 2), padding=snt.SAME)

    conv = snt.Conv2D(4, 5, padding=snt.SAME, regularizers=regularizers)
    relu = tf.nn.relu(conv(max_pool))

    max_pool = tf.nn.max_pool(relu, (2, 2), (2, 2), padding=snt.SAME)

    flatten = snt.BatchFlatten()(max_pool)

    linear = snt.Linear(32, regularizers=regularizers)(flatten)

    return snt.Linear(10, regularizers=regularizers)(linear)


def make_train_op(minibatch,
                  batch_size,
                  batch_loss,
                  layer_collection,
                  loss_fn,
                  prev_train_batch=None,
                  placement_strategy=None,
                  print_logs=False,
                  tf_replicator=None):
  """Constructs optimizer and train op.

  Args:
    minibatch: A list/tuple of Tensors (typically representing the current
      mini-batch of input images and labels).
    batch_size: Tensor of shape (). Size of the training mini-batch.
    batch_loss: Tensor of shape (). Mini-batch loss tensor.
    layer_collection: LayerCollection object. Registry for model parameters.
      Required when using a K-FAC optimizer.
    loss_fn: Function which takes as input a mini-batch and returns the loss.
    prev_train_batch: `Tensor` of the previous training batch, can be accessed
      from the data_reader.CachedReader cached_batch property. (Default: None)
    placement_strategy: `str`, the placement_strategy argument for
      `KfacOptimizer`. (Default: None)
    print_logs: `Bool`. If True we print logs using K-FAC's built-in
      tf.print-based logs printer. (Default: False)
    tf_replicator: A Replicator object or None. If not None, K-FAC will set
        itself up to work inside of the provided TF-Replicator object.
        (Default: None)

  Returns:
    train_op: Op that can be used to update model parameters.
    optimizer: Optimizer used to produce train_op.

  Raises:
    ValueError: If layer_collection is None when K-FAC is selected as an
      optimization method.
  """
  global_step = tf.train.get_or_create_global_step()

  if FLAGS.optimizer == 'kfac':
    if FLAGS.lrmu_adaptation == 'on':
      learning_rate = None
      momentum = None
      momentum_type = 'qmodel'
    elif FLAGS.lrmu_adaptation == 'only_lr':
      learning_rate = None
      momentum = FLAGS.momentum
      momentum_type = 'qmodel_fixedmu'
    elif FLAGS.lrmu_adaptation == 'off':
      learning_rate = FLAGS.learning_rate
      momentum = FLAGS.momentum
      # momentum_type = 'regular'
      momentum_type = 'adam'

    if FLAGS.adapt_damping:
      damping = FLAGS.initial_damping
    else:
      damping = FLAGS.damping

    optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
        invert_every=FLAGS.inverse_update_period,
        cov_update_every=FLAGS.cov_update_period,
        learning_rate=learning_rate,
        damping=damping,
        cov_ema_decay=0.95,
        momentum=momentum,
        momentum_type=momentum_type,
        layer_collection=layer_collection,
        batch_size=batch_size,
        num_burnin_steps=FLAGS.num_burnin_steps,
        adapt_damping=FLAGS.adapt_damping,
        # Note that many of the arguments below don't do anything when
        # adapt_damping=False.
        update_damping_immediately=FLAGS.update_damping_immediately,
        is_chief=True,
        prev_train_batch=prev_train_batch,
        loss=batch_loss,
        loss_fn=loss_fn,
        damping_adaptation_decay=0.9,
        damping_adaptation_interval=FLAGS.damping_adaptation_interval,
        min_damping=FLAGS.l2_reg,
        train_batch=minibatch,
        placement_strategy=placement_strategy,
        print_logs=print_logs,
        tf_replicator=tf_replicator
        )

  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.momentum,
        epsilon=FLAGS.damping,
        beta2=0.99)

  return optimizer.minimize(batch_loss, global_step=global_step), optimizer


def compute_loss(logits=None,
                 labels=None,
                 layer_collection=None,
                 return_error=False):
  """Compute loss value."""
  graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_regularization_loss = tf.reduce_sum(graph_regularizers)

  loss_matrix = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                               labels=labels)
  loss = tf.reduce_mean(loss_matrix, axis=0)
  regularized_loss = loss + total_regularization_loss

  if layer_collection is not None:
    # Make sure never to confuse this with register_sigmoid_cross_entropy_loss!
    layer_collection.register_softmax_cross_entropy_loss(logits,
                                                         seed=FLAGS.seed + 1)
    layer_collection.auto_register_layers()

  if return_error:
    error = 1.0 - tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.argmax(logits, axis=1, output_type=tf.int32)),
        tf.float32))
    return regularized_loss, error

  return regularized_loss


def load_mnist():
  """Creates MNIST dataset and wraps it inside cached data reader.

  Returns:
    cached_reader: `data_reader.CachedReader` instance which wraps MNIST
      dataset.
    num_examples: int. The number of training examples.
  """
  # Wrap the data set into cached_reader which provides variable sized training
  # and caches the read train batch.

  if not FLAGS.use_alt_data_reader:
    # Version 1 using data_reader.py (slow!)
    dataset, num_examples = mnist.load_mnist_as_dataset(flatten_images=True)
    if FLAGS.use_batch_size_schedule:
      max_batch_size = num_examples
    else:
      max_batch_size = FLAGS.batch_size

    # Shuffle before repeat is correct unless you want repeat cases in the
    # same batch.
    dataset = (dataset.shuffle(num_examples).repeat()
               .batch(max_batch_size).prefetch(5))
    dataset = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    # This version of CachedDataReader requires the dataset to be shuffled
    return data_reader.CachedDataReader(dataset, max_batch_size), num_examples

  else:
    # Version 2 using data_reader_alt.py (faster)
    images, labels, num_examples = mnist.load_mnist_as_tensors(
        flatten_images=True)
    dataset = (images, labels)

    # This version of CachedDataReader requires the dataset to NOT be shuffled
    return data_reader_alt.CachedDataReader(dataset, num_examples), num_examples


def _get_batch_size_schedule(num_examples):
  """Returns training batch size schedule."""
  minibatch_maxsize_targetiter = 100  # We use a smaller target iter here than
                                      # in the autoencoder example.
  minibatch_maxsize = num_examples
  minibatch_startsize = 1000

  div = (float(minibatch_maxsize_targetiter-1)
         / math.log(float(minibatch_maxsize)/minibatch_startsize, 2))
  return [
      min(int(2.**(float(k)/div) * minibatch_startsize), minibatch_maxsize)
      for k in range(minibatch_maxsize_targetiter)
  ]


def group_assign(dest, source):
  return tf.group(*(d.assign(s) for d, s in zip(dest, source)))


def construct_train_quants():

  with tf.device(FLAGS.device):
    # Load dataset.
    cached_reader, num_examples = load_mnist()
    batch_size_schedule = _get_batch_size_schedule(num_examples)
    batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')

    minibatch = cached_reader(batch_size)
    training_model = Model()
    layer_collection = kfac.LayerCollection()

    if FLAGS.use_sua_approx:
      layer_collection.set_default_conv2d_approximation('kron_sua')

    ema = tf.train.ExponentialMovingAverage(FLAGS.polyak_decay,
                                            zero_debias=True)

    def loss_fn(minibatch, layer_collection=None, return_error=False):
      features, labels = minibatch
      logits = training_model(features)
      return compute_loss(
          logits=logits,
          labels=labels,
          layer_collection=layer_collection,
          return_error=return_error)

    (batch_loss, batch_error) = loss_fn(
        minibatch, layer_collection=layer_collection, return_error=True)

    train_vars = training_model.variables

    # Make training op:
    train_op, opt = make_train_op(
        minibatch,
        batch_size,
        batch_loss,
        layer_collection,
        loss_fn=loss_fn,
        prev_train_batch=cached_reader.cached_batch)

    with tf.control_dependencies([train_op]):
      train_op = ema.apply(train_vars)

    # Make eval ops:
    images, labels, num_examples = mnist.load_mnist_as_tensors(
        flatten_images=True)

    eval_model = Model()
    eval_model(images)  # We need this dummy call because for some reason the
                        # variables won't exist otherwise...
    eval_vars = eval_model.variables

    update_eval_model = group_assign(eval_vars, train_vars)

    with tf.control_dependencies([update_eval_model]):
      logits = eval_model(images)
      eval_loss, eval_error = compute_loss(
          logits=logits, labels=labels, return_error=True)

      with tf.control_dependencies([eval_loss, eval_error]):
        update_eval_model_avg = group_assign(
            eval_vars, (ema.average(t) for t in train_vars))

        with tf.control_dependencies([update_eval_model_avg]):
          logits = eval_model(images)
          eval_loss_avg, eval_error_avg = compute_loss(
              logits=logits, labels=labels, return_error=True)

  return (train_op, opt, batch_loss, batch_error, batch_size_schedule,
          batch_size, eval_loss, eval_error, eval_loss_avg, eval_error_avg)


def main(_):

  # If using update_damping_immediately resource variables must be enabled.
  if FLAGS.update_damping_immediately:
    tf.enable_resource_variables()

  if not FLAGS.use_sua_approx:
    # Temporary measure to save memory with giant batches
    kfac.fisher_factors.set_global_constants(
        sub_sample_inputs=True,
        inputs_to_extract_patches_factor=0.1)

  tf.set_random_seed(FLAGS.seed)
  (train_op, opt, batch_loss, batch_error, batch_size_schedule, batch_size,
   eval_loss, eval_error,
   eval_loss_avg, eval_error_avg) = construct_train_quants()

  global_step = tf.train.get_or_create_global_step()

  if FLAGS.optimizer == 'kfac':
    # We need to put the control depenency on train_op here so that we are
    # guaranteed to get the up-to-date values of these various quantities.
    # Otherwise there is a race condition and we might get the old values,
    # nondeterministically. Another solution would be to get these values in
    # a separate sess.run call, but this can sometimes cause problems with
    # training frameworks that use hooks (see the comments below).
    with tf.control_dependencies([train_op]):
      learning_rate = opt.learning_rate
      momentum = opt.momentum
      damping = opt.damping
      rho = opt.rho
      qmodel_change = opt.qmodel_change

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  # Train model.

  # It's good practice to put everything into a single sess.run call. The
  # reason is that certain "training frameworks" like to run hooks at each
  # sess.run call, and there is an implicit expectation there will only
  # be one sess.run call every "iteration" of the "optimizer". For example,
  # a framework might try to print the loss at each sess.run call, causing
  # the mini-batch to be advanced, thus completely breaking the "cached
  # batch" mechanism that the damping adaptation method may rely on. (Plus
  # there will also be the extra cost of having to reevaluate the loss
  # twice.)  That being said we don't completely do that here because it's
  # inconvenient.
  with tf.train.MonitoredTrainingSession(save_checkpoint_secs=30,
                                         config=config) as sess:
    for _ in range(FLAGS.train_steps):
      i = sess.run(global_step)

      if FLAGS.use_batch_size_schedule:
        batch_size_ = batch_size_schedule[min(i, len(batch_size_schedule) - 1)]
      else:
        batch_size_ = FLAGS.batch_size

      if FLAGS.optimizer == 'kfac':
        (_, batch_loss_, batch_error_, learning_rate_, momentum_, damping_,
         rho_, qmodel_change_) = sess.run([train_op, batch_loss, batch_error,
                                           learning_rate, momentum, damping,
                                           rho, qmodel_change],
                                          feed_dict={batch_size: batch_size_})
      else:
        _, batch_loss_, batch_error_ = sess.run(
            [train_op, batch_loss, batch_error],
            feed_dict={batch_size: batch_size_})

      # Print training stats.
      tf.logging.info(
          'iteration: %d', i)
      tf.logging.info(
          'mini-batch size: %d | mini-batch loss = %f | mini-batch error = %f ',
          batch_size_, batch_loss_, batch_error_)

      if FLAGS.optimizer == 'kfac':
        tf.logging.info(
            'learning_rate = %f | momentum = %f',
            learning_rate_, momentum_)
        tf.logging.info(
            'damping = %f | rho = %f | qmodel_change = %f',
            damping_, rho_, qmodel_change_)

      # "Eval" here means just compute stuff on the full training set.
      if (i+1) % FLAGS.eval_every == 0:
        eval_loss_, eval_error_, eval_loss_avg_, eval_error_avg_ = sess.run(
            [eval_loss, eval_error, eval_loss_avg, eval_error_avg])
        tf.logging.info('-----------------------------------------------------')
        tf.logging.info('eval_loss = %f | eval_error = %f',
                        eval_loss_, eval_error_)
        tf.logging.info('eval_loss_avg = %f | eval_error_avg = %f',
                        eval_loss_avg_, eval_error_avg_)
        tf.logging.info('-----------------------------------------------------')
      else:
        tf.logging.info('----')


if __name__ == '__main__':
  tf.app.run(main)
