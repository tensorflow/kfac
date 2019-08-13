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
"""RNN trained to do sequential MNIST classification using K-FAC.

This demonstrates the use of the RNN approximations from the paper
"Kronecker-factored Curvature Approximations for Recurrent Neural Networks".

The setup here is similar to the autoencoder example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# Dependency imports
from absl import flags
import kfac
import tensorflow as tf

from kfac.examples import mnist
from kfac.python.ops.kfac_utils import data_reader
from kfac.python.ops.kfac_utils import data_reader_alt


# We need this for now since linear layers without biases don't work with
# automatic scanning at the moment
_INCLUDE_INPUT_BIAS = True


flags.DEFINE_string('kfac_approx', 'kron_indep',
                    'The type of approximation to use for the recurrent '
                    'layers. "kron_indep" is the one which assumes '
                    'independence across time, "kron_series_1" is "Option 1" '
                    'from the paper, and "kron_series_2" is "Option 2".')

flags.DEFINE_integer('inverse_update_period', 5,
                     '# of steps between computing inverse of Fisher factor '
                     'matrices.')
flags.DEFINE_integer('cov_update_period', 1,
                     '# of steps between computing covaraiance matrices.')
flags.DEFINE_integer('damping_adaptation_interval', 5,
                     '# of steps between updating the damping parameter.')

flags.DEFINE_float('learning_rate', 3e-4,
                   'Learning rate to use when adaptation="off".')
flags.DEFINE_float('momentum', 0.9,
                   'Momentum decay value to use when '
                   'lrmu_adaptation="off" or "only_lr".')

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

flags.DEFINE_integer('num_hidden', 128, 'Hidden state dimension of the RNN.')

flags.DEFINE_boolean('use_auto_registration', False,
                     'Whether to use the automatic registration feature.')

flags.DEFINE_string('device', '/gpu:0',
                    'The device to run the major ops on.')


FLAGS = flags.FLAGS


def make_train_op(batch_size,
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

  optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=FLAGS.inverse_update_period,
      cov_update_every=FLAGS.cov_update_period,
      learning_rate=learning_rate,
      damping=150.,  # When using damping adaptation it is advisable to start
                     # with a high value. This value is probably far too high
                     # to use for most neural nets if you aren't using damping
                     # adaptation. (Although it always depends on the scale of
                     # the loss.)
      cov_ema_decay=0.95,
      momentum=momentum,
      momentum_type=momentum_type,
      layer_collection=layer_collection,
      batch_size=batch_size,
      num_burnin_steps=5,
      adapt_damping=True,
      is_chief=True,
      prev_train_batch=cached_reader.cached_batch,
      loss=batch_loss,
      loss_fn=loss_fn,
      damping_adaptation_decay=0.95,
      damping_adaptation_interval=FLAGS.damping_adaptation_interval,
      min_damping=1e-5
      )
  return optimizer.minimize(batch_loss, global_step=global_step), optimizer


def eval_model(x, num_classes, layer_collection=None):
  """Evaluate the model given the data and possibly register it."""

  num_hidden = FLAGS.num_hidden
  num_timesteps = x.shape[1]
  num_input = x.shape[2]

  # Strip off the annoying last dimension of size 1 (added for convenient use
  # with conv nets).
  x = x[..., 0]

  # Unstack to get a list of 'num_timesteps' tensors of
  # shape (batch_size, num_input)
  x_unstack = tf.unstack(x, num_timesteps, 1)

  # We need to do this manually without cells since we need to get access
  # to the pre-activations (i.e. the output of the "linear layers").
  w_in = tf.get_variable('w_in', shape=[num_input, num_hidden])
  if _INCLUDE_INPUT_BIAS:
    b_in = tf.get_variable('b_in', shape=[num_hidden])

  w_rec = tf.get_variable('w_rec', shape=[num_hidden, num_hidden])
  b_rec = tf.get_variable('b_rec', shape=[num_hidden])

  a = tf.zeros([tf.shape(x_unstack[0])[0], num_hidden], dtype=tf.float32)

  # Here 'a' are the activations, 's' the pre-activations
  a_list = []
  s_in_list = []
  s_rec_list = []
  s_list = []

  for input_ in x_unstack:

    a_list.append(a)

    s_in = tf.matmul(input_, w_in)
    if _INCLUDE_INPUT_BIAS:
      s_in += b_in
    s_rec = tf.matmul(a, w_rec) + b_rec
    # s_rec = b_rec + tf.matmul(a, w_rec)  # this breaks the graph scanner
    s = s_in + s_rec

    s_in_list.append(s_in)
    s_rec_list.append(s_rec)
    s_list.append(s)

    a = tf.tanh(s)

  final_rnn_output = a

  # NOTE: we can uncomment the lines below without changing how the algorithm
  # behaves.  This is because the derivative of the loss w.r.t. to s is the
  # the same as it is for both s_in and s_rec.  This can be seen easily from
  # the chain rule.
  #
  # s_rec_list = s_list
  # s_in_list = s_list

  if _INCLUDE_INPUT_BIAS:
    pin = (w_in, b_in)
  else:
    pin = w_in

  if layer_collection:
    layer_collection.register_fully_connected_multi(pin, x_unstack,
                                                    s_in_list,
                                                    approx=FLAGS.kfac_approx)

    layer_collection.register_fully_connected_multi((w_rec, b_rec), a_list,
                                                    s_rec_list,
                                                    approx=FLAGS.kfac_approx)

  # Output parameters (need this no matter how we construct the RNN):
  w_out = tf.get_variable('w_out', shape=[num_hidden, num_classes])
  b_out = tf.get_variable('b_out', shape=[num_classes])

  logits = tf.matmul(final_rnn_output, w_out) + b_out

  if layer_collection:
    layer_collection.register_fully_connected((w_out, b_out), final_rnn_output,
                                              logits)

  return logits


def compute_loss(inputs, labels, num_classes, layer_collection=None):
  """Compute loss value."""

  with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    if FLAGS.use_auto_registration:
      logits = eval_model(inputs, num_classes)
    else:
      logits = eval_model(inputs, num_classes,
                          layer_collection=layer_collection)

  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
  loss = tf.reduce_mean(losses)

  if layer_collection is not None:
    layer_collection.register_softmax_cross_entropy_loss(logits)
    if FLAGS.use_auto_registration:
      layer_collection.auto_register_layers()

  return loss


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
    dataset, num_examples = mnist.load_mnist_as_dataset(flatten_images=False)
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
        flatten_images=False)
    dataset = (images, labels)

    # This version of CachedDataReader requires the dataset to NOT be shuffled
    return data_reader_alt.CachedDataReader(dataset, num_examples), num_examples


def main(_):
  # Load dataset.
  cached_reader, num_examples = load_mnist()
  num_classes = 10

  minibatch_maxsize_targetiter = 500
  minibatch_maxsize = num_examples
  minibatch_startsize = 1000

  div = (float(minibatch_maxsize_targetiter-1)
         / math.log(float(minibatch_maxsize)/minibatch_startsize, 2))
  batch_size_schedule = [
      min(int(2.**(float(k)/div) * minibatch_startsize), minibatch_maxsize)
      for k in range(500)
  ]

  batch_size = tf.placeholder(shape=(), dtype=tf.int32, name='batch_size')

  layer_collection = kfac.LayerCollection()

  def loss_fn(minibatch, layer_collection=None):
    return compute_loss(minibatch[0], minibatch[1], num_classes,
                        layer_collection=layer_collection)

  minibatch = cached_reader(batch_size)
  batch_loss = loss_fn(minibatch, layer_collection=layer_collection)

  # Make training op
  with tf.device(FLAGS.device):
    train_op, opt = make_train_op(
        batch_size,
        batch_loss,
        layer_collection,
        loss_fn=loss_fn,
        cached_reader=cached_reader)

  learning_rate = opt.learning_rate
  momentum = opt.momentum
  damping = opt.damping
  rho = opt.rho
  qmodel_change = opt.qmodel_change
  global_step = tf.train.get_or_create_global_step()

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  # Train model.
  with tf.train.MonitoredTrainingSession(save_checkpoint_secs=30,
                                         config=config) as sess:
    while not sess.should_stop():
      i = sess.run(global_step)

      if FLAGS.use_batch_size_schedule:
        batch_size_ = batch_size_schedule[min(i, len(batch_size_schedule) - 1)]
      else:
        batch_size_ = FLAGS.batch_size

      _, batch_loss_ = sess.run([train_op, batch_loss],
                                feed_dict={batch_size: batch_size_})

      # We get these things in a separate sess.run() call because they are
      # stored as variables in the optimizer. (So there is no computational cost
      # to getting them, and if we don't get them after the previous call is
      # over they might not be updated.)
      (learning_rate_, momentum_, damping_, rho_,
       qmodel_change_) = sess.run([learning_rate, momentum, damping, rho,
                                   qmodel_change])

      # Print training stats.
      tf.logging.info(
          'iteration: %d', i)
      tf.logging.info(
          'mini-batch size: %d | mini-batch loss = %f',
          batch_size_, batch_loss_)
      tf.logging.info(
          'learning_rate = %f | momentum = %f',
          learning_rate_, momentum_)
      tf.logging.info(
          'damping = %f | rho = %f | qmodel_change = %f',
          damping_, rho_, qmodel_change_)
      tf.logging.info('----')


if __name__ == '__main__':
  tf.app.run(main)


