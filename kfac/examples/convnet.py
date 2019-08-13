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
"""Train a ConvNet on MNIST using K-FAC.

This library demonstrates how to use K-FAC to train a 5-layer ConvNet on MNIST
using K-FAC.

Note that this example is basically untuned and is not meant to work as an
actual demonstration of the power of the method. It may not even converge. It
merely demonstrates the how to set up K-FAC to run under the various standard
modes of operation in Tensorflow, like SyncReplicas, Estimator, etc.

For an example of the method tuned properly and working well, see for example
the autoencoder_mnist.py example, which replicates the exact experiment from
the original K-FAC paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import kfac
import numpy as np
import tensorflow as tf

from kfac.examples import mnist


__all__ = [
    "conv_layer",
    "fc_layer",
    "max_pool_layer",
    "build_model",
    "minimize_loss_single_machine",
    "distributed_grads_only_and_ops_chief_worker",
    "distributed_grads_and_ops_dedicated_workers",
    "train_mnist_single_machine",
    "train_mnist_distributed_sync_replicas",
    "train_mnist_multitower"
]


# Inverse update ops will be run every _INVERT_EVRY iterations.
_INVERT_EVERY = 10

# Covariance matrices will be update  _COV_UPDATE_EVERY iterations.
_COV_UPDATE_EVERY = 1

# Displays loss every _REPORT_EVERY iterations.
_REPORT_EVERY = 10

# Use manual registration
_USE_MANUAL_REG = False


def fc_layer(layer_id, inputs, output_size):
  """Builds a fully connected layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, input_size]. Each row corresponds
      to a single example.
    output_size: int. Number of output dimensions after fully connected layer.

  Returns:
    preactivations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately before the activation function.
    activations: Tensor of shape [num_examples, output_size]. Values of the
      layer immediately after the activation function.
    params: Tuple of (weights, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Dense(
      output_size,
      kernel_initializer=tf.random_normal_initializer(),
      name="fc_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.tanh(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def conv_layer(layer_id, inputs, kernel_size, out_channels):
  """Builds a convolutional layer with ReLU non-linearity.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height of the convolution kernel. The kernel is
      assumed to be square.
    out_channels: int. Number of output features per pixel.

  Returns:
    preactivations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately before the activation function.
    activations: Tensor of shape [num_examples, width, height, out_channels].
      Values of the layer immediately after the activation function.
    params: Tuple of (kernel, bias), parameters for this layer.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  layer = tf.layers.Conv2D(
      out_channels,
      kernel_size=[kernel_size, kernel_size],
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding="SAME",
      name="conv_%d" % layer_id)
  preactivations = layer(inputs)
  activations = tf.nn.relu(preactivations)

  # layer.weights is a list. This converts it a (hashable) tuple.
  return preactivations, activations, (layer.kernel, layer.bias)


def max_pool_layer(layer_id, inputs, kernel_size, stride):
  """Build a max-pooling layer.

  Args:
    layer_id: int. Integer ID for this layer's variables.
    inputs: Tensor of shape [num_examples, width, height, in_channels]. Each row
      corresponds to a single example.
    kernel_size: int. Width and height to pool over per input channel. The
      kernel is assumed to be square.
    stride: int. Step size between pooling operations.

  Returns:
    Tensor of shape [num_examples, width/stride, height/stride, out_channels].
    Result of applying max pooling to 'inputs'.
  """
  # TODO(b/67004004): Delete this function and rely on tf.layers exclusively.
  with tf.variable_scope("pool_%d" % layer_id):
    return tf.nn.max_pool(
        inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
        padding="SAME",
        name="pool")


def build_model(examples,
                labels,
                num_labels,
                layer_collection,
                register_layers_manually=False):
  """Builds a ConvNet classification model.

  Args:
    examples: Tensor of shape [num_examples, num_features]. Represents inputs of
      model.
    labels: Tensor of shape [num_examples]. Contains integer IDs to be predicted
      by softmax for each example.
    num_labels: int. Number of distinct values 'labels' can take on.
    layer_collection: LayerCollection instance. Layers will be registered here.
    register_layers_manually: bool. If True then register the layers with
      layer_collection manually. (Default: False)

  Returns:
    loss: 0-D Tensor representing loss to be minimized.
    accuracy: 0-D Tensor representing model's accuracy.
  """
  # Build a ConvNet. For each layer with parameters, we'll keep track of the
  # preactivations, activations, weights, and bias.
  tf.logging.info("Building model.")
  pre0, act0, params0 = conv_layer(
      layer_id=0, inputs=examples, kernel_size=5, out_channels=16)
  act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=3, stride=2)
  pre2, act2, params2 = conv_layer(
      layer_id=2, inputs=act1, kernel_size=5, out_channels=16)
  act3 = max_pool_layer(layer_id=3, inputs=act2, kernel_size=3, stride=2)
  flat_act3 = tf.reshape(act3, shape=[-1, int(np.prod(act3.shape[1:4]))])
  logits, _, params4 = fc_layer(
      layer_id=4, inputs=flat_act3, output_size=num_labels)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.cast(labels, dtype=tf.int32),
                       tf.argmax(logits, axis=1, output_type=tf.int32)),
              dtype=tf.float32))

  with tf.device("/cpu:0"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

  layer_collection.register_softmax_cross_entropy_loss(
      logits, name="logits")

  if register_layers_manually:
    layer_collection.register_conv2d(params0, (1, 1, 1, 1), "SAME", examples,
                                     pre0)
    layer_collection.register_conv2d(params2, (1, 1, 1, 1), "SAME", act1,
                                     pre2)
    layer_collection.register_fully_connected(params4, flat_act3, logits)

  return loss, accuracy


def minimize_loss_single_machine(loss,
                                 accuracy,
                                 layer_collection,
                                 device=None,
                                 session_config=None):
  """Minimize loss with K-FAC on a single machine.

  Creates `PeriodicInvCovUpdateKfacOpt` which handles inverse and covariance
  computation op placement and execution. A single Session is responsible for
  running all of K-FAC's ops. The covariance and inverse update ops are placed
  on `device`. All model variables are on CPU.

  Args:
    loss: 0-D Tensor. Loss to be minimized.
    accuracy: 0-D Tensor. Accuracy of classifier on current minibatch.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.
    device: string or None. The covariance and inverse update ops are run on
      this device. If empty or None, the default device will be used.
      (Default: None)
    session_config: None or tf.ConfigProto. Configuration for tf.Session().

  Returns:
    final value for 'accuracy'.
  """
  device_list = [] if not device else [device]

  # Train with K-FAC.
  g_step = tf.train.get_or_create_global_step()
  optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=_INVERT_EVERY,
      cov_update_every=_COV_UPDATE_EVERY,
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=device_list,
      inv_devices=device_list,
      trans_devices=device_list,
      momentum=0.9)

  with tf.device(device):
    train_op = optimizer.minimize(loss, global_step=g_step)

  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession(config=session_config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _ = sess.run(
          [g_step, loss, accuracy, train_op])

      if global_step_ % _REPORT_EVERY == 0:
        tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
                        global_step_, loss_, accuracy_)

  return accuracy_


def minimize_loss_single_machine_manual(loss,
                                        accuracy,
                                        layer_collection,
                                        device=None,
                                        session_config=None):
  """Minimize loss with K-FAC on a single machine(Illustrative purpose only).

  This function does inverse and covariance computation manually
  for illustrative pupose. Check `minimize_loss_single_machine` for
  automatic inverse and covariance op placement and execution.
  A single Session is responsible for running all of K-FAC's ops. The covariance
  and inverse update ops are placed on `device`. All model variables are on CPU.

  Args:
    loss: 0-D Tensor. Loss to be minimized.
    accuracy: 0-D Tensor. Accuracy of classifier on current minibatch.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.
    device: string or None. The covariance and inverse update ops are run on
      this device. If empty or None, the default device will be used.
      (Default: None)
    session_config: None or tf.ConfigProto. Configuration for tf.Session().

  Returns:
    final value for 'accuracy'.
  """
  device_list = [] if not device else [device]

  # Train with K-FAC.
  g_step = tf.train.get_or_create_global_step()
  optimizer = kfac.KfacOptimizer(
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=device_list,
      inv_devices=device_list,
      trans_devices=device_list,
      momentum=0.9)
  (cov_update_thunks,
   inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()

  def make_update_op(update_thunks):
    update_ops = [thunk() for thunk in update_thunks]
    return tf.group(*update_ops)

  cov_update_op = make_update_op(cov_update_thunks)
  with tf.control_dependencies([cov_update_op]):
    inverse_op = tf.cond(
        tf.equal(tf.mod(g_step, _INVERT_EVERY), 0),
        lambda: make_update_op(inv_update_thunks), tf.no_op)
    with tf.control_dependencies([inverse_op]):
      with tf.device(device):
        train_op = optimizer.minimize(loss, global_step=g_step)

  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession(config=session_config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _ = sess.run(
          [g_step, loss, accuracy, train_op])

      if global_step_ % _REPORT_EVERY == 0:
        tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
                        global_step_, loss_, accuracy_)

  return accuracy_


def _is_gradient_task(task_id, num_tasks):
  """Returns True if this task should update the weights."""
  if num_tasks < 3:
    return True
  return 0 <= task_id < 0.6 * num_tasks


def _is_cov_update_task(task_id, num_tasks):
  """Returns True if this task should update K-FAC's covariance matrices."""
  if num_tasks < 3:
    return False
  return 0.6 * num_tasks <= task_id < num_tasks - 1


def _is_inv_update_task(task_id, num_tasks):
  """Returns True if this task should update K-FAC's preconditioner."""
  if num_tasks < 3:
    return False
  return task_id == num_tasks - 1


def _num_gradient_tasks(num_tasks):
  """Number of tasks that will update weights."""
  if num_tasks < 3:
    return num_tasks
  return int(np.ceil(0.6 * num_tasks))


def _make_distributed_train_op(
    task_id,
    num_worker_tasks,
    num_ps_tasks,
    layer_collection
):
  """Creates optimizer and distributed training op.

  Constructs KFAC optimizer and wraps it in `sync_replicas` optimizer. Makes
  the train op.

  Args:
   task_id: int. Integer in [0, num_worker_tasks). ID for this worker.
    num_worker_tasks: int. Number of workers in this distributed training setup.
    num_ps_tasks: int. Number of parameter servers holding variables. If 0,
      parameter servers are not used.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.

  Returns:
    sync_optimizer: `tf.train.SyncReplicasOptimizer` instance which wraps KFAC
      optimizer.
    optimizer: Instance of `KfacOptimizer`.
    global_step: `tensor`, Global step.
  """
  tf.logging.info("Task id : %d", task_id)
  with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
    global_step = tf.train.get_or_create_global_step()
    optimizer = kfac.KfacOptimizer(
        learning_rate=0.0001,
        cov_ema_decay=0.95,
        damping=0.001,
        layer_collection=layer_collection,
        momentum=0.9)
    sync_optimizer = tf.train.SyncReplicasOptimizer(
        opt=optimizer,
        replicas_to_aggregate=_num_gradient_tasks(num_worker_tasks),
        total_num_replicas=num_worker_tasks)
    return sync_optimizer, optimizer, global_step


def distributed_grads_only_and_ops_chief_worker(
    task_id, is_chief, num_worker_tasks, num_ps_tasks, master, checkpoint_dir,
    loss, accuracy, layer_collection, invert_every=10):
  """Minimize loss with a synchronous implementation of K-FAC.

  All workers perform gradient computation. Chief worker applies gradient after
  averaging the gradients obtained from all the workers. All workers block
  execution until the update is applied. Chief worker runs covariance and
  inverse update ops. Covariance and inverse matrices are placed on parameter
  servers in a round robin manner. For further details on synchronous
  distributed optimization check `tf.train.SyncReplicasOptimizer`.

  Args:
    task_id: int. Integer in [0, num_worker_tasks). ID for this worker.
    is_chief: `boolean`, `True` if the worker is chief worker.
    num_worker_tasks: int. Number of workers in this distributed training setup.
    num_ps_tasks: int. Number of parameter servers holding variables. If 0,
      parameter servers are not used.
    master: string. IP and port of TensorFlow runtime process. Set to empty
      string to run locally.
    checkpoint_dir: string or None. Path to store checkpoints under.
    loss: 0-D Tensor. Loss to be minimized.
    accuracy: dict mapping strings to 0-D Tensors. Additional accuracy to
      run with each step.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.
    invert_every: `int`, Number of steps between update the inverse.

  Returns:
    final value for 'accuracy'.

  Raises:
    ValueError: if task_id >= num_worker_tasks.
  """

  sync_optimizer, optimizer, global_step = _make_distributed_train_op(
      task_id, num_worker_tasks, num_ps_tasks, layer_collection)
  (cov_update_thunks,
   inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()

  tf.logging.info("Starting training.")
  hooks = [sync_optimizer.make_session_run_hook(is_chief)]

  def make_update_op(update_thunks):
    update_ops = [thunk() for thunk in update_thunks]
    return tf.group(*update_ops)

  if is_chief:
    cov_update_op = make_update_op(cov_update_thunks)
    with tf.control_dependencies([cov_update_op]):
      inverse_op = tf.cond(
          tf.equal(tf.mod(global_step, invert_every), 0),
          lambda: make_update_op(inv_update_thunks),
          tf.no_op)
      with tf.control_dependencies([inverse_op]):
        train_op = sync_optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = sync_optimizer.minimize(loss, global_step=global_step)

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=checkpoint_dir,
      hooks=hooks,
      stop_grace_period_secs=0,
      config=config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _ = sess.run(
          [global_step, loss, accuracy, train_op])
      tf.logging.info("global_step: %d | loss: %f | accuracy: %s", global_step_,
                      loss_, accuracy_)
  return accuracy_


def distributed_grads_and_ops_dedicated_workers(
    task_id, is_chief, num_worker_tasks, num_ps_tasks, master, checkpoint_dir,
    loss, accuracy, layer_collection):
  """Minimize loss with a synchronous implementation of K-FAC.

  Different workers are responsible for different parts of K-FAC's Ops. The
  first 60% of tasks compute gradients; the next 20% accumulate covariance
  statistics; the last 20% invert the matrices used to precondition gradients.
  The chief worker computes and applies the update.

  Args:
    task_id: int. Integer in [0, num_worker_tasks). ID for this worker.
    is_chief: `boolean`, `True` if the worker is chief worker.
    num_worker_tasks: int. Number of workers in this distributed training setup.
    num_ps_tasks: int. Number of parameter servers holding variables. If 0,
      parameter servers are not used.
    master: string. IP and port of TensorFlow runtime process. Set to empty
      string to run locally.
    checkpoint_dir: string or None. Path to store checkpoints under.
    loss: 0-D Tensor. Loss to be minimized.
    accuracy: dict mapping strings to 0-D Tensors. Additional accuracy to
      run with each step.
    layer_collection: LayerCollection instance describing model architecture.
      Used by K-FAC to construct preconditioner.

  Returns:
    final value for 'accuracy'.

  Raises:
    ValueError: if task_id >= num_worker_tasks.
  """
  sync_optimizer, optimizer, global_step = _make_distributed_train_op(
      task_id, num_worker_tasks, num_ps_tasks, layer_collection)
  _, cov_update_op, inv_update_ops, _, _, _ = optimizer.make_ops_and_vars()
  train_op = sync_optimizer.minimize(loss, global_step=global_step)
  inv_update_queue = kfac.op_queue.OpQueue(inv_update_ops)

  tf.logging.info("Starting training.")
  is_chief = (task_id == 0)
  hooks = [sync_optimizer.make_session_run_hook(is_chief)]

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=checkpoint_dir,
      hooks=hooks,
      stop_grace_period_secs=0,
      config=config) as sess:
    while not sess.should_stop():
      # Choose which op this task is responsible for running.
      if _is_gradient_task(task_id, num_worker_tasks):
        learning_op = train_op
      elif _is_cov_update_task(task_id, num_worker_tasks):
        learning_op = cov_update_op
      elif _is_inv_update_task(task_id, num_worker_tasks):
        learning_op = inv_update_queue.next_op(sess)
      else:
        raise ValueError("Which op should task %d do?" % task_id)

      global_step_, loss_, accuracy_, _ = sess.run(
          [global_step, loss, accuracy, learning_op])
      tf.logging.info("global_step: %d | loss: %f | accuracy: %s", global_step_,
                      loss_, accuracy_)

  return accuracy_


def train_mnist_single_machine(num_epochs,
                               use_fake_data=False,
                               device=None,
                               manual_op_exec=False):
  """Train a ConvNet on MNIST.

  Args:
    num_epochs: int. Number of passes to make over the training set.
    use_fake_data: bool. If True, generate a synthetic dataset.
    device: string or None. The covariance and inverse update ops are run on
      this device. If empty or None, the default device will be used.
      (Default: None)
    manual_op_exec: bool, If `True` then `minimize_loss_single_machine_manual`
      is called for training which handles inverse and covariance computation.
      This is shown only for illustrative purpose. Otherwise
      `minimize_loss_single_machine` is called which relies on
      `PeriodicInvCovUpdateOpt` for op placement and execution.

  Returns:
    accuracy of model on the final minibatch of training data.
  """
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")
  (examples, labels) = mnist.load_mnist_as_iterator(num_epochs,
                                                    128,
                                                    use_fake_data=use_fake_data,
                                                    flatten_images=False)

  # Build a ConvNet.
  layer_collection = kfac.LayerCollection()
  loss, accuracy = build_model(
      examples, labels, num_labels=10, layer_collection=layer_collection,
      register_layers_manually=_USE_MANUAL_REG)
  if not _USE_MANUAL_REG:
    layer_collection.auto_register_layers()

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  config = tf.ConfigProto(allow_soft_placement=True)

  # Fit model.
  if manual_op_exec:
    return minimize_loss_single_machine_manual(
        loss, accuracy, layer_collection, device=device, session_config=config)
  else:
    return minimize_loss_single_machine(
        loss, accuracy, layer_collection, device=device, session_config=config)


def train_mnist_multitower(num_epochs, num_towers,
                           devices, use_fake_data=False, session_config=None):
  """Train a ConvNet on MNIST.

  Training data is split equally among the towers. Each tower computes loss on
  its own batch of data and the loss is aggregated on the CPU. The model
  variables are placed on first tower. The covariance and inverse update ops
  and variables are placed on specified devices in a round robin manner.

  Args:
    num_epochs: int. Number of passes to make over the training set.
    num_towers: int. Number of towers.
    devices: list of strings. List of devices to place the towers.
    use_fake_data: bool. If True, generate a synthetic dataset.
    session_config: None or tf.ConfigProto. Configuration for tf.Session().

  Returns:
    accuracy of model on the final minibatch of training data.
  """
  num_towers = 1 if not devices else len(devices)
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")
  tower_batch_size = 128
  batch_size = tower_batch_size * num_towers
  tf.logging.info(
      ("Loading MNIST into memory. Using batch_size = %d = %d towers * %d "
       "tower batch size.") % (batch_size, num_towers, tower_batch_size))
  (examples, labels) = mnist.load_mnist_as_iterator(num_epochs,
                                                    batch_size,
                                                    use_fake_data=use_fake_data,
                                                    flatten_images=False)

  # Split minibatch across towers.
  examples = tf.split(examples, num_towers)
  labels = tf.split(labels, num_towers)

  # Build an MLP. Each tower's layers will be added to the LayerCollection.
  layer_collection = kfac.LayerCollection()
  tower_results = []
  for tower_id in range(num_towers):
    with tf.device(devices[tower_id]):
      with tf.name_scope("tower%d" % tower_id):
        with tf.variable_scope(tf.get_variable_scope(), reuse=(tower_id > 0)):
          tf.logging.info("Building tower %d." % tower_id)
          tower_results.append(
              build_model(
                  examples[tower_id],
                  labels[tower_id],
                  10,
                  layer_collection,
                  register_layers_manually=_USE_MANUAL_REG))
  losses, accuracies = zip(*tower_results)
  # When using multiple towers we only want to perform automatic
  # registation once, after the final tower is made
  if not _USE_MANUAL_REG:
    layer_collection.auto_register_layers()

  # Average across towers.
  loss = tf.reduce_mean(losses)
  accuracy = tf.reduce_mean(accuracies)

  # Fit model.
  g_step = tf.train.get_or_create_global_step()
  optimizer = kfac.PeriodicInvCovUpdateKfacOpt(
      invert_every=_INVERT_EVERY,
      cov_update_every=_COV_UPDATE_EVERY,
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      cov_devices=devices,
      inv_devices=devices,
      trans_devices=devices,
      momentum=0.9)

  with tf.device(devices[0]):
    train_op = optimizer.minimize(loss, global_step=g_step)

  # Without setting allow_soft_placement=True there will be problems when
  # the optimizer tries to place certain ops like "mod" on the GPU (which isn't
  # supported).
  if not session_config:
    session_config = tf.ConfigProto(allow_soft_placement=True)

  tf.logging.info("Starting training.")
  with tf.train.MonitoredTrainingSession(config=session_config) as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _ = sess.run(
          [g_step, loss, accuracy, train_op])

      if global_step_ % _REPORT_EVERY == 0:
        tf.logging.info("global_step: %d | loss: %f | accuracy: %s",
                        global_step_, loss_, accuracy_)


def train_mnist_distributed_sync_replicas(task_id,
                                          is_chief,
                                          num_worker_tasks,
                                          num_ps_tasks,
                                          master,
                                          num_epochs,
                                          op_strategy,
                                          use_fake_data=False):
  """Train a ConvNet on MNIST using Sync replicas optimizer.

  Args:
    task_id: int. Integer in [0, num_worker_tasks). ID for this worker.
    is_chief: `boolean`, `True` if the worker is chief worker.
    num_worker_tasks: int. Number of workers in this distributed training setup.
    num_ps_tasks: int. Number of parameter servers holding variables.
    master: string. IP and port of TensorFlow runtime process.
    num_epochs: int. Number of passes to make over the training set.
    op_strategy: `string`, Strategy to run the covariance and inverse
      ops. If op_strategy == `chief_worker` then covariance and inverse
      update ops are run on chief worker otherwise they are run on dedicated
      workers.

    use_fake_data: bool. If True, generate a synthetic dataset.

  Returns:
    accuracy of model on the final minibatch of training data.

  Raises:
    ValueError: If `op_strategy` not in ["chief_worker", "dedicated_workers"].
  """
  # Load a dataset.
  tf.logging.info("Loading MNIST into memory.")
  (examples, labels) = mnist.load_mnist_as_iterator(num_epochs,
                                                    128,
                                                    use_fake_data=use_fake_data,
                                                    flatten_images=False)

  # Build a ConvNet.
  layer_collection = kfac.LayerCollection()
  with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
    loss, accuracy = build_model(
        examples, labels, num_labels=10, layer_collection=layer_collection,
        register_layers_manually=_USE_MANUAL_REG)
  if not _USE_MANUAL_REG:
    layer_collection.auto_register_layers()

  # Fit model.
  checkpoint_dir = None
  if op_strategy == "chief_worker":
    return distributed_grads_only_and_ops_chief_worker(
        task_id, is_chief, num_worker_tasks, num_ps_tasks, master,
        checkpoint_dir, loss, accuracy, layer_collection)
  elif op_strategy == "dedicated_workers":
    return distributed_grads_and_ops_dedicated_workers(
        task_id, is_chief, num_worker_tasks, num_ps_tasks, master,
        checkpoint_dir, loss, accuracy, layer_collection)
  else:
    raise ValueError("Only supported op strategies are : {}, {}".format(
        "chief_worker", "dedicated_workers"))


def train_mnist_estimator(num_epochs, use_fake_data=False):
  """Train a ConvNet on MNIST using tf.estimator.

  Args:
    num_epochs: int. Number of passes to make over the training set.
    use_fake_data: bool. If True, generate a synthetic dataset.

  Returns:
    accuracy of model on the final minibatch of training data.
  """

  # Load a dataset.
  def input_fn():
    tf.logging.info("Loading MNIST into memory.")
    return mnist.load_mnist_as_iterator(num_epochs=num_epochs,
                                        batch_size=64,
                                        flatten_images=False,
                                        use_fake_data=use_fake_data)

  def model_fn(features, labels, mode, params):
    """Model function for MLP trained with K-FAC.

    Args:
      features: Tensor of shape [batch_size, input_size]. Input features.
      labels: Tensor of shape [batch_size]. Target labels for training.
      mode: tf.estimator.ModeKey. Must be TRAIN.
      params: ignored.

    Returns:
      EstimatorSpec for training.

    Raises:
      ValueError: If 'mode' is anything other than TRAIN.
    """
    del params

    if mode != tf.estimator.ModeKeys.TRAIN:
      raise ValueError("Only training is supported with this API.")

    # Build a ConvNet.
    layer_collection = kfac.LayerCollection()
    loss, accuracy = build_model(
        features, labels, num_labels=10, layer_collection=layer_collection,
        register_layers_manually=_USE_MANUAL_REG)
    if not _USE_MANUAL_REG:
      layer_collection.auto_register_layers()

    # Train with K-FAC.
    global_step = tf.train.get_or_create_global_step()
    optimizer = kfac.KfacOptimizer(
        learning_rate=tf.train.exponential_decay(
            0.00002, global_step, 10000, 0.5, staircase=True),
        cov_ema_decay=0.95,
        damping=0.001,
        layer_collection=layer_collection,
        momentum=0.9)

    (cov_update_thunks,
     inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()

    def make_update_op(update_thunks):
      update_ops = [thunk() for thunk in update_thunks]
      return tf.group(*update_ops)

    def make_batch_executed_op(update_thunks, batch_size=1):
      return tf.group(*kfac.utils.batch_execute(
          global_step, update_thunks, batch_size=batch_size))

    # Run cov_update_op every step. Run 1 inv_update_ops per step.
    cov_update_op = make_update_op(cov_update_thunks)
    with tf.control_dependencies([cov_update_op]):
      # But make sure to execute all the inverse ops on the first step
      inverse_op = tf.cond(tf.equal(global_step, 0),
                           lambda: make_update_op(inv_update_thunks),
                           lambda: make_batch_executed_op(inv_update_thunks))
      with tf.control_dependencies([inverse_op]):
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Print metrics every 5 sec.
    hooks = [
        tf.train.LoggingTensorHook(
            {
                "loss": loss,
                "accuracy": accuracy
            }, every_n_secs=5),
    ]
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, training_hooks=hooks)

  run_config = tf.estimator.RunConfig(
      model_dir="/tmp/mnist", save_checkpoints_steps=1, keep_checkpoint_max=100)

  # Train until input_fn() is empty with Estimator. This is a prerequisite for
  # TPU compatibility.
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
  estimator.train(input_fn=input_fn)


if __name__ == "__main__":
  tf.app.run()
