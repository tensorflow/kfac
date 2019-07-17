# Distributed Training

## Table of Contents
  * [Register the layers](#register-the-layers)
  * [Build the optimizer](#build-the-optimizer)
  * [Fit the model](#fit-the-model)
  * [TIPS](#tips)
<br>

This example showcases how to use K-FAC in a distributed setting using
`SyncReplicas` optimizer. While most methods benefit from increased compute,
K-FAC particularly shines as the number of workers (and, in turn, batch size)
increases.

**Note:** This tutorial extends the single-machine
[Convolutional example][conv_ex] to distributed training. It is highly
recommended you read that first, as shared bits are omitted below!

[conv_ex]:
https://github.com/tensorflow/kfac/tree/master/docs/examples/convolutional.md

**Note:** This tutorial expects you to be familiar with distributed training.
Check out https://www.tensorflow.org/deploy/distributed if this is new to you.

**Example code**:
https://github.com/tensorflow/kfac/tree/master/kfac/examples/convnet_mnist_distributed_main.py

## Build the Model

When training on a single machine, one doesn't need to think about which
"device" a variable is placed on (there's only 1 to choose from!). In a
distributed setting, variables live on ["Parameter Servers"][parameter-servers].
Placing a variable on a parameter server is as simple as using
`tf.train.replica_device_setter()`, which is illustrated in the below code.

```python
  with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
    pre0, act0, w0, b0 = conv_layer(
        layer_id=0, inputs=examples, kernel_size=5, out_channels=16)
    act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=3, stride=2)
    ...
```

[parameter-servers]: https://www.tensorflow.org/deploy/distributed

## Register the layers

Layer registration is identical to the single-machine case. See ["Register the
layers"][register-layers-conv] in the Convolutional example for details.

[register-layers-conv]: https://github.com/tensorflow/kfac/tree/master/docs/examples/convolutional.md?#register-the-layers-and-loss

## Build the optimizer

Like the model itself, the K-FAC optimizer also creates variables. Don't forget
to wrap it in a similar `replica_device_setter()` too!

```python
  with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
    ...
    optimizer = opt.KfacOptimizer(
        learning_rate=0.0001,
        cov_ema_decay=0.95,
        damping=0.001,
        layer_collection=layer_collection,
        momentum=0.9)
    ...
```

## Fit the model

When training on a single-machine, a single training loop is responsible for
executing all of K-FAC's training operations: updating weights, updating
statistics, and inverting the preconditioner matrix. As all of the work happens
on a single machine, one stands little to gain by parallelization.

There are different strategies of parallelizing the gradient, covariance and
inverse computation across workers in a distributed setting. We will illustrate
here two such strategies that work specifically with `SyncReplicas` optimizer
for distributed training.

The first strategy for distributed training is to compute gradient in a
distributed fashion across all the workers, but have the inverse and covariance
ops executed only on the chief worker.

**Code**:
https://github.com/tensorflow/kfac/tree/master/kfac/examples/convnet.py

```python
  optimizer = opt.KfacOptimizer(...)
  sync_optimizer = tf.train.SyncReplicasOptimizer(opt=optimizer, ...)
  (cov_update_thunks, inv_update_thunks) = optimizer.make_vars_and_create_op_thunks()

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
```

In the second strategy, each worker's training loop is responsible for executing
only one of K-FAC's three training ops,

1.  Compute gradients.
1.  Workers updating covariance matrices can asynchronously update the moving
    average similar to the way asynchronous SGD updates weights.
1.  Workers inverting the preconditioning matrix can independently and
    asynchronously invert its blocks, one at a time. Blocks are chosen according
    to a randomly shuffled queue.

```python
  optimizer = opt.KfacOptimizer(...)
  inv_update_queue = oq.OpQueue(optimizer.inv_updates_dict.values())
  sync_optimizer = tf.train.SyncReplicasOptimizer(
      opt=optimizer,
      replicas_to_aggregate=_num_gradient_tasks(num_worker_tasks))
  train_op = sync_optimizer.minimize(loss, global_step=global_step)

  with tf.train.MonitoredTrainingSession(...) as sess:
    while not sess.should_stop():
      if _is_gradient_task(task_id, num_worker_tasks):
        learning_op = train_op
      elif _is_cov_update_task(task_id, num_worker_tasks):
        learning_op = optimizer.cov_update_op
      elif _is_inv_update_task(task_id, num_worker_tasks):
        learning_op = inv_update_queue.next_op(sess)

      global_step_, loss_, statistics_, _ = sess.run(
          [global_step, loss, statistics, learning_op])
```

## TIPS

1.  Check the [hyper params tuning][hp_tune] section for more details on tuning
    various KFAC parameters.

[hp_tune]: https://github.com/tensorflow/kfac/tree/master/docs/examples/parameters.md
