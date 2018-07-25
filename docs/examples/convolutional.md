# Convolutional

[TOC]

K-FAC needs to know about the structure of your model in order to effectively
optimize it. In particular, it needs to know about:

1.  Each convolutional and feed forward layer's inputs and outputs.
1.  All of the model parameters.
1.  The type of the loss function and its inputs.

Let's explore how we can use K-FAC to solve digit classification with MNIST
using a simple convolutional model. In the following example we will illustrate
how to use `PeriodicInvCovUpdateOpt` which is a subclass of `KfacOptimizer`.
`PeriodicInvCovUpdateOpt` handles placement and execution of covariance and
inverse ops. We will also illustrate how to register the layers both manually
and automatically using the graph scanner.

**Code**:
https://github.com/tensorflow/kfac/tree/master/kfac/examples/convnet_mnist_single_main.py

## Build the Model

First, we begin by defining a model. In this case, we'll load MNIST and
construct a 5-layer ConvNet. The model has 2 Conv/MaxPool pairs and a final
linear layer. If we are registering the layers manually we need to keep the
inputs and outputs and parameters (weights & bias) around, which is illustrated
here.

```python
  # Load a dataset.
  examples, labels = mnist.load_mnist(
      data_dir,
      num_epochs=num_epochs,
      batch_size=128,
      use_fake_data=use_fake_data,
      flatten_images=False)

  # Build a ConvNet.
  pre0, act0, params0 = conv_layer(
      layer_id=0, inputs=examples, kernel_size=5, out_channels=16)
  act1 = max_pool_layer(layer_id=1, inputs=act0, kernel_size=3, stride=2)
  pre2, act2, params2 = conv_layer(
      layer_id=2, inputs=act1, kernel_size=5, out_channels=16)
  act3 = max_pool_layer(layer_id=3, inputs=act2, kernel_size=3, stride=2)
  flat_act3 = tf.reshape(act3, shape=[-1, int(np.prod(act3.shape[1:4]))])
  logits, params4 = linear_layer(
      layer_id=4, inputs=flat_act3, output_size=num_labels)
  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits))
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(labels, tf.argmax(logits, axis=1)), dtype=tf.float32))
```

## Register the layers and loss

`layer_collection.auto_register_layers` automatically registers all the layers
for typical/standard models. However one must still manually register the loss
function. In the case of cross-entropy loss functions on softmaxes this amounts
to calling `layer_collection.register_categorical_predictive_distribution` with
the logits as an argument. Note that the inputs/outputs of non-parameterized
layers such as max pooling and reshaping _do not_ need to be registered.

```python
  # Register parameters with graph_search.
  tf.logging.info("Building KFAC Optimizer.")
  layer_collection = lc.LayerCollection()
  layer_collection.register_categorical_predictive_distribution(logits)
  # Set the layer at params0 to use a diagonal approximation
  # instead of default Kronecker factor based approximation.
  layer_collection.define_linked_parameters(
        params0, approximation=layer_collection.APPROX_DIAGONAL_NAME)
  layer_collection.auto_register_layers()
```

In the example above we demonstrate how to use a non-default Fisher
approximation (diagonal) for one of the conv layers. (The default is usually
Kronecker-factored.) This is done by calling
`layer_collection.define_linked_parameters`, which identifies the given
variables as being part of a particular layer, and sets the approximation that
is to be used for that layer. Any registrations performed later, whether done by
the graph scanner or performed manually by the user, will use this approximation
(unless overridden by the `approx` argument to the registration function).

Layers can also be registered manually. This is required for types of layers
that the automatic graph scanner doesn't recognize.

Note that One can also use a combination of manual and automatic registration by
calling `auto_register_layers()` after performing some manual registration. Any
layers registered manually before will be ignored by the scanner. We register
each layer's inputs, outputs, and parameters with an instance of
`LayerCollection`. For convolution layers, we use `register_conv2d`. For fully
connected (or linear) layers, `register_fully_connected`.

```python
  # Register parameters manually.
  tf.logging.info("Building KFAC Optimizer.")
  layer_collection = lc.LayerCollection()
  layer_collection.register_categorical_predictive_distribution(logits)

  layer_collection.register_conv2d(params0, (1, 1, 1, 1), "SAME", examples,
                                   pre0,
                                   approx=kfac_ff.APPROX_DIAGONAL_NAME)
  layer_collection.register_conv2d(params2, (1, 1, 1, 1), "SAME", act1, pre2)
  layer_collection.register_fully_connected(params4, flat_act3, logits)
```

In this example we demonstrate how to use a non-default Fisher approximation
(diagonal) for one of the layers. (The default is usually Kronecker-factored.)
This is done by passing `approx=kfac_ff.APPROX_DIAGONAL_NAME` to the
registration function `layer_collection.register_conv2d`. Note that if One has
already used `define_linked_parameters` to set the approximation then it is not
required to specify it again via the `approx` argument.

## Build the optimizer

Finally, we instantiate the optimizer. In addition to the `learning_rate` and
`momentum`, the optimizer has 2 additional hyperparameters,

1.  `cov_ema_decay`: Check [hyper parameters][hyper_params] section for more
    details.
1.  `damping`: This is a critical parameter and needs to be tuned for good
    performance. Check [hyper parameters][hyper_params] section for more
    details.

[hyper_params]:
https://github.com/tensorflow/kfac/tree/master/docs/examples/parameters.md

```python
  # Train with K-FAC.
  global_step = tf.train.get_or_create_global_step()
  optimizer = periodic_inv_cov_update_kfac_opt.PeriodicInvCovUpdateKfacOpt(
      invert_every=10,
      cov_update_every=1,
      learning_rate=0.0001,
      cov_ema_decay=0.95,
      damping=0.001,
      layer_collection=layer_collection,
      placement_strategy="round_robin",
      momentum=0.9)
  train_op = optimizer.minimize(loss, global_step=global_step)
```

## Fit the model

Optimizing with KFAC is similar to using a standard optimizer, where there is an
"update op" that computes and applies the update to the model's parameters.
However, KFAC introduces two additional sets of ops that must also be executed
as part of the algorithm (although not necessarily at every iteration). These
are called the "covariance update ops" and "inverse update ops", respectively.
The covariance update ops update the various "covariance" matrices used to
compute the Fisher block approximations for the layers. The inverse update ops
meanwhile are responsible for computing inverses of the approximate Fisher
blocks (using algorithms that exploit their special structure).

`PeriodicInvCovUpdateKfacOpt`, which is a subclass of `KfacOptimizer` class,
folds these extra ops into the standard update op, so that they execute
periodically on certain iterations, according to the `cov_update_every` and
`invert_every` arguments. Users seeking more fine-grained control of the timing
and placement of the ops can use the base `KfacOptimizer` class.

```python
  with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
      global_step_, loss_, accuracy_, _, _ = sess.run(
          [global_step, loss, accuracy, train_op])
```

## TIPS

1.  Check the [hyper params tuning][hp_tune] section for more details on tuning
    various KFAC parameters.

[hp_tune]: https://github.com/tensorflow/kfac/tree/master/docs/examples/parameters.md
[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
[preconditioner]: https://en.wikipedia.org/wiki/Preconditioner#Preconditioning_in_optimization
