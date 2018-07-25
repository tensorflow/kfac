# Home

Kronecker factored approximate curvature

**K-FAC in TensorFlow** is an implementation of K-FAC, an approximate
second-order optimization method, in TensorFlow. When applied to feedforward and
convolutional neural networks, K-FAC can converge much faster (`>3.5x`) and with
fewer iterations (`>14x`) than SGD with Momentum.

[TOC]

## What is K-FAC?

K-FAC, short for "Kronecker-factored Approximate Curvature", is an approximation
to the [Natural Gradient][natural_gradient] algorithm designed specifically for
neural networks. It maintains an approximation to the [Fisher Information
matrix][fisher_information], whose inverse is used as a preconditioner for
(stochastic) gradient descent.

K-FAC can be used in place of SGD, Adam, and other `Optimizer` implementations.
However it is slightly more restrictive compared to SGD, Adam as it makes some
assumptions on the structure of the model and the loss function.

Unlike most optimizers, K-FAC exploits structure in the model itself (e.g. "What
are the weights for layer i?"). As such, you must add some additional code while
constructing your model to use K-FAC.

[natural_gradient]: http://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746
[fisher_information]: https://en.wikipedia.org/wiki/Fisher_information#Matrix_form

## Why should I use K-FAC?

K-FAC can take advantage of the curvature of the optimization problem, resulting
in **faster training**. For an 8-layer Autoencoder, K-FAC converges to the same
loss as SGD with Momentum in 3.8x fewer seconds and 14.7x fewer updates. See
reference code [here][autoencoder-code] and plots comparing KFAC with SGD below.

![](https://github.com/tensorflow/kfac/tree/master/docs/autoencoder.png?raw=True)

[autoencoder-code]: https://github.com/tensorflow/kfac/tree/master/kfac/examples/kfac_mnist_autoencoder_auto_damping.py

## How do I use K-FAC?

Using K-FAC requires three steps,

1.  Registering layer inputs, weights, and pre-activations with a
    `kfac.LayerCollection`.
2.  Register loss functions.
3.  Minimizing the loss with a `PeriodicInvCovUpdateOptimizer`.

```python
import kfac
# Build model.
w = tf.get_variable("w", ...)
b = tf.get_variable("b", ...)
logits = tf.matmul(x, w) + b
loss = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Register loss.
layer_collection = kfac.LayerCollection()
layer_collection.register_categorical_predictive_distribution(logits)

# Register layers.
layer_collection.auto_register_layers()

# Construct training ops.
optimizer = kfac.PeriodicInvCovUpdateOptimizer(..., layer_collection=layer_collection)
train_op = optimizer.minimize(loss)

# Minimize loss.
with tf.Session() as sess:
  ...
  sess.run([train_op])
```

Check out the Convnet training [example][convexamplesec] for more details. Also
check [`PeriodicInvCovUpdate`][periodicincovupdate] optimizer to see how the
covariance and invariance ops placement and execution can be handled
automatically.

[convexamplesec]: https://github.com/tensorflow/kfac/tree/master/docs/examples/convolutional.md
[periodicincovupdate]: https://github.com/tensorflow/kfac/tree/master/kfac/python/ops/kfac_utils/periodic_inv_cov_update_kfac_opt.py
