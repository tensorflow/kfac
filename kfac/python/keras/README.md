# K-FAC for Keras

**K-FAC for Keras** is an implementation of K-FAC, an approximate second-order
optimization method, in TensorFlow. When applied to feedforward and
convolutional neural networks, K-FAC can converge much faster (`>3.5x`) and with
fewer iterations (`>14x`) than SGD with Momentum. You can read more about it in
the paper [here][paper] and the GitHub docs [here][index].

[index]: https://github.com/tensorflow/kfac/tree/master/docs/index.md
[paper]: https://arxiv.org/abs/1503.05671

## Why should I use K-FAC for Keras?

In addition to the reasons outlined on the GitHub docs, the Keras version
handles layer and loss registration automatically and works with Keras's
convenient training API. See the reference code [here][cifar10].

[cifar10]: https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/KFAC_vs_Adam_on_CIFAR10.ipynb
[cifar10tpu]: https://github.com/tensorflow/kfac/tree/master/kfac/examples/keras/KFAC_vs_Adam_on_CIFAR10_TPU.ipynb

## How do I use K-FAC for Keras?

Using this optimizer is almost the same as using any other Keras optimizer,
except you must also pass the loss and model to the optimizer. The optimizer
will automatically register the model layers and loss so K-FAC can compute the
fisher approximations.

```python
import tensorflow as tf
import kfac

# Build Keras Model (can use functional or sequential)
model = tf.keras.Model(...)
loss = 'sparse_categorical_crossentropy' # or a tf.keras.losses.* instance

# Construct Optimizer
optimizer = kfac.keras.optimizers.Kfac(learning_rate=0.001,
                                       damping=0.01,
                                       model=model,
                                       loss=loss)

# Compile and Fit Model
model.compile(optimizer=optimizer, loss=loss, ...)
model.fit(...)
```

Check out our CIFAR-10 CNN training [example][cifar10] and
[TPU Strategy example][cifar10tpu] for more details.

This optimizer currently supports the following tf.keras.layers types: Conv2D,
Conv1D, Dense, BatchNormalization, LayerNormalization and Embedding. The
following tf.keras.losses are supported: sparse_categorical_crossentropy,
categorical_crossentropy, binary_crossentropy, and mean_squared_error. You may
use any architecture with these basic layers and losses, including multiple
branches and loss functions.

To use an unsupported layer or loss, you can register layers manually using
a LayerCollection object and pass that to the optimizer constructor. Examples
of using LayerCollection are [here][layercollection].

[layercollection]: https://github.com/tensorflow/kfac/tree/master/kfac/examples

## How is K-FAC Different from Other Keras Optimizers?

1.  When using your model as a callable (i.e. `output = model(input)`), `input`
    must be a Keras layer. If it is a normal tensor, you can wrap it as follows:
    `new_input = tf.keras.layers.Input(tensor=input)`. This is so Keras
    registers the layer as an inbound_node during the call, allowing our layer
    collection to register it correctly. By default, our automatic layer
    collection will register only the latest use of the model.
2.  Only a subset of the hyperparameters can be accessed and modified after
    instantiation. These are: learning_rate, damping, momentum,
    weight_decay_coeff, norm_constraint, and batch_size. These hyperparameters
    will work the same as normal hyperparameters in native Keras optimizers and
    can be used with tools like hyperparameter scheduler callbacks. You can see
    exactly which hyperparameters are modifiable by checking the
    `optimizer.mutable_hyperparameters` property. Note that damping cannot be
    modified when using adaptive damping, and momentum/learning_rate cannot be
    modified when using qmodel momentum. Also, if any of the hyperparameters are
    `None` during instantiation, they will not be modifiable during training.
3.  This optimizer is fully compatible with tf.keras.models.save_model or
    model.save(). To load the compiled model with the optimizer, you must use
    our saving_utils.load_model method, which is identical to
    tf.keras.models.load_model except it registers the model with the optimizer
    after compiling the model and before loading the optimizer's weights.
    Example:

    ```python
    import tensorflow as tf
    import kfac

    model = tf.keras.Model(...)
    loss = tf.keras.losses.MSE()  # could be a serialized loss function
    optimizer = kfac.keras.optimizers.Kfac(learning_rate=0.001,
                                           damping=0.01,
                                           model=model,
                                           loss=loss)
    model.compile(optimizer, loss)
    model.fit(...)
    model.save('saved_model.hdf5')  # or tf.keras.models.save_model(model)
    ...
    loaded_model = kfac.keras.saving_utils.load_model('saved_model.hdf5')
    loaded_model.fit(...)
    ```
