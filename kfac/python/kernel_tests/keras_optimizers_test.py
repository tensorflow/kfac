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
"""Tests for keras/optimizers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.util import serialization
from kfac.python.keras import optimizers
from kfac.python.keras import utils

layers = tf.keras.layers
losses = tf.keras.losses
_SEED = 1234


# TODO(b/135916953): Use TensorFlow test_utils instead of below helpers.
def _get_synthetic_mnist_dataset(train_size=64, test_size=16):
  num_classes = 10
  img_rows, img_cols = 28, 28

  rng = np.random.RandomState(_SEED)
  num_examples = train_size + test_size
  images = rng.rand(num_examples, img_rows * img_cols).astype(np.float32)
  images = np.reshape(images, [num_examples, img_rows, img_cols, 1])
  labels = rng.randint(num_classes, size=num_examples)
  one_hot_labels = np.eye(num_classes)[labels].astype(np.float32)

  return ((images[:train_size], one_hot_labels[:train_size]),
          (images[train_size:], one_hot_labels[train_size:]))


def _get_synthetic_mnist_train_tensors(train_size=64, batch_size=10):
  (x_train, y_train), _ = _get_synthetic_mnist_dataset(train_size=train_size)
  dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  dataset = dataset.repeat().batch(batch_size)
  return dataset.make_one_shot_iterator().get_next()


def _generate_target_fn(num_examples):
  """Generated a random 2d target function for regression.

  Args:
    num_examples: The number of evenly spaced examples along the function to
      generate.

  Returns:
    A tuple of the x tensor and the y tensor for the generated function.
  """
  inds = np.arange(num_examples)
  x = np.sort(np.random.rand(num_examples) - 0.5)
  x = np.expand_dims(x, axis=1)
  y = np.transpose(x)
  dist = np.square(x - y)  # Should be scipy cdist(x, x, metric='sqeuclidean')
  k = np.exp(-dist / 0.01)
  k += np.eye(k.shape[0]) * 1e-6
  l = np.linalg.cholesky(k)
  random_y = np.random.randn(x.shape[0], 1)
  y = np.dot(l, random_y) + np.random.randn(x.shape[0], 1) * 1e-1
  return x[inds, :], y[inds, :]


def _generate_regression_data(num_eg, num_train_eg):
  x_all, y_all = _generate_target_fn(num_eg)
  x_all = x_all.astype(np.float32)
  y_all = y_all.astype(np.float32)

  inds = np.arange(num_eg)
  np.random.shuffle(inds)
  x_train = x_all[inds[:num_train_eg]]
  y_train = y_all[inds[:num_train_eg]]

  x_test = x_all[inds[num_train_eg:]]
  y_test = y_all[inds[num_train_eg:]]

  return (x_train, y_train), (x_test, y_test)


def _simple_mlp():
  return tf.keras.Sequential([
      layers.Dense(32, input_shape=(1,), activation='tanh'),
      layers.Dense(32, activation='tanh'),
      layers.Dense(1)
  ])


def _mnist_model(use_bias=True, use_separate_activation=True):
  """A complex architecture to test the variable registration.

  This model is not intended to be a "good" mnist classifier.
  It uses Lambda layers, concats, and separate branches to test effectively.

  Args:
    use_bias: boolean. Whether all the layers use a bias term or not.
    use_separate_activation: boolean. Whether the layers have the activation
      within the layer or use a separate activation layer.

  Returns:
    A Keras model containing the mnist classifier.
  """
  activation = 'linear' if use_separate_activation else 'relu'
  output_activation = 'linear' if use_separate_activation else 'softmax'

  inp = layers.Input(shape=(28, 28, 1))

  branch1 = layers.Lambda(lambda x: tf.squeeze(x, -1))(inp)
  branch1 = layers.Conv1D(3, kernel_size=7, activation=activation,
                          use_bias=use_bias)(branch1)
  if use_separate_activation:
    branch1 = layers.Activation('relu')(branch1)
  branch1 = layers.GlobalMaxPool1D()(branch1)

  branch2 = layers.Conv2D(16, kernel_size=(3, 3), activation=activation,
                          use_bias=use_bias)(inp)
  if use_separate_activation:
    branch2 = layers.Activation('relu')(branch2)
  branch2 = layers.MaxPooling2D(pool_size=(4, 4))(branch2)
  branch2 = layers.Flatten()(branch2)
  branch2 = layers.Dense(20, use_bias=use_bias)(branch2)
  if use_separate_activation:
    branch2 = layers.Activation('relu')(branch2)

  out = layers.concatenate([branch1, branch2])
  out = layers.Dense(10, use_bias=use_bias, activation=output_activation)(out)
  if use_separate_activation:
    out = layers.Activation('softmax')(out)

  return tf.keras.Model(inputs=inp, outputs=out)


def _train_model(data,
                 model,
                 loss,
                 lr=0.001,
                 damping=0.001,
                 batch_size=32,
                 epochs=1,
                 loss_weights=None):
  """Compiles and fits model to data and returns trainging results.

  Args:
    data: Tuple of numpy arrays shaped ((x_train, y_train), (x_test, y_test)).
    model: Uncompiled Keras model with inputs/output shapes matching the data.
    loss: tf.keras.losses loss function or serialized (string) loss function.
    lr: Learning rate for optimizer.
    damping: Damping parameter for KFAC.
    batch_size: Batch size used for training.
    epochs: Number of training epochs.
    loss_weights: List of weights or dict mapping layer names to loss function
      weight.

  Returns:
    A History object. Calling History.history gives you a dictionary with
    training and validation results.
  """
  (x_train, y_train), valid_data = data
  opt = optimizers.Kfac(learning_rate=lr, damping=damping, model=model,
                            loss=loss, loss_weights=loss_weights)
  model.compile(opt, loss, loss_weights=loss_weights)

  return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                   validation_data=valid_data, verbose=0)


class KfacOptimizerTest(parameterized.TestCase, tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(KfacOptimizerTest, self).__init__(*args, **kwargs)
    self._mnist_data = _get_synthetic_mnist_dataset()

  def setUp(self):
    super(KfacOptimizerTest, self).setUp()
    tf.random.set_random_seed(_SEED)
    np.random.seed(_SEED)

  def testFunctionalInstantiation(self):
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizers.Kfac(learning_rate=0.002, damping=0.04,
                        model=model, loss='binary_crossentropy')

  def testSequentialInstantiation(self):
    model = tf.keras.Sequential([
        layers.Conv2D(7, (3, 3), input_shape=(28, 28, 3)),
        layers.Activation('relu'),
        layers.Conv2D(13, (3, 3), activation='relu'),
        layers.GlobalMaxPool2D(),
        layers.Activation('softmax')
    ])
    optimizers.Kfac(learning_rate=0.03, damping=0.00007,
                        model=model, loss='binary_crossentropy')

  def testInstantiationWithLayerCollection(self):
    model = _simple_mlp()
    lc = utils.get_layer_collection(model, 'mse')
    opt = optimizers.Kfac(
        learning_rate=0.1, damping=0.2, layer_collection=lc)
    model.compile(optimizer=opt, loss='mse')
    opt.get_updates(model.total_loss, model.trainable_weights)

  def testRNNFails(self):
    model = tf.keras.Sequential()
    model.add(layers.Embedding(43, 128))
    model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = optimizers.Kfac(learning_rate=0.003, damping=0.003,
                              model=model, loss='binary_crossentropy')
    with self.assertRaisesRegex(ValueError,
                                '.*lstm.* has more than one parent tensor.$'):
      opt._create_optimizer()

  @parameterized.named_parameters(('BiasCombinedActivation', True, True),
                                  ('BiasSeparateActivation', True, False),
                                  ('NoBiasCombinedActivation', False, True),
                                  ('NoBiasSeparateActivation', False, False))
  def testBiasAndActivations(self, use_bias, use_separate_activation):
    model = _mnist_model(use_bias=use_bias,
                         use_separate_activation=use_separate_activation)
    _train_model(self._mnist_data, model, 'categorical_crossentropy')

  def testRegression(self):
    hist = _train_model(
        _generate_regression_data(200, 150), _simple_mlp(), 'mse', epochs=5)
    val_loss = hist.history['val_loss']
    self.assertGreater(val_loss[0], val_loss[-1])

  def testClipNormFails(self):
    with self.assertRaises(ValueError):
      optimizers.Kfac(learning_rate=0.001, damping=0.001,
                          model=_simple_mlp(), loss='mse', clipnorm=0.1)

  def testClipValueFails(self):
    with self.assertRaises(ValueError):
      optimizers.Kfac(learning_rate=0.01, damping=0.01,
                          model=_simple_mlp(), loss='mse', clipvalue=0.1)

  def testLossTensor(self):
    loss_tensor = tf.convert_to_tensor(2.0)
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                              model=_simple_mlp(), loss='mse',
                              loss_tensor=loss_tensor)
    self.assertEqual(opt.optimizer._loss_tensor, loss_tensor)

  def testArgsKwargs(self):
    """Test if kwargs are correctly forwarded to tensorflow_kfac."""
    kwargs = {
        'learning_rate': 3,
        'damping': 5,
        'momentum': 7,
        'min_damping': 9,
        'num_burnin_steps': 11,
        'invert_every': 13,
        'fisher_approx': {
            layers.Dense: 'kron_in_diag',
            'dense_1': 'kron_both_diag'
        },
    }
    model = _simple_mlp()
    opt = optimizers.Kfac(model=model, loss='mse', **kwargs)
    self.assertEqual(opt.optimizer._min_damping, kwargs['min_damping'])
    self.assertEqual(opt.optimizer._num_burnin_steps,
                     kwargs['num_burnin_steps'])
    self.assertEqual(opt.optimizer._invert_every, kwargs['invert_every'])

    fisher_block_0 = opt.optimizer.layers.fisher_blocks[model.layers[0].weights]
    self.assertTrue(fisher_block_0._diagonal_approx_for_input)
    self.assertFalse(fisher_block_0._diagonal_approx_for_output)
    fisher_block_1 = opt.optimizer.layers.fisher_blocks[model.layers[1].weights]
    self.assertTrue(fisher_block_1._diagonal_approx_for_input)
    self.assertTrue(fisher_block_1._diagonal_approx_for_output)

    with tf.Session() as sess:
      # In Keras, typically you do not use sessions directly. When you use a
      # Keras component, the required variables are initialized for you because
      # they are tracked. Here, we explicitly run the variables in a session so
      # they must be initialized.
      sess.run(tf.global_variables_initializer())
      self.assertEqual(sess.run(opt.optimizer.momentum), kwargs['momentum'])
      self.assertEqual(sess.run(opt.optimizer.learning_rate),
                       kwargs['learning_rate'])
      self.assertEqual(sess.run(opt.optimizer.damping), kwargs['damping'])

  def testConfig(self):
    fisher_approx = {layers.Dense: 'kron_in_diag', 'dense_1': 'kron_both_diag'}
    kwargs = {
        'loss': 'mse',
        'momentum': 7,
        'num_burnin_steps': 11,
        'min_damping': 9,
        'invert_every': 13,
        'fisher_approx': fisher_approx,
        'seed': 12,
    }
    opt = optimizers.Kfac(
        learning_rate=3, damping=5, model=_simple_mlp(), **kwargs)
    opt.learning_rate = 23
    opt.damping = 27
    config = opt.get_config()
    self.assertEqual(config['learning_rate'], 23)
    self.assertEqual(config['damping'], 27)
    dense_approx = fisher_approx.pop(layers.Dense)
    fisher_approx[utils._CLASS_NAME_PREFIX + 'Dense'] = dense_approx
    for key, val in kwargs.items():
      self.assertEqual(config[key], val)
      # Below is how Keras's model.save saves the configs. If the config is not
      # serializable, it will throw a TypeError or OverflowError.
    json.dumps(config, default=serialization.get_json_type).encode('utf8')

  @parameterized.named_parameters(('_LossName', {'loss': 'mse'}),
                                  ('_LossFunction', {'loss': losses.MSE}))
  def testFromConfig(self, kwargs_updates):
    kwargs = {
        'learning_rate': 3,
        'damping': 5,
        'momentum': 7,
        'min_damping': 9,
        'num_burnin_steps': 11,
        'invert_every': 13,
        'fisher_approx': {
            layers.Dense: 'kron_in_diag',
            'dense_1': 'kron_both_diag'
        },
    }
    kwargs.update(kwargs_updates)
    opt = optimizers.Kfac(model=_simple_mlp(), **kwargs)
    config = opt.get_config()
    config['name'] = 'diff_scope_name'
    opt2 = optimizers.Kfac.from_config(config)
    config2 = opt2.get_config()
    config2.pop('name')
    config.pop('name')
    self.assertEqual(config, config2)
    # Below is how Keras's model.save saves the configs. If the config is not
    # serializable, it will throw a TypeError or OverflowError.
    json.dumps(config, default=serialization.get_json_type).encode('utf8')
    json.dumps(config2, default=serialization.get_json_type).encode('utf8')

  @parameterized.named_parameters(('_Tensor', tf.convert_to_tensor),
                                  ('_Float', float))
  def testGettingHyper(self, hyper_ctor):
    kwarg_values = {'learning_rate': 3, 'damping': 20, 'momentum': 13,
                    'batch_size': 16}
    kwargs = {k: hyper_ctor(v) for k, v in kwarg_values.items()}
    opt = optimizers.Kfac(model=_simple_mlp(), loss='mse', **kwargs)
    get_value = backend.get_value
    tf_opt = opt.optimizer
    with self.subTest(name='MatchesFloat'):
      for name, val in kwarg_values.items():
        self.assertEqual(get_value(getattr(opt, name)), val)
    with self.subTest(name='MatchesTfOpt'):
      self.assertEqual(get_value(opt.lr), get_value(tf_opt.learning_rate))
      self.assertEqual(get_value(opt.damping), get_value(tf_opt.damping))
      self.assertEqual(get_value(opt.momentum), get_value(tf_opt.momentum))
      self.assertEqual(get_value(opt.batch_size), get_value(tf_opt._batch_size))

  def testGettingVariableHyperFails(self):
    self.skipTest('This is not fixed in TF 1.14 yet.')
    opt = optimizers.Kfac(model=_simple_mlp(),
                          loss='mse',
                          learning_rate=tf.Variable(0.1),
                          damping=tf.Variable(0.1))
    with self.assertRaisesRegex(tf.errors.FailedPreconditionError,
                                '.*uninitialized.*'):
      backend.get_value(opt.learning_rate)

  @parameterized.named_parameters(
      (('_' + name, name, val+1)
       for val, name in enumerate(optimizers._MUTABLE_HYPER_PARAMS)))
  def testSetTFVariableHyper(self, name, val):
    kwargs = {'learning_rate': 0.01, 'damping': 0.001}
    kwargs[name] = tf.Variable(45.0)
    opt = optimizers.Kfac(model=_simple_mlp(), loss='mse', **kwargs)
    setattr(opt, name, val)

    with self.subTest(name='AssignedCorrectly'):
      self.assertEqual(backend.get_value(getattr(opt, name)), val)
      if hasattr(opt.optimizer, name):
        self.assertEqual(backend.get_value(getattr(opt.optimizer, name)), val)

    with self.subTest(name='SetError'):
      with self.assertRaisesRegex(ValueError, 'Dynamic reassignment only.*'):
        setattr(opt, name, tf.convert_to_tensor(2))
      with self.assertRaisesRegex(ValueError, 'Dynamic reassignment only.*'):
        setattr(opt, name, tf.Variable(2))

  @parameterized.named_parameters(
      (('_' + name, name, val + 1)
       for val, name in enumerate(optimizers._MUTABLE_HYPER_PARAMS)))
  def testSetFloatHyper(self, name, val):
    kwargs = {'learning_rate': 0.01, 'damping': 0.001}
    kwargs[name] = 45.0
    opt = optimizers.Kfac(model=_simple_mlp(), loss='mse', **kwargs)
    setattr(opt, name, val)

    with self.subTest(name='AssignedCorrectly'):
      self.assertEqual(backend.get_value(getattr(opt, name)), val)
      if hasattr(opt.optimizer, name):
        self.assertEqual(backend.get_value(getattr(opt.optimizer, name)), val)

    with self.subTest(name='SetError'):
      with self.assertRaisesRegex(ValueError, 'Dynamic reassignment only.*'):
        setattr(opt, name, tf.convert_to_tensor(2))
      with self.assertRaisesRegex(ValueError, 'Dynamic reassignment only.*'):
        setattr(opt, name, tf.Variable(2))

  @parameterized.named_parameters(
      (('_' + name, name, val + 1)
       for val, name in enumerate(optimizers._MUTABLE_HYPER_PARAMS)))
  def testModifyingTensorHypersFails(self, name, val):
    kwargs = {'learning_rate': 3, 'damping': 5, 'momentum': 7, 'batch_size': 9}
    kwargs[name] = tf.convert_to_tensor(val)
    opt = optimizers.Kfac(model=_simple_mlp(), loss='mse', **kwargs)
    with self.subTest(name='AssignedCorrectly'):
      self.assertEqual(backend.get_value(getattr(opt, name)), val)
    with self.subTest(name='RaisesError'):
      with self.assertRaisesRegex(AttributeError,
                                  "Can't set attribute: {}".format(name)):
        setattr(opt, name, 17)

  def testLRBackwardsCompatibility(self):
    """This tests learning rate getting/setting used by old Keras callbacks."""
    opt = optimizers.Kfac(
        learning_rate=3, damping=5, model=_simple_mlp(), loss='mse')
    self.assertEqual(backend.get_value(opt.lr), 3)
    self.assertEqual(backend.get_value(opt.learning_rate), 3)
    opt.lr = 7
    self.assertEqual(backend.get_value(opt.lr), 7)
    self.assertEqual(backend.get_value(opt.learning_rate), 7)
    backend.set_value(opt.lr, 9)
    self.assertEqual(backend.get_value(opt.lr), 9)
    self.assertEqual(backend.get_value(opt.learning_rate), 9)
    backend.set_value(opt.learning_rate, 11)
    self.assertEqual(backend.get_value(opt.lr), 11)
    self.assertEqual(backend.get_value(opt.learning_rate), 11)

  def testMultipleLossTraining(self):
    inp = layers.Input(shape=(28, 28, 1))

    branch1 = layers.Conv2D(13, 7, activation='relu')(inp)
    branch1 = layers.GlobalMaxPool2D()(branch1)
    branch1 = layers.Dense(1, name='path1')(branch1)

    branch2 = layers.Conv2D(16, 3, activation='relu')(inp)
    branch2 = layers.MaxPooling2D(pool_size=(4, 4))(branch2)
    branch2 = layers.Flatten()(branch2)
    branch2 = layers.Dense(9, name='path2')(branch2)

    model = tf.keras.Model(inputs=inp, outputs=[branch1, branch2])
    loss = {'path1': 'binary_crossentropy', 'path2': 'categorical_crossentropy'}
    loss_weights = {'path1': 0.1, 'path2': 0.9}

    (x, y), (valid_x, valid_y) = _get_synthetic_mnist_dataset()
    y1, y2 = y[:, 0:1], y[:, 1:]
    valid_y1, valid_y2 = valid_y[:, 0:1], valid_y[:, 1:]
    data = (x, (y1, y2)), (valid_x, (valid_y1, valid_y2))

    _train_model(data, model, loss, loss_weights=loss_weights)

  @parameterized.named_parameters(('_LossName', 'categorical_crossentropy'),
                                  ('_LossFunction', losses.binary_crossentropy))
  def testRegisterLayersWithModel(self, loss):
    model = _mnist_model()
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.001)
    opt.register_layers(model=model, loss=loss)
    model.compile(optimizer=opt, loss=loss)
    opt.get_updates(model.total_loss, model.trainable_weights)

  def testRegisterLayersWithLayerCollection(self):
    model, loss = _mnist_model(), 'categorical_crossentropy'
    lc = utils.get_layer_collection(model, loss)
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.001)
    opt.register_layers(layer_collection=lc)
    model.compile(optimizer=opt, loss=loss)
    opt.get_updates(model.total_loss, model.trainable_weights)

  @parameterized.named_parameters(('_LossName', 'categorical_crossentropy'),
                                  ('_LossFunction', losses.binary_crossentropy))
  def testRegisterLayersCompiledModel(self, loss):
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.001)
    model = _mnist_model()
    model.compile(optimizer=opt, loss=loss)
    opt.register_layers(model=model)
    model.compile(optimizer=opt, loss=loss)
    opt.get_updates(model.total_loss, model.trainable_weights)

  def testTrainWithoutCreatingOptimizerFails(self):
    with self.assertRaisesRegex(ValueError, '.*provide a model with a loss.*'):
      opt = optimizers.Kfac(learning_rate=0.01, damping=0.001)
      model = _mnist_model()
      model.compile(optimizer=opt, loss='categorical_crossentropy')
      grads_vars = opt.get_gradients(model.total_loss, model.trainable_weights)
      opt.apply_gradients(grads_vars)

  def testEmptyCreateKfacOptimizerFails(self):
    with self.assertRaisesRegex(ValueError, '.*provide a model with a loss.*'):
      opt = optimizers.Kfac(learning_rate=0.01, damping=0.001)
      opt._create_optimizer()

  def testSeed(self):
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                              model=_simple_mlp(), loss='mse', seed=4321)
    lc = opt.optimizer.layers
    self.assertEqual(lc._loss_dict['squared_error_loss'][0]._default_seed, 4321)

  def testNewOptSameVarScope(self):
    model = _simple_mlp()
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=0.01, model=model, loss='mse')
    opt._create_optimizer()
    opt2 = optimizers.Kfac(
        learning_rate=0.02, damping=0.03, model=model, loss='mse')
    opt2._create_optimizer()

  def testGetSetWeights(self):
    def model_maker():
      return tf.keras.Sequential([layers.Dense(2, input_shape=(3,))])

    x = np.random.random((1, 3))
    y = np.random.random((1, 2))
    loss = 'mse'
    model = model_maker()
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.1,
                              model=model, loss=loss, seed=1234)
    model.compile(optimizer=opt, loss=loss)
    model.train_on_batch(x, y)
    opt_weights = opt.get_weights()

    self.assertEqual(1, opt_weights[0])  # iterations
    self.assertEqual(1, opt_weights[6])  # counter
    self.assertEqual(0, opt_weights[7])  # burn in counter

    config = opt.get_config()
    config['name'] = 'diff_name'
    opt2 = optimizers.Kfac.from_config(config)
    model2 = model_maker()
    model2.compile(optimizer=opt2, loss=loss)
    opt2.register_layers(model=model2)
    # Set weights should only work after a call to get_updates/apply_gradients.
    x = np.random.random((1, 3))
    y = np.random.random((1, 2))
    model2.train_on_batch(x, y)
    opt2.set_weights(opt_weights)

    for w1, w2 in zip(opt_weights, opt2.get_weights()):
      self.assertAllClose(w1, w2)

    model2.set_weights(model.get_weights())
    x = np.random.random((1, 3))
    y = np.random.random((1, 2))
    model.train_on_batch(x, y)
    model2.train_on_batch(x, y)

    for w1, w2 in zip(opt.get_weights(), opt2.get_weights()):
      self.assertAllClose(w1, w2)

  @parameterized.named_parameters(('_HasShift', True), ('_NoShift', False))
  def testTrainModelWithNormalization(self, has_shift):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 1)),
        layers.BatchNormalization(center=has_shift, fused=False),
        layers.Conv2D(23, 3),
        layers.LayerNormalization(center=has_shift),
        layers.GlobalMaxPool2D(),
        layers.Dense(10, activation='softmax')
    ])
    (x_train, y_train), _ = _get_synthetic_mnist_dataset()
    approx = {layers.LayerNormalization: 'full'}
    loss = 'categorical_crossentropy'
    opt = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                              model=model, loss=loss, fisher_approx=approx)
    model.compile(opt, loss)
    return model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

  @parameterized.named_parameters(('_HasShift', True), ('_NoShift', False))
  def testTrainModelWithFusedBN(self, has_shift):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 1)),
        layers.BatchNormalization(center=has_shift, fused=True),
        layers.GlobalMaxPool2D(),
        layers.Dense(10, activation='softmax')
    ])
    (x_train, y_train), _ = _get_synthetic_mnist_dataset()
    loss = 'categorical_crossentropy'
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=0.01, model=model, loss=loss)
    model.compile(opt, loss)
    return model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

  @parameterized.named_parameters(('_HasShift', True), ('_NoShift', False))
  def testTrainModelWithFusedBNAndLearningPhase(self, has_shift):
    tf.keras.backend.set_learning_phase(1)
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 1)),
        layers.BatchNormalization(center=has_shift, fused=True),
        layers.GlobalMaxPool2D(),
        layers.Dense(10, activation='softmax')
    ])
    (x_train, y_train), _ = _get_synthetic_mnist_dataset()
    loss = 'categorical_crossentropy'
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=0.01, model=model, loss=loss)
    model.compile(opt, loss)
    return model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

  @parameterized.named_parameters(('_WithShape', {'input_shape': (28, 28, 1)}),
                                  ('_WithoutShape', {}))
  def testCustomTrainingLoopSequential(self, input_conv_kwargs):
    # Without the input_shape the only inbound node is the correct one, with the
    # input_shape there are two, and we want the second one.
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, **input_conv_kwargs),
        layers.BatchNormalization(fused=False),
        layers.Conv2D(23, 3),
        layers.LayerNormalization(),
        layers.GlobalMaxPool2D(),
        layers.Dense(10, activation='softmax', name='output_test')
    ])
    x, y = _get_synthetic_mnist_train_tensors(batch_size=10)
    model_input = tf.keras.Input(tensor=x)
    output = model(model_input)
    loss = tf.keras.losses.binary_crossentropy(output, y)
    optimizer = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                                    model=model, loss='binary_crossentropy')
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(3):
        sess.run([train_op])

  def testCustomTrainingLoopFunctionalInpTensor(self):
    # This case should work trivially--the only inbound node is the correct one.
    x, y = _get_synthetic_mnist_train_tensors(batch_size=10)

    # Build Model
    inp = tf.keras.Input(tensor=x)
    x = layers.Conv2D(13, 5)(inp)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.Conv2D(23, 3)(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalMaxPool2D()(x)
    out = layers.Dense(10, activation='softmax', name='output_test')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    loss = tf.keras.losses.binary_crossentropy(model.output, y)
    optimizer = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                                    model=model, loss='binary_crossentropy')
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(3):
        sess.run([train_op])

  def testCustomTrainingLoopFunctionalInpShape(self):
    # We need to ensure correct inbound node is used for layer collection.
    x, y = _get_synthetic_mnist_train_tensors(batch_size=10)
    model_input = tf.keras.Input(tensor=x)

    # Build Model
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(13, 5)(inp)
    x = layers.BatchNormalization(fused=True)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(23, 3)(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalMaxPool2D()(x)
    out = layers.Dense(10, activation='softmax', name='output_test')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    output = model(model_input)
    loss = tf.keras.losses.binary_crossentropy(output, y)
    optimizer = optimizers.Kfac(damping=0.01, learning_rate=0.01,
                                    model=model, loss='binary_crossentropy')
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(3):
        sess.run([train_op])

  def testCustomTrainingLoopMakeOptimizerBeforeModelCall(self):
    # We defer the creation of the layer_collection to the minimize call for
    # this situation, because if we make the layer_collection immediately it
    # will capture the wrong inbound node.
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5),
        layers.BatchNormalization(fused=False),
        layers.Conv2D(23, 3),
        layers.LayerNormalization(),
        layers.GlobalMaxPool2D(),
        layers.Dense(10, activation='softmax', name='output_test')
    ])
    optimizer = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                                    model=model, loss='binary_crossentropy')
    x, y = _get_synthetic_mnist_train_tensors(batch_size=10)
    model_input = tf.keras.Input(tensor=x)
    output = model(model_input)
    loss = tf.keras.losses.binary_crossentropy(output, y)
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      for _ in range(3):
        sess.run([train_op])

  def testCustomTrainingUnwrappedTensorFails(self):
    # This test does not test our implementation, but is here so if Keras ever
    # adds functionality to support raw tensors as Nodes, this test will fail
    # and we can remove the restriction from our documentation.
    model = _simple_mlp()
    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat().batch(10)
    x, y = dataset.make_one_shot_iterator().get_next()
    pred = model(x)
    loss = tf.keras.losses.binary_crossentropy(pred, y)
    optimizer = optimizers.Kfac(learning_rate=0.01, damping=0.01,
                                    model=model, loss='binary_crossentropy')
    train_op = optimizer.minimize(loss, var_list=model.trainable_weights)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                  '.*You must feed a value for placeholder.*'):
        sess.run([train_op])

  def testTrainingNestedModel(self):
    inputs = tf.keras.Input(shape=(1,))
    y1 = _simple_mlp()(inputs)
    y2 = _simple_mlp()(inputs)
    y3 = _simple_mlp()(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = optimizers.Kfac(learning_rate=0.01,
                                    damping=0.01,
                                    model=ensemble_model,
                                    loss='binary_crossentropy')
    ensemble_model.compile(optimizer, 'binary_crossentropy')

    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat().batch(10)
    x, y = dataset.make_one_shot_iterator().get_next()
    ensemble_model.train_on_batch(x, y)

  def testCustomTrainLoopNestedModel(self):
    inputs = tf.keras.Input(shape=(1,))
    y1 = _simple_mlp()(inputs)
    y2 = _simple_mlp()(inputs)
    y3 = _simple_mlp()(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat().batch(10)
    x, y = dataset.make_one_shot_iterator().get_next()
    x = layers.Input(tensor=x)

    optimizer = optimizers.Kfac(learning_rate=0.01,
                                    damping=0.01,
                                    model=ensemble_model,
                                    loss='binary_crossentropy')

    pred = ensemble_model(x)
    loss = tf.keras.losses.binary_crossentropy(pred, y)
    train_op = optimizer.minimize(
        loss, var_list=ensemble_model.trainable_weights)
    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run([train_op])

  @parameterized.named_parameters(
      ('_NoKwargs', {'norm_constraint', 'batch_size'}, {}),
      ('_MomentumNormKwargs',
       set(),
       {'momentum': 1, 'norm_constraint': 2, 'batch_size': 16}),
      ('_QModel',
       {'momentum', 'learning_rate', 'norm_constraint', 'batch_size'},
       {'momentum': None, 'momentum_type': 'qmodel', 'learning_rate': None}),
      ('_AdaptiveDamping',
       {'damping', 'norm_constraint', 'batch_size'},
       {'adapt_damping': True, 'damping_adaptation_interval': 20}))
  def testMutableHypers(self, not_mutable, kwargs_update):
    kwargs = {'learning_rate': 0.01, 'damping': 0.001}
    kwargs.update(kwargs_update)
    opt = optimizers.Kfac(model=_simple_mlp(), loss='mse', **kwargs)
    mutable = optimizers._MUTABLE_HYPER_PARAMS - not_mutable
    self.assertEqual(set(opt.mutable_hyperparameters), mutable)

  def testPositionalArgsFail(self):
    with self.assertRaisesRegex(ValueError,
                                'Do not pass positional arguments.*'):
      optimizers.Kfac(0.1, 0.1, model=_simple_mlp(), loss='mse')

  def testSettingName(self):
    model = _simple_mlp()
    optimizer = optimizers.Kfac(damping=0.01, learning_rate=0.01,
                                    model=model, loss='mse')
    optimizer.name = 'new_name'
    self.assertEqual(optimizer._name, 'new_name')
    self.assertEqual(optimizer.get_config()['name'], 'new_name')
    self.assertEqual(optimizer._kfac_kwargs['name'], 'new_name')
    model.compile(optimizer, 'mse')
    model._make_train_function()
    with self.assertRaisesRegex(ValueError,
                                '.*after the variables are created.*'):
      optimizer.name = 'another_name'


if __name__ == '__main__':
  tf.test.main()
