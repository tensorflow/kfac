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
"""Tests for keras/utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from kfac.python.keras import utils
from kfac.python.ops import fisher_blocks
from kfac.python.ops import loss_functions

layers = tf.keras.layers
losses = tf.keras.losses
_SEED = 1234


def _mlp():
  return tf.keras.Sequential([
      layers.Embedding(100, 13, input_length=1),
      layers.Flatten(),
      layers.Dense(32, activation='tanh'),
      layers.Dense(32, activation='tanh'),
      layers.Dense(1)
  ])


def _cnn():
  return tf.keras.Sequential([
      layers.Conv2D(7, 5, input_shape=(28, 28, 3)),
      layers.Activation('relu'),
      layers.Conv2D(13, (3, 3), activation='relu'),
      layers.GlobalMaxPool2D(),
      layers.Activation('softmax')
  ])


def _two_loss_model(num_branch1_outputs=1, num_branch2_outputs=9):
  inp = layers.Input(shape=(28, 28, 1))

  branch1 = layers.Lambda(lambda x: tf.squeeze(x, -1))(inp)
  branch1 = layers.Conv1D(13, 7, activation='relu')(branch1)
  branch1 = layers.GlobalMaxPool1D()(branch1)
  branch1 = layers.Dense(num_branch1_outputs, name='out1')(branch1)

  branch2 = layers.Conv2D(16, 3, activation='relu')(inp)
  branch2 = layers.MaxPooling2D(pool_size=(4, 4))(branch2)
  branch2 = layers.Flatten()(branch2)
  branch2 = layers.Dense(num_branch2_outputs, name='out2')(branch2)

  return inp, (branch1, branch2)


class GetLayerCollectionTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(GetLayerCollectionTest, self).setUp()
    tf.reset_default_graph()
    tf.random.set_random_seed(_SEED)

  @parameterized.named_parameters(
      ('_Categorical', 'categorical_crossentropy',
       loss_functions.CategoricalLogitsNegativeLogProbLoss),
      ('_Binary', 'binary_crossentropy',
       loss_functions.MultiBernoulliNegativeLogProbLoss),
      ('_Sparse', losses.sparse_categorical_crossentropy,
       loss_functions.CategoricalLogitsNegativeLogProbLoss))
  def testValidLogitLossFunctionsCNN(self, loss, kfac_loss):
    """Ensures correct tensorflow_kfac loss function and variable for a CNN.

    Args:
      loss: A losses function (in serialized form or actual reference)
      kfac_loss: tensorflow_kfac.python.ops loss function.
    """
    with tf.Graph().as_default():
      model = _cnn()
      lc = utils.get_layer_collection(model, loss)
      self.assertIsInstance(lc.losses[0], kfac_loss)
      self.assertEqual(lc.losses[0].params,
                       utils.get_parent(model.layers[-1].output))

  @parameterized.named_parameters(
      ('_Categorical', 'categorical_crossentropy',
       loss_functions.CategoricalLogitsNegativeLogProbLoss),
      ('_Binary', 'binary_crossentropy',
       loss_functions.MultiBernoulliNegativeLogProbLoss),
      ('_Sparse', losses.sparse_categorical_crossentropy,
       loss_functions.CategoricalLogitsNegativeLogProbLoss))
  def testValidLogitLossFunctionsMLP(self, loss, kfac_loss):
    """Ensures correct tensorflow_kfac loss function and variable for a MLP.

    Args:
      loss: A losses function (in serialized form or actual reference)
      kfac_loss: tensorflow_kfac.python.ops loss function.
    """
    with tf.Graph().as_default():
      model = _mlp()
      lc = utils.get_layer_collection(model, loss)
      self.assertIsInstance(lc.losses[0], kfac_loss)
      self.assertEqual(lc.losses[0].params, model.layers[-1].output)

  @parameterized.named_parameters(('_LongCNN', 'mean_squared_error', _cnn),
                                  ('ShortCNN', 'mse', _cnn),
                                  ('_LongMLP', losses.mean_squared_error, _mlp),
                                  ('ShortMLP', 'mse', _mlp),
                                  ('_Class', losses.MeanSquaredError(), _mlp))
  def testValidMSE(self, loss, model_builder):
    """Ensures variations of MSE and output variables work.

    Args:
      loss: A tf.keras.losses function (in serialized form or actual reference)
      model_builder: Function that returns a Keras model.
    """
    model = model_builder()
    lc = utils.get_layer_collection(model, loss)
    self.assertIsInstance(lc.losses[0],
                          loss_functions.NormalMeanNegativeLogProbLoss)
    self.assertEqual(lc.losses[0].params, model.layers[-1].output)

  @parameterized.named_parameters(('_NotRealLoss', 'blah blah blah'),
                                  ('_RealButInvalid', 'cosine'),
                                  ('_SimilarName', 'msle'))
  def testInvalidLossFunctions(self, loss):
    with self.assertRaisesRegex(ValueError, '.*loss function:.*'):
      model = _mlp()
      utils.get_layer_collection(model, loss)

  @parameterized.named_parameters(('_CNN', _cnn), ('_MLP', _mlp))
  def testLayerRegistration(self, model_builder):
    model = model_builder()
    model.layers[0].trainable = False

    lc = utils.get_layer_collection(model, 'mse')
    registered = set(lc.registered_variables)

    variables = set()
    for layer in model.layers[1:]:
      if layer.trainable and layer.count_params():
        variables |= set(layer.weights)

    self.assertEqual(registered, variables)

  @parameterized.named_parameters(
      ('_DictLoss',
       {'out1': 'binary_crossentropy', 'out2': 'categorical_crossentropy'},
       {'out1': 0.1, 'out2': 0.9}),
      ('_ListLoss',
       ['binary_crossentropy', 'categorical_crossentropy'],
       [0.1, 0.9]))
  def testMultipleLoss(self, loss, loss_weights):
    inputs, (out1, out2) = _two_loss_model()
    model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])
    lc = utils.get_layer_collection(model, loss, loss_weights=loss_weights)

    self.assertLen(lc.loss_coeffs.keys(), 2)
    self.assertLen(lc.loss_colocation_ops.keys(), 2)

    l1 = lc._loss_dict['sigmoid_cross_entropy_loss']
    l2 = lc._loss_dict['sparse_softmax_cross_entropy_loss']

    self.assertLen(l1, 1)
    self.assertLen(l2, 1)

    l1, l2 = l1[0], l2[0]

    self.assertIsInstance(l1,
                          loss_functions.MultiBernoulliNegativeLogProbLoss)
    self.assertIsInstance(l2,
                          loss_functions.CategoricalLogitsNegativeLogProbLoss)
    self.assertEqual(lc.loss_coeffs[l1], 0.1)
    self.assertEqual(lc.loss_coeffs[l2], 0.9)
    self.assertEqual(lc.loss_colocation_ops[l1], out1)
    self.assertEqual(lc.loss_colocation_ops[l2], out2)

    self.assertEqual(lc.loss_coeffs[l1], 0.1)
    self.assertEqual(lc.loss_coeffs[l2], 0.9)

  @parameterized.named_parameters(('_EmptyDict', {}),
                                  ('_PartialDict', {'out2': 0.3}))
  def testMultipleLossWeights(self, loss_weights):
    inputs, (out1, out2) = _two_loss_model()
    model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])
    loss = ['binary_crossentropy', 'categorical_crossentropy']
    lc = utils.get_layer_collection(model, loss, loss_weights=loss_weights)

    l1 = lc._loss_dict['sigmoid_cross_entropy_loss'][0]
    self.assertEqual(lc.loss_coeffs[l1], 1.0)

  @parameterized.named_parameters(
      ('_MissingDict', {'out2': 'categorical_crossentropy'}),
      ('_MissingList', ['categorical_crossentropy']),
      ('_ExtraDict', {'out1': 'binary_crossentropy',
                      'out2': 'categorical_crossentropy',
                      'blah': 'mse'}),
      ('_ExtraList', ['mse', 'binary_crossentropy',
                      'categorical_crossentropy']),
      ('_WrongName', {'out1': 'binary_crossentropy',
                      'path2': 'categorical_crossentropy'}))
  def testLossErrors(self, loss):
    with self.assertRaisesRegex(ValueError, '.*loss dict.*'):
      inputs, (out1, out2) = _two_loss_model()
      model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])
      utils.get_layer_collection(model, loss)

  @parameterized.named_parameters(
      ('_EmptyList', []),
      ('_MissingList', [0.1]),
      ('_ExtraList', [0.1, 0.9, 0.3]),
      ('_ExtraDict', {'out1': 0.1, 'out2': 0.9, 'blahblah': 0.4}),
      ('_Set', {0.1, 0.3}))
  def testLossWeightErrors(self, loss_weights):
    with self.assertRaisesRegex(ValueError, '.*loss_weights.*'):
      inputs, (out1, out2) = _two_loss_model()
      model = tf.keras.Model(inputs=inputs, outputs=[out1, out2])
      loss = ['binary_crossentropy', 'categorical_crossentropy']
      utils.get_layer_collection(model, loss, loss_weights=loss_weights)

  @parameterized.named_parameters(
      ('_Seperable', layers.SeparableConv2D(13, 5)),
      ('_ChannelsFirst', layers.Conv2D(11, 3, data_format='channels_first')))
  def testInvalidCNNLayers(self, layer):
    with self.assertRaisesRegex(ValueError, '.*convolutional layer.*'):
      model = tf.keras.Sequential([layers.Input(shape=(28, 28, 3)), layer])
      utils.get_layer_collection(model, 'mse')

  @parameterized.named_parameters(
      ('_List', ['kron', 'kron_in_diag', 'kron_out_diag', 'kron_both_diag']),
      ('_Dict', {'l1': 'kron', 'l2': 'kron_in_diag', 'l3': 'kron_out_diag',
                 'l4': 'kron_both_diag'}),
      ('_DictOneMissing', {'l2': 'kron_in_diag', 'l3': 'kron_out_diag',
                           'l4': 'kron_both_diag'}))
  def testFisherApproxLayerNames(self, fisher_approx):
    model = tf.keras.Sequential([
        layers.Dense(10, input_shape=(20,), name='l1'),
        layers.Activation('relu'),
        layers.Dense(13, activation='relu', name='l2'),
        layers.Dense(23, trainable=False),
        layers.Dense(17, name='l3'),
        layers.Activation('relu'),
        layers.Dense(3, name='l4')])
    lc = utils.get_layer_collection(model, 'mse', fisher_approx=fisher_approx)
    trainable_layers = [model.layers[i] for i in [0, 2, 4, 6]]
    expected_in_diag_approx = [False, True, False, True]
    expected_out_diag_approx = [False, False, True, True]

    for layer, in_diag, out_diag in zip(trainable_layers,
                                        expected_in_diag_approx,
                                        expected_out_diag_approx):
      self.assertEqual(
          in_diag, lc.fisher_blocks[layer.weights]._diagonal_approx_for_input)
      self.assertEqual(
          out_diag, lc.fisher_blocks[layer.weights]._diagonal_approx_for_output)

  @parameterized.named_parameters(
      ('_ClassOnly', {layers.Conv2D: 'diagonal'},
       (fisher_blocks.ConvDiagonalFB, fisher_blocks.ConvDiagonalFB)),
      ('_NameAndClass', {layers.Conv2D: 'diagonal', 'conv2d_1': None},
       (fisher_blocks.ConvDiagonalFB, fisher_blocks.ConvKFCBasicFB)))
  def testFisherApproxLayerClass(self, fisher_approx, block_types):
    model = _cnn()
    lc = utils.get_layer_collection(model, 'mse',
                                    fisher_approx=fisher_approx)
    trainable_layers = [model.layers[0], model.layers[2]]
    for layer, block_type in zip(trainable_layers, block_types):
      self.assertIsInstance(lc.fisher_blocks[layer.weights], block_type)

  @parameterized.named_parameters(
      ('_EmptyList', []),
      ('_ExtraDict', {'conv2d': 'diagonal', layers.Conv2D: 'kron',
                      'UWaterloo': 'kron'}),
      ('_ExtraList', ['kron', 'diagonal', 'diagonal']),
      ('_WrongName', {'conv2d': 'kron', 'path2': 'kron'}))
  def testFisherApproxErrors(self, fisher_approx):
    with self.assertRaisesRegex(ValueError, '.*fisher_approx.*'):
      utils.get_layer_collection(_cnn(), 'mse', fisher_approx=fisher_approx)

  @parameterized.named_parameters(
      ('_List', ['full', 'diagonal'], ['full', 'diagonal']),
      ('_SerializedDict',
       {'dense1': 'full', 'dense2': 'diagonal'},
       {'dense1': 'full', 'dense2': 'diagonal'}),
      ('_PartiallySerializedDict',
       {layers.Dense: 'full', utils._CLASS_NAME_PREFIX + 'Conv2D': 'full'},
       {utils._CLASS_NAME_PREFIX + 'Dense': 'full',
        utils._CLASS_NAME_PREFIX + 'Conv2D': 'full'}),
      ('_Dict',
       {layers.Dense: 'diagonal', layers.Conv2D: 'full'},
       {utils._CLASS_NAME_PREFIX + 'Dense': 'diagonal',
        utils._CLASS_NAME_PREFIX + 'Conv2D': 'full'}))
  def testSerializeFisherApprox(self, approx, correctly_serialized_approx):
    serialized_approx = utils.serialize_fisher_approx(approx)
    self.assertEqual(serialized_approx, correctly_serialized_approx)

  def testSeed(self):
    lc = utils.get_layer_collection(model=_mlp(), loss='mse', seed=4321)
    self.assertEqual(lc._loss_dict['squared_error_loss'][0]._default_seed, 4321)

  @parameterized.named_parameters(('_HasShift', True), ('_NoShift', False))
  def testNormalizationLayers(self, has_shift):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 3)),
        layers.BatchNormalization(center=has_shift, name='bn'),
        layers.Conv2D(23, 3),
        layers.LayerNormalization(center=has_shift),
        layers.GlobalMaxPool2D(),
    ])
    fisher_approx = {layers.LayerNormalization: 'full', 'bn': 'diagonal'}
    lc = utils.get_layer_collection(model, 'mse', fisher_approx=fisher_approx)
    bn_weights = model.layers[1].trainable_weights
    ln_weights = model.layers[3].trainable_weights
    if not has_shift:
      bn_weights, ln_weights = bn_weights[0], ln_weights[0]
    bn_block = lc.fisher_blocks[bn_weights]
    ln_block = lc.fisher_blocks[ln_weights]
    self.assertIsInstance(bn_block, fisher_blocks.ScaleAndShiftDiagonalFB)
    self.assertIsInstance(ln_block, fisher_blocks.ScaleAndShiftFullFB)
    self.assertEqual(bn_block._has_shift, has_shift)
    self.assertEqual(ln_block._has_shift, has_shift)

  def testErrorWithBatchNormNoScale(self):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 3)),
        layers.BatchNormalization(scale=False, fused=False),
        layers.GlobalMaxPool2D(),
    ])
    with self.assertRaisesRegex(ValueError, '.*scale=False.*'):
      utils.get_layer_collection(model, 'binary_crossentropy')

  def testErrorWithLayerNormNoScale(self):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 3)),
        layers.LayerNormalization(scale=False),
        layers.GlobalMaxPool2D(),
    ])
    with self.assertRaisesRegex(ValueError, '.*scale=False.*'):
      utils.get_layer_collection(model, 'binary_crossentropy')

  def testNumBatchNormUsesWithPhase(self):
    tf.keras.backend.set_learning_phase(1)
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 3)),
        layers.BatchNormalization(fused=True),
        layers.GlobalMaxPool2D(),
    ])
    lc = utils.get_layer_collection(model, 'binary_crossentropy')
    for w in model.layers[1].trainable_weights:
      self.assertEqual(lc._vars_to_uses[w], 1)

  def testNumBatchNormUsesNoPhase(self):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5, input_shape=(28, 28, 3)),
        layers.BatchNormalization(fused=True),
        layers.GlobalMaxPool2D(),
    ])
    lc = utils.get_layer_collection(model, 'binary_crossentropy')
    for w in model.layers[1].trainable_weights:
      self.assertEqual(lc._vars_to_uses[w], 2)

  def testModelAsCallable(self):
    model = tf.keras.Sequential([
        layers.Conv2D(13, 5),
        layers.BatchNormalization(name='bn', fused=False),
        layers.Conv2D(23, 3),
        layers.LayerNormalization(),
        layers.GlobalMaxPool2D(),
    ])
    inp = tf.random_normal((10, 28, 28, 3))
    inp = tf.keras.Input(tensor=inp)
    inp2 = tf.random_normal((10, 28, 28, 3))
    inp2 = tf.keras.Input(tensor=inp2)

    fisher_approx = {layers.LayerNormalization: 'full', 'bn': 'diagonal'}
    _ = model(inp)
    _ = model(inp2)  # with multiple calls, the latest should be registered.
    lc = utils.get_layer_collection(model, 'mse', fisher_approx=fisher_approx)

    for i in (0, 2):
      conv_block = lc.fisher_blocks[model.layers[i].trainable_weights]
      conv_inp = model.layers[i].inbound_nodes[-1].input_tensors
      conv_out = model.layers[i].inbound_nodes[-1].output_tensors
      self.assertEqual(conv_inp, conv_block._inputs[0])
      self.assertEqual(conv_out, conv_block._outputs[0])

  @parameterized.named_parameters(
      ('_DictApprox', {layers.Dense: 'kron_in_diag',
                       'l1': 'kron_out_diag',
                       'l3': 'kron_both_diag'}),
      ('_ListApprox', ['kron_out_diag', 'kron_in_diag', 'kron_both_diag']))
  def testNestedModels(self, fisher_approx):
    # Note this is not a valid trainable model, it was just created to test
    # order of the dict and list test the DFS order in utils as well.
    layer1 = layers.Dense(10, input_shape=(1,), name='l1')
    layer2 = layers.Dense(10, activation='relu', name='l2')
    layer3 = layers.Dense(10, activation='relu', name='l3')

    inner_model0 = tf.keras.Sequential([layer1])

    inner_model1 = tf.keras.Sequential()
    inner_model1.add(inner_model0)
    inner_model1.add(layers.Activation('relu'))
    inner_model1.add(layer2)

    inner_inp = layers.Input(shape=(1,))
    x = layer3(inner_inp)
    x = layers.Reshape(target_shape=(10, 1))(x)
    x = layers.GlobalMaxPool1D()(x)
    inner_model2 = tf.keras.Model(inputs=inner_inp, outputs=x)

    inp = layers.Input(shape=(1,))
    branch1 = inner_model1(inp)
    branch2 = inner_model2(inp)
    out = layers.Add()([branch1, branch2])
    model = tf.keras.Model(inputs=inp, outputs=out)

    lc = utils.get_layer_collection(
        model=model, loss='mse', fisher_approx=fisher_approx)

    expected_in_diag_approx = [False, True, True]
    expected_out_diag_approx = [True, False, True]
    trainable_layers = [layer1, layer2, layer3]
    for layer, in_diag, out_diag in zip(trainable_layers,
                                        expected_in_diag_approx,
                                        expected_out_diag_approx):
      self.assertIsInstance(lc.fisher_blocks[layer.weights],
                            fisher_blocks.FullyConnectedKFACBasicFB)
      self.assertEqual(
          in_diag, lc.fisher_blocks[layer.weights]._diagonal_approx_for_input)
      self.assertEqual(
          out_diag, lc.fisher_blocks[layer.weights]._diagonal_approx_for_output)

  def testMultiOutputNestedModelFails(self):
    inp = tf.keras.Input(shape=(1,))
    out1 = layers.Dense(1)(inp)
    out2 = layers.Dense(1)(inp)
    model = tf.keras.Model(inputs=inp, outputs=[out1, out2])

    inp2 = tf.keras.Input(shape=(1,))
    out = model(inp2)
    model = tf.keras.Model(inputs=inp2, outputs=out)

    with self.assertRaisesRegex(
        ValueError, 'Nested models with multiple outputs are unsupported.'):
      utils.get_layer_collection(model, loss=['mse', 'mse'])


class SerializeLossTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('_String', 'binary_crossentropy', 'binary_crossentropy'),
      ('_KerasLoss', losses.binary_crossentropy, 'binary_crossentropy'),
      ('_Dict',
       {'out1': 'binary_crossentropy', 'out2': losses.mean_squared_error},
       {'out1': 'binary_crossentropy', 'out2': 'mean_squared_error'}),
      ('_List',
       ['mse', tf.keras.losses.categorical_crossentropy],
       ['mse', 'categorical_crossentropy']))
  def testSerializeLoss(self, loss, correctly_serialized_loss):
    serialized_loss = utils.serialize_loss(loss)
    self.assertEqual(serialized_loss, correctly_serialized_loss)


class GetLossFnTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GetLossFnTest, self).setUp()
    tf.reset_default_graph()
    tf.random.set_random_seed(_SEED)

  @parameterized.parameters(
      ('categorical_crossentropy', (11, 10), True, True),
      ('sparse_categorical_crossentropy', (11,), True, False),
      ('categorical_crossentropy', (11, 10), False, True),
      ('sparse_categorical_crossentropy', (11,), False, False),
      (losses.CategoricalCrossentropy(), (11, 10), True, True),
      (losses.categorical_crossentropy, (11, 10), False, True))
  def testCrossEntropy(self, loss, label_shape, is_logits, use_regularization):
    conv_kwargs = {'kernel_regularizer': 'l2'} if use_regularization else {}
    model_layers = [
        layers.Conv2D(7, 5, input_shape=(32, 32, 3), **conv_kwargs),
        layers.Activation('relu'),
        layers.Conv2D(10, (3, 3), activation='relu', **conv_kwargs),
        layers.GlobalMaxPool2D()
    ]
    if is_logits:
      model_layers.append(layers.Activation('softmax'))
    model = tf.keras.Sequential(model_layers)
    model.compile('sgd', loss)
    loss_fn = utils.get_loss_fn(model=model, loss=loss)

    x = tf.constant(np.random.random((11, 32, 32, 3)).astype(np.float32))
    y = tf.constant(np.random.random(label_shape).astype(np.float32))
    model_loss = model.evaluate(x, y, steps=1)
    fn_loss = tf.keras.backend.get_value(loss_fn((x, y)))
    fn_loss_w_pred = tf.keras.backend.get_value(
        loss_fn((x, y), prediction=model(x)))
    self.assertAlmostEqual(model_loss, fn_loss, fn_loss_w_pred)

    model.train_on_batch(np.random.random((11, 32, 32, 3)),
                         np.random.random(label_shape))

    x = tf.constant(np.random.random((11, 32, 32, 3)).astype(np.float32))
    y = tf.constant(np.random.random(label_shape).astype(np.float32))
    model_loss = model.test_on_batch(x, y)
    fn_loss = tf.keras.backend.get_value(loss_fn((x, y)))
    fn_loss_w_pred = tf.keras.backend.get_value(
        loss_fn((x, y), prediction=model(x)))
    self.assertAlmostEqual(model_loss, fn_loss, fn_loss_w_pred)

  @parameterized.parameters('categorical_crossentropy',
                            losses.CategoricalCrossentropy(),
                            losses.CategoricalCrossentropy(from_logits=False),
                            losses.categorical_crossentropy)
  def testCrossEntropyCustomLoop(self, loss):
    model_layers = [
        layers.Conv2D(7, 5, input_shape=(32, 32, 3)),
        layers.Activation('relu'),
        layers.Conv2D(10, (3, 3), kernel_regularizer='l2'),
        layers.GlobalMaxPool2D()
    ]
    model = tf.keras.Sequential(model_layers)
    model.compile('sgd', loss)
    loss_fn = utils.get_loss_fn(model=model, loss=loss)

    x = np.random.random((11, 32, 32, 3)).astype(np.float32)
    y = np.random.random((11, 10)).astype(np.float32)
    tf_x = tf.constant(x)
    tf_y = tf.constant(y)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      model_loss = sess.run(
          model.total_loss,
          feed_dict={'conv2d_input:0': x, 'global_max_pooling2d_target:0': y})
      fn_loss = sess.run(loss_fn((tf_x, tf_y)))
      fn_loss_w_pred = sess.run(loss_fn((tf_x, tf_y), prediction=model(tf_x)))
    self.assertAlmostEqual(model_loss, fn_loss, fn_loss_w_pred)

  @parameterized.parameters(
      'mse', 'MSE', 'mean_squared_error', losses.mean_squared_error)
  def testMSE(self, loss):
    model = _mlp()
    model.compile('sgd', loss)
    loss_fn = utils.get_loss_fn(model=model, loss=loss)

    x = tf.constant(np.random.random((23, 1)).astype(np.float32))
    y = tf.constant(np.random.random((23, 1)).astype(np.float32))
    model_loss = model.test_on_batch(x, y)
    fn_loss = tf.keras.backend.get_value(loss_fn((x, y)))
    fn_loss_w_pred = tf.keras.backend.get_value(
        loss_fn((x, y), prediction=model(x)))
    self.assertAlmostEqual(model_loss, fn_loss, fn_loss_w_pred)

  @parameterized.parameters(
      ({'out1': 'mse', 'out2': losses.categorical_crossentropy},
       [0.3, 0.7]),
      (['categorical_crossentropy', losses.MeanSquaredError()],
       {'out2': 0.1}))
  def testMultiLoss(self, multi_loss, loss_weights):
    inps, outs = _two_loss_model()
    model = tf.keras.Model(inputs=inps, outputs=outs)
    model.compile('sgd', multi_loss, loss_weights=loss_weights)
    loss_fn = utils.get_loss_fn(
        model=model, loss=multi_loss, loss_weights=loss_weights)

    x = tf.constant(np.random.random((11, 28, 28, 1)).astype(np.float32))
    y_1 = tf.constant(np.random.random((11, 1)).astype(np.float32))
    y_2 = tf.constant(np.random.random((11, 9)).astype(np.float32))
    # test_on_batch returns the total loss and the two individual losses.
    # We just want the total, so we use model_loss[0].
    model_loss = model.test_on_batch(x, [y_1, y_2])[0]
    fn_loss = tf.keras.backend.get_value(loss_fn((x, [y_1, y_2])))
    fn_loss_w_pred = tf.keras.backend.get_value(
        loss_fn((x, [y_1, y_2]), prediction=model(x)))
    self.assertAlmostEqual(model_loss, fn_loss, fn_loss_w_pred)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
