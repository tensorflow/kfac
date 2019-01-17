# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for kfac.layer_collection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from kfac.python.ops import fisher_blocks
from kfac.python.ops import fisher_factors
from kfac.python.ops import layer_collection


class MockFisherBlock(object):
  """A fake FisherBlock."""

  num_registered_towers = 2

  def __init__(self, name='MockFisherBlock'):
    self.name = name

  def __eq__(self, other):
    return isinstance(other, MockFisherBlock) and other.name == self.name

  def __hash__(self):
    return hash(self.name)


class LayerParametersDictTest(tf.test.TestCase):

  def testSetItem(self):
    """Ensure insertion, contains, retrieval works for supported key types."""
    with tf.Graph().as_default():
      lp_dict = layer_collection.LayerParametersDict()

      x = tf.constant(0)
      y0 = tf.constant(0)
      y1 = tf.constant(0)
      z0 = tf.constant(0)
      z1 = tf.constant(0)
      keys = [x, (y0, y1), [z0, z1]]
      for key in keys:
        lp_dict[key] = key

      for key in keys:
        self.assertTrue(key in lp_dict)
        self.assertEqual(lp_dict[key], key)

  def testSetItemOverlap(self):
    """Ensure insertion fails if key overlaps with existing key."""
    with tf.Graph().as_default():
      lp_dict = layer_collection.LayerParametersDict()

      x = tf.constant(0)
      y = tf.constant(0)
      lp_dict[x] = 'value'

      with self.assertRaises(ValueError):
        lp_dict[(x, y)] = 'value'

      # Ensure 'y' wasn't inserted.
      self.assertTrue(x in lp_dict)
      self.assertFalse(y in lp_dict)


class LayerCollectionTest(tf.test.TestCase):

  def testLayerCollectionInit(self):
    lc = layer_collection.LayerCollection()
    self.assertEqual(0, len(lc.get_blocks()))
    self.assertEqual(0, len(lc.get_factors()))
    self.assertFalse(lc.losses)

  def testRegisterBlocks(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      lc.register_fully_connected(
          tf.constant(1), tf.constant(2), tf.constant(3))
      lc.register_fully_connected(
          tf.constant(1),
          tf.constant(2),
          tf.constant(3),
          approx=layer_collection.APPROX_DIAGONAL_NAME)
      lc.register_conv2d(
          params=tf.ones((2, 3, 4, 5)),
          strides=[1, 1, 1, 1],
          padding='SAME',
          inputs=tf.ones((1, 2, 3, 4)),
          outputs=tf.ones((1, 1, 1, 5)))
      lc.register_conv2d(
          params=tf.ones((2, 3, 4, 5)),
          strides=[1, 1, 1, 1],
          padding='SAME',
          inputs=tf.ones((1, 2, 3, 4)),
          outputs=tf.ones((1, 1, 1, 5)),
          approx=layer_collection.APPROX_DIAGONAL_NAME)
      lc.register_separable_conv2d(
          depthwise_params=tf.ones((3, 3, 1, 2)),
          pointwise_params=tf.ones((1, 1, 2, 4)),
          inputs=tf.ones((32, 5, 5, 1)),
          depthwise_outputs=tf.ones((32, 5, 5, 2)),
          pointwise_outputs=tf.ones((32, 5, 5, 4)),
          strides=[1, 1, 1, 1],
          padding='SAME')
      lc.register_convolution(
          params=tf.ones((3, 3, 1, 8)),
          inputs=tf.ones((32, 5, 5, 1)),
          outputs=tf.ones((32, 5, 5, 8)),
          padding='SAME')
      lc.register_generic(
          tf.constant(5), 16, approx=layer_collection.APPROX_FULL_NAME)
      lc.register_generic(
          tf.constant(6), 16, approx=layer_collection.APPROX_DIAGONAL_NAME)
      lc.register_fully_connected_multi(
          tf.constant(1), (tf.constant(2), tf.constant(3)),
          (tf.constant(4), tf.constant(5)))
      lc.register_conv2d_multi(
          params=tf.ones((2, 3, 4, 5)),
          strides=[1, 1, 1, 1],
          padding='SAME',
          inputs=(tf.ones((1, 2, 3, 4)), tf.ones((5, 6, 7, 8))),
          outputs=(tf.ones((1, 1, 1, 5)), tf.ones((2, 2, 2, 10))))
      lc.register_embedding_multi(
          tf.constant((1,)), (tf.constant(2), tf.constant(3)),
          (tf.constant(4), tf.constant(5)))
      lc.register_embedding_multi(
          tf.constant((1,)), (tf.constant(2), tf.constant(3)),
          (tf.constant(4), tf.constant(5)), transpose=[False, True])

      self.assertEqual(13, len(lc.get_blocks()))

  def testRegisterBlocksMultipleRegistrations(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      key = tf.constant(1)
      lc.register_fully_connected(key, tf.constant(2), tf.constant(3))
      with self.assertRaises(ValueError) as cm:
        lc.register_generic(key, 16)
      self.assertIn('already in LayerCollection', str(cm.exception))

  def testRegisterSingleParamNotRegistered(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {tf.get_variable('y', initializer=tf.constant(1,)): '1'}
    lc.register_block(x, 'foo')

  def testShouldRegisterSingleParamRegistered(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {x: '1'}
    with self.assertRaises(ValueError) as cm:
      lc.register_block(x, 'foo')
    self.assertIn('already in LayerCollection', str(cm.exception))

  def testRegisterSingleParamRegisteredInTuple(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y): '1'}
    with self.assertRaises(ValueError) as cm:
      lc.register_block(x, 'foo')
    self.assertIn('was already registered', str(cm.exception))

  def testRegisterTupleParamNotRegistered(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {tf.get_variable('z', initializer=tf.constant(1,)): '1'}

    lc.register_block((x, y), 'foo')
    self.assertEqual(set(['1', 'foo']), set(lc.get_blocks()))

  def testRegisterTupleParamRegistered(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y): '1'}

    with self.assertRaises(ValueError) as cm:
      lc.register_block((x, y), 'foo')
    self.assertIn('already in LayerCollection', str(cm.exception))

  def testRegisterTupleParamRegisteredInSuperset(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    z = tf.get_variable('z', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y, z): '1'}

    with self.assertRaises(ValueError) as cm:
      lc.register_block((x, y), 'foo')
    self.assertIn('was already registered', str(cm.exception))

  def testRegisterTupleParamSomeRegistered(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    z = tf.get_variable('z', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {x: MockFisherBlock('1'), z: MockFisherBlock('2')}

    with self.assertRaises(ValueError) as cm:
      lc.register_block((x, y), MockFisherBlock('foo'))
    self.assertIn('was already registered', str(cm.exception))

  def testRegisterTupleVarSomeRegisteredInOtherTuples(self):
    x = tf.get_variable('x', initializer=tf.constant(1,))
    y = tf.get_variable('y', initializer=tf.constant(1,))
    z = tf.get_variable('z', initializer=tf.constant(1,))
    w = tf.get_variable('w', initializer=tf.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, z): '1', (z, w): '2'}

    with self.assertRaises(ValueError) as cm:
      lc.register_block((x, y), 'foo')
    self.assertIn('was already registered', str(cm.exception))

  def testRegisterCategoricalPredictiveDistribution(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      logits = tf.eye(2)

      lc = layer_collection.LayerCollection()
      lc.register_categorical_predictive_distribution(logits, seed=200)
      single_loss = sess.run(lc.total_sampled_loss())

      lc2 = layer_collection.LayerCollection()
      lc2.register_categorical_predictive_distribution(logits, seed=200)
      lc2.register_categorical_predictive_distribution(logits, seed=200)
      double_loss = sess.run(lc2.total_sampled_loss())
      self.assertAlmostEqual(2 * single_loss, double_loss)

  def testLossFunctionByName(self):
    """Ensure loss functions can be identified by name."""
    with tf.Graph().as_default():
      logits = tf.eye(2)
      lc = layer_collection.LayerCollection()

      # Create a new loss function by name.
      lc.register_categorical_predictive_distribution(logits, name='loss1')
      self.assertEqual(1, len(lc.towers_by_loss))

      # Add logits to same loss function.
      lc.register_categorical_predictive_distribution(
          logits, name='loss1', reuse=True)
      self.assertEqual(1, len(lc.towers_by_loss))

      # Add another new loss function.
      lc.register_categorical_predictive_distribution(logits, name='loss2')
      self.assertEqual(2, len(lc.towers_by_loss))

  def testLossFunctionWithoutName(self):
    """Ensure loss functions get unique names if 'name' not specified."""
    with tf.Graph().as_default():
      logits = tf.eye(2)
      lc = layer_collection.LayerCollection()

      # Create a new loss function with default names.
      lc.register_categorical_predictive_distribution(logits)
      lc.register_categorical_predictive_distribution(logits)
      self.assertEqual(2, len(lc.losses))

  def testCategoricalPredictiveDistributionMultipleMinibatches(self):
    """Ensure multiple minibatches are registered."""
    with tf.Graph().as_default():
      batch_size = 3
      output_size = 2
      logits = tf.zeros([batch_size, output_size])
      targets = tf.ones([batch_size], dtype=tf.int32)
      lc = layer_collection.LayerCollection()

      # Create a new loss function.
      lc.register_categorical_predictive_distribution(
          logits, targets=targets, name='loss1')

      # Can add when reuse=True
      lc.register_categorical_predictive_distribution(
          logits, targets=targets, name='loss1', reuse=True)

      # Can add when reuse=VARIABLE_SCOPE and reuse=True there.
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        lc.register_categorical_predictive_distribution(
            logits,
            targets=targets,
            name='loss1',
            reuse=layer_collection.VARIABLE_SCOPE)

      # Can't add when reuse=False
      with self.assertRaises(KeyError):
        lc.register_categorical_predictive_distribution(
            logits, targets=targets, name='loss1', reuse=False)

      # Can't add when reuse=VARIABLE_SCOPE and reuse=False there.
      with self.assertRaises(KeyError):
        lc.register_categorical_predictive_distribution(
            logits,
            targets=targets,
            name='loss1',
            reuse=layer_collection.VARIABLE_SCOPE)

      self.assertEqual(len(lc.towers_by_loss), 1)
      # Three successful registrations.
      self.assertEqual(len(lc.towers_by_loss[0]), 3)

  def testRegisterCategoricalPredictiveDistributionBatchSize1(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      logits = tf.random_normal((1, 2))
      lc = layer_collection.LayerCollection()

      lc.register_categorical_predictive_distribution(logits, seed=200)

  def testRegisterCategoricalPredictiveDistributionSpecifiedTargets(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      logits = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
      lc = layer_collection.LayerCollection()
      targets = tf.constant([0, 1], dtype=tf.int32)

      lc.register_categorical_predictive_distribution(logits, targets=targets)
      single_loss = sess.run(lc.total_loss())
      self.assertAlmostEqual(1.6265233, single_loss)

  def testRegisterNormalPredictiveDistribution(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      predictions = tf.constant([[1., 2.], [3., 4]], dtype=tf.float32)

      lc = layer_collection.LayerCollection()
      lc.register_normal_predictive_distribution(predictions, 1., seed=200)
      single_loss = sess.run(lc.total_sampled_loss())

      lc2 = layer_collection.LayerCollection()
      lc2.register_normal_predictive_distribution(predictions, 1., seed=200)
      lc2.register_normal_predictive_distribution(predictions, 1., seed=200)
      double_loss = sess.run(lc2.total_sampled_loss())

      self.assertAlmostEqual(2 * single_loss, double_loss)

  def testRegisterNormalPredictiveDistributionSpecifiedTargets(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      predictions = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
      lc = layer_collection.LayerCollection()
      targets = tf.constant([[3., 1.], [4., 2.]], dtype=tf.float32)

      lc.register_normal_predictive_distribution(
          predictions, 2.**2, targets=targets)
      single_loss = sess.run(lc.total_loss())
      self.assertAlmostEqual(7.6983433, single_loss)

  def ensureLayerReuseWorks(self, register_fn):
    """Ensure the 'reuse' keyword argument function as intended.

    Args:
      register_fn: function for registering a layer. Arguments are
        layer_collection, reuse, and approx.
    """
    # Fails on second if reuse=False.
    lc = layer_collection.LayerCollection()
    register_fn(lc)
    with self.assertRaises(ValueError):
      register_fn(lc, reuse=False)

    # Succeeds on second if reuse=True.
    lc = layer_collection.LayerCollection()
    register_fn(lc)
    register_fn(lc, reuse=True)

    # Fails on second if reuse=VARIABLE_SCOPE and no variable reuse.
    lc = layer_collection.LayerCollection()
    register_fn(lc)
    with self.assertRaises(ValueError):
      register_fn(lc, reuse=layer_collection.VARIABLE_SCOPE)

    # Succeeds on second if reuse=VARIABLE_SCOPE and variable reuse.
    lc = layer_collection.LayerCollection()
    register_fn(lc)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      register_fn(lc, reuse=layer_collection.VARIABLE_SCOPE)

    # Fails if block type changes.
    lc = layer_collection.LayerCollection()
    register_fn(lc, approx=layer_collection.APPROX_KRONECKER_NAME)
    with self.assertRaises(ValueError):
      register_fn(lc, approx=layer_collection.APPROX_DIAGONAL_NAME, reuse=True)

    # Fails if reuse requested but no FisherBlock exists.
    lc = layer_collection.LayerCollection()
    with self.assertRaises(KeyError):
      register_fn(lc, reuse=True)

  def testRegisterFullyConnectedReuse(self):
    """Ensure the 'reuse' works with register_fully_connected."""
    with tf.Graph().as_default():
      inputs = tf.ones([2, 10])
      outputs = tf.zeros([2, 5])
      params = (
          tf.get_variable('w', [10, 5]),  #
          tf.get_variable('b', [5]))

      def register_fn(lc, **kwargs):
        lc.register_fully_connected(
            params=params, inputs=inputs, outputs=outputs, **kwargs)

      self.ensureLayerReuseWorks(register_fn)

  def testRegisterConv2dReuse(self):
    """Ensure the 'reuse' works with register_conv2d."""
    with tf.Graph().as_default():
      inputs = tf.ones([2, 5, 5, 10])
      outputs = tf.zeros([2, 5, 5, 3])
      params = (
          tf.get_variable('w', [1, 1, 10, 3]),  #
          tf.get_variable('b', [3]))

      def register_fn(lc, **kwargs):
        lc.register_conv2d(
            params=params,
            strides=[1, 1, 1, 1],
            padding='SAME',
            inputs=inputs,
            outputs=outputs,
            **kwargs)

      self.ensureLayerReuseWorks(register_fn)

  def testReuseWithInvalidRegistration(self):
    """Invalid registrations shouldn't overwrite existing blocks."""
    with tf.Graph().as_default():
      inputs = tf.ones([2, 5, 5, 10])
      outputs = tf.zeros([2, 5, 5, 3])
      w = tf.get_variable('w', [1, 1, 10, 3])
      b = tf.get_variable('b', [3])
      lc = layer_collection.LayerCollection()
      lc.register_fully_connected(w, inputs, outputs)
      self.assertEqual(lc.fisher_blocks[w].num_registered_towers, 1)
      with self.assertRaises(KeyError):
        lc.register_fully_connected((w, b), inputs, outputs, reuse=True)
      self.assertNotIn((w, b), lc.fisher_blocks)
      self.assertEqual(lc.fisher_blocks[w].num_registered_towers, 1)
      lc.register_fully_connected(w, inputs, outputs, reuse=True)
      self.assertEqual(lc.fisher_blocks[w].num_registered_towers, 2)

  def testMakeOrGetFactor(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      key = tf.constant(1)
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((tf.constant(2),), 16))

      self.assertEqual(2, len(lc.get_factors()))
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      self.assertTrue(
          all([var.name.startswith('LayerCollection') for var in variables]))

  def testMakeOrGetFactorCustomScope(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      scope = 'Foo'
      lc = layer_collection.LayerCollection(name=scope)
      key = tf.constant(1)
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((tf.constant(2),), 16))

      self.assertEqual(2, len(lc.get_factors()))
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      self.assertTrue(all([var.name.startswith(scope) for var in variables]))

  def testIdentifyLinkedParametersSomeRegisteredInOtherTuples(self):
    x = tf.get_variable('x', shape=())
    y = tf.get_variable('y', shape=())
    z = tf.get_variable('z', shape=())
    lc = layer_collection.LayerCollection()
    lc.define_linked_parameters((x, y))

    with self.assertRaises(ValueError):
      lc.define_linked_parameters((x, z))

  def testIdentifySubsetPreviouslyRegisteredTensor(self):
    x = tf.get_variable('x', shape=())
    y = tf.get_variable('y', shape=())
    lc = layer_collection.LayerCollection()
    lc.define_linked_parameters((x, y))

    with self.assertRaises(ValueError):
      lc.define_linked_parameters(x)

  def testSpecifyApproximation(self):
    w_0 = tf.get_variable('w_0', [10, 10])
    w_1 = tf.get_variable('w_1', [10, 10])

    b_0 = tf.get_variable('b_0', [10])
    b_1 = tf.get_variable('b_1', [10])

    x_0 = tf.placeholder(tf.float32, shape=(32, 10))
    x_1 = tf.placeholder(tf.float32, shape=(32, 10))

    pre_bias_0 = tf.matmul(x_0, w_0)
    pre_bias_1 = tf.matmul(x_1, w_1)

    # Build the fully connected layers in the graph.
    pre_bias_0 + b_0  # pylint: disable=pointless-statement
    pre_bias_1 + b_1  # pylint: disable=pointless-statement

    lc = layer_collection.LayerCollection()
    lc.define_linked_parameters(
        w_0, approximation=layer_collection.APPROX_DIAGONAL_NAME)
    lc.define_linked_parameters(
        w_1, approximation=layer_collection.APPROX_DIAGONAL_NAME)
    lc.define_linked_parameters(
        b_0, approximation=layer_collection.APPROX_FULL_NAME)
    lc.define_linked_parameters(
        b_1, approximation=layer_collection.APPROX_FULL_NAME)

    lc.register_fully_connected(w_0, x_0, pre_bias_0)
    lc.register_fully_connected(
        w_1, x_1, pre_bias_1, approx=layer_collection.APPROX_KRONECKER_NAME)
    self.assertIsInstance(lc.fisher_blocks[w_0],
                          fisher_blocks.FullyConnectedDiagonalFB)
    self.assertIsInstance(lc.fisher_blocks[w_1],
                          fisher_blocks.FullyConnectedKFACBasicFB)

    lc.register_generic(b_0, batch_size=1)
    lc.register_generic(
        b_1, batch_size=1, approx=layer_collection.APPROX_DIAGONAL_NAME)
    self.assertIsInstance(lc.fisher_blocks[b_0], fisher_blocks.FullFB)
    self.assertIsInstance(lc.fisher_blocks[b_1], fisher_blocks.NaiveDiagonalFB)

  def testDefaultLayerCollection(self):
    with tf.Graph().as_default():
      # Can't get default if there isn't one set.
      with self.assertRaises(ValueError):
        layer_collection.get_default_layer_collection()

      # Can't set default twice.
      lc = layer_collection.LayerCollection()
      layer_collection.set_default_layer_collection(lc)
      with self.assertRaises(ValueError):
        layer_collection.set_default_layer_collection(lc)

      # Same as one set.
      self.assertTrue(lc is layer_collection.get_default_layer_collection())

      # Can set to None.
      layer_collection.set_default_layer_collection(None)
      with self.assertRaises(ValueError):
        layer_collection.get_default_layer_collection()

      # as_default() is the same as setting/clearing.
      with lc.as_default():
        self.assertTrue(lc is layer_collection.get_default_layer_collection())
      with self.assertRaises(ValueError):
        layer_collection.get_default_layer_collection()


if __name__ == '__main__':
  tf.test.main()
