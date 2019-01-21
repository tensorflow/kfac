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
"""Tests for kfac.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from kfac.python.ops import utils


class SequenceDictTest(tf.test.TestCase):

  def testSequenceDictInit(self):
    seq_dict = utils.SequenceDict()
    self.assertFalse(seq_dict._dict)

  def testSequenceDictInitWithIterable(self):
    reg_dict = {'a': 'foo', 'b': 'bar'}
    itr = zip(reg_dict.keys(), reg_dict.values())
    seq_dict = utils.SequenceDict(itr)
    self.assertEqual(reg_dict, seq_dict._dict)

  def testGetItemSingleKey(self):
    seq_dict = utils.SequenceDict({'a': 'foo', 'b': 'bar'})
    self.assertEqual('foo', seq_dict['a'])

  def testGetItemMultipleKeys(self):
    seq_dict = utils.SequenceDict({'a': 'foo', 'b': 'bar'})
    self.assertEqual(['foo', 'bar'], seq_dict[('a', 'b')])

  def testSetItemSingleKey(self):
    seq_dict = utils.SequenceDict()
    seq_dict['a'] = 'foo'
    self.assertEqual([('a', 'foo')], seq_dict.items())

  def testSetItemMultipleKeys(self):
    seq_dict = utils.SequenceDict()
    keys = ('a', 'b', 'c')
    values = ('foo', 'bar', 'baz')
    seq_dict[keys] = values
    self.assertItemsEqual(list(zip(keys, values)), seq_dict.items())


class SubGraphTest(tf.test.TestCase):

  def testBasicGraph(self):
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[5., 6.], [7., 8.]])
    c = a + b
    d = a * b
    sub_graph = utils.SubGraph((c,))
    self.assertTrue(sub_graph.is_member(a))
    self.assertTrue(sub_graph.is_member(b))
    self.assertTrue(sub_graph.is_member(c))
    self.assertFalse(sub_graph.is_member(d))

  def testRepeatedAdds(self):
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[5., 6.], [7., 8.]])
    c = a + b + a  # note that a appears twice in this graph
    sub_graph = utils.SubGraph((c,))
    self.assertTrue(sub_graph.is_member(a))
    self.assertTrue(sub_graph.is_member(b))
    self.assertTrue(sub_graph.is_member(c))

  def testFilterList(self):
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[5., 6.], [7., 8.]])
    c = a + b
    d = a * b
    sub_graph = utils.SubGraph((c,))
    input_list = [b, d]
    filtered_list = sub_graph.filter_list(input_list)
    self.assertEqual(filtered_list, [b])

  def testVariableUses(self):
    with tf.Graph().as_default():
      var = tf.get_variable('var', shape=[10, 10])
      resource_var = tf.get_variable(
          'resource_var', shape=[10, 10], use_resource=True)
      x = tf.zeros([3, 10])
      z0 = tf.matmul(x, var) + tf.matmul(x, var)
      z1 = tf.matmul(x, resource_var)
      sub_graph = utils.SubGraph((z0, z1))
      self.assertEqual(2, sub_graph.variable_uses(var))
      self.assertEqual(1, sub_graph.variable_uses(resource_var))

  def testVariableUsesRelayOps(self):
    with tf.Graph().as_default():
      a = tf.get_variable("a", shape=[2, 2])
      b = tf.get_variable("b", shape=[2, 2])
      ai = tf.identity(a)
      c = tf.matmul(ai, b)
      d = tf.matmul(ai, b)

      sub_graph = utils.SubGraph((c, d))
      self.assertEqual(2, sub_graph.variable_uses(a))
      self.assertEqual(2, sub_graph.variable_uses(b))


class UtilsTest(tf.test.TestCase):

  def _fully_connected_layer_params(self):
    weights_part = tf.constant([[1., 2.], [4., 3.]])
    bias_part = tf.constant([1., 2.])
    return (weights_part, bias_part)

  def _conv_layer_params(self):
    weights_shape = 2, 2, 3, 4
    biases_shape = weights_shape[-1:]
    weights = tf.constant(np.random.RandomState(0).randn(*weights_shape))
    biases = tf.constant(np.random.RandomState(1).randn(*biases_shape))
    return (weights, biases)

  def testFullyConnectedLayerParamsTupleToMat2d(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      layer_params = self._fully_connected_layer_params()
      output = utils.layer_params_to_mat2d(layer_params)
      self.assertListEqual([3, 2], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output), np.array([[1., 2.], [4., 3.], [1., 2.]]))

  def testFullyConnectedLayerParamsTensorToMat2d(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      layer_params = self._fully_connected_layer_params()
      output = utils.layer_params_to_mat2d(layer_params[0])
      self.assertListEqual([2, 2], output.get_shape().as_list())
      self.assertAllClose(sess.run(output), np.array([[1., 2.], [4., 3.]]))

  def testConvLayerParamsTupleToMat2d(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      layer_params = self._conv_layer_params()
      output = utils.layer_params_to_mat2d(layer_params)
      self.assertListEqual([2 * 2 * 3 + 1, 4], output.get_shape().as_list())

  def testKron(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      mat1 = np.array([[1., 2.], [3., 4.]])
      mat2 = np.array([[5., 6.], [7., 8.]])
      mat1_tf = tf.constant(mat1)
      mat2_tf = tf.constant(mat2)
      ans_tf = sess.run(utils.kronecker_product(mat1_tf, mat2_tf))
      ans_np = np.kron(mat1, mat2)
      self.assertAllClose(ans_tf, ans_np)

  def testMat2dToFullyConnectedLayerParamsTuple(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      vector_template = self._fully_connected_layer_params()
      mat2d = tf.constant([[5., 4.], [3., 2.], [1., 0.]])

      output = sess.run(utils.mat2d_to_layer_params(vector_template, mat2d))

      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 2)
      a, b = output
      self.assertAllClose(a, np.array([[5., 4.], [3., 2.]]))
      self.assertAllClose(b, np.array([1., 0.]))

  def testMat2dToFullyConnectedLayerParamsTensor(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      vector_template = self._fully_connected_layer_params()[0]
      mat2d = tf.constant([[5., 4.], [3., 2.]])

      output = sess.run(utils.mat2d_to_layer_params(vector_template, mat2d))

      self.assertAllClose(output, np.array([[5., 4.], [3., 2.]]))

  def testTensorsToColumn(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)

      vector = tf.constant(np.array([[0., 1.], [2., 3.]]))
      output = utils.tensors_to_column(vector)
      self.assertListEqual([4, 1], output.get_shape().as_list())
      self.assertAllClose(sess.run(output), np.array([0., 1., 2., 3.])[:, None])

      vector = self._fully_connected_layer_params()
      output = utils.tensors_to_column(vector)
      self.assertListEqual([6, 1], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output), np.array([1., 2., 4., 3., 1., 2.])[:, None])

      vector = list(vector)
      vector.append(tf.constant([[6.], [7.], [8.], [9.]]))

      output = utils.tensors_to_column(vector)
      self.assertListEqual([10, 1], output.get_shape().as_list())
      self.assertAllClose(
          sess.run(output),
          np.array([1., 2., 4., 3., 1., 2., 6., 7., 8., 9.])[:, None])

  def testColumnToTensors(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)

      vector_template = tf.constant(np.array([[0., 1.], [2., 3.]]))
      colvec = tf.constant(np.arange(4.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))
      self.assertAllClose(output, np.array([[0., 1.], [2., 3.]]))

      vector_template = self._fully_connected_layer_params()
      colvec = tf.constant(np.arange(6.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))

      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 2)
      a, b = output
      self.assertAllClose(a, np.array([[0., 1.], [2., 3.]]))
      self.assertAllClose(b, np.array([4., 5.]))

      vector_template = list(vector_template)
      vector_template.append(tf.constant([[6.], [7.], [8.], [9.]]))
      colvec = tf.constant(np.arange(10.)[:, None])
      output = sess.run(utils.column_to_tensors(vector_template, colvec))
      self.assertIsInstance(output, tuple)
      self.assertEqual(len(output), 3)
      a, b, c = output
      self.assertAllClose(a, np.array([[0., 1.], [2., 3.]]))
      self.assertAllClose(b, np.array([4., 5.]))
      self.assertAllClose(c, np.array([[6.], [7.], [8.], [9.]]))

  def testPosDefInvCholesky(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      np.random.seed(0)
      square = lambda x: np.dot(x, x.T)

      size = 3
      x = square(np.random.randn(size, size))
      damp = 0.1
      identity = tf.eye(size, dtype=tf.float64)

      tf_inv = utils.posdef_inv_cholesky(tf.constant(x), identity, damp)
      np_inv = np.linalg.inv(x + damp * np.eye(size))
      self.assertAllClose(sess.run(tf_inv), np_inv)

  def testPosDefInvMatrixInverse(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      np.random.seed(0)
      square = lambda x: np.dot(x, x.T)

      size = 3
      x = square(np.random.randn(size, size))
      damp = 0.1
      identity = tf.eye(size, dtype=tf.float64)

      tf_inv = utils.posdef_inv_matrix_inverse(tf.constant(x), identity, damp)
      np_inv = np.linalg.inv(x + damp * np.eye(size))
      self.assertAllClose(sess.run(tf_inv), np_inv)

  def testCrossReplicaMean(self):
    """Ensures that cross_replica_mean() executes only when num_shards > 1."""
    with tf.Graph().as_default():
      with tpu_function.tpu_shard_context(4):
        tensor = tf.zeros([], dtype=tf.float32)
        mean = utils.cross_replica_mean(tensor)
      self.assertNotEqual(mean, tensor)

    with tf.Graph().as_default():
      with tpu_function.tpu_shard_context(1):
        tensor = tf.zeros([], dtype=tf.float32)
        mean = utils.cross_replica_mean(tensor)
      self.assertEqual(mean, tensor)

    with tf.Graph().as_default():
      with self.assertRaises(ValueError):  # Outside of TPU context.
        tensor = tf.zeros([], dtype=tf.float32)
        mean = utils.cross_replica_mean(tensor)

  def testBatchExecute(self):
    """Ensure batch_execute runs in a round-robin fashion."""

    def increment_var(var):
      return lambda: var.assign_add(1)

    with tf.Graph().as_default(), self.test_session() as sess:
      i = tf.get_variable('i', initializer=0)
      accumulators = [
          tf.get_variable('var%d' % j, initializer=0) for j in range(3)
      ]
      thunks = [increment_var(var) for var in accumulators]
      increment_accumulators = utils.batch_execute(i, thunks, 2)
      increment_i = i.assign_add(1)

      sess.run(tf.global_variables_initializer())

      # Ensure one op per thunk.
      self.assertEqual(3, len(increment_accumulators))

      # Ensure round-robin execution.
      values = []
      for _ in range(5):
        sess.run(increment_accumulators)
        sess.run(increment_i)
        values.append(sess.run(accumulators))
      self.assertAllClose(
          [
              [1, 1, 0],  #
              [2, 1, 1],  #
              [2, 2, 2],  #
              [3, 3, 2],  #
              [4, 3, 3]
          ],
          values)

  def testExtractConvolutionPatches(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      batch_size = 10
      image_spatial_shape = [9, 10, 11]
      in_channels = out_channels = 32
      kernel_spatial_shape = [5, 3, 3]
      spatial_strides = [1, 2, 1]
      spatial_dilation = [1, 1, 1]
      padding = 'SAME'

      images = tf.random_uniform(
          [batch_size] + image_spatial_shape + [in_channels], seed=0)
      kernel_shape = kernel_spatial_shape + [in_channels, out_channels]
      kernel = tf.random_uniform(kernel_shape, seed=1)

      # Ensure shape matches expectation.
      patches = utils.extract_convolution_patches(
          images,
          kernel_shape,
          padding,
          strides=spatial_strides,
          dilation_rate=spatial_dilation)
      result_spatial_shape = (
          patches.shape.as_list()[1:1 + len(image_spatial_shape)])
      self.assertEqual(patches.shape.as_list(),
                       [batch_size] + result_spatial_shape +
                       kernel_spatial_shape + [in_channels])

      # Ensure extract...patches() + matmul() and convolution() implementation
      # give the same answer.
      outputs = tf.nn.convolution(
          images,
          kernel,
          padding,
          strides=spatial_strides,
          dilation_rate=spatial_dilation)

      patches_flat = tf.reshape(
          patches, [-1, np.prod(kernel_spatial_shape) * in_channels])
      kernel_flat = tf.reshape(kernel, [-1, out_channels])
      outputs_flat = tf.matmul(patches_flat, kernel_flat)

      outputs_, outputs_flat_ = sess.run([outputs, outputs_flat])
      self.assertAllClose(outputs_.flatten(), outputs_flat_.flatten())

  def testExtractPointwiseConv2dPatches(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      batch_size = 10
      image_height = image_width = 8
      in_channels = out_channels = 3
      kernel_height = kernel_width = 1
      strides = [1, 1, 1, 1]
      padding = 'VALID'

      images = tf.random_uniform(
          [batch_size, image_height, image_width, in_channels], seed=0)
      kernel_shape = [kernel_height, kernel_width, in_channels, out_channels]
      kernel = tf.random_uniform(kernel_shape, seed=1)

      # Ensure shape matches expectation.
      patches = utils.extract_pointwise_conv2d_patches(images, kernel_shape)
      self.assertEqual(patches.shape.as_list(), [
          batch_size, image_height, image_width, kernel_height, kernel_width,
          in_channels
      ])

      # Ensure extract...patches() + matmul() and conv2d() implementation
      # give the same answer.
      outputs = tf.nn.conv2d(images, kernel, strides, padding)

      patches_flat = tf.reshape(
          patches, [-1, kernel_height * kernel_width * in_channels])
      kernel_flat = tf.reshape(kernel, [-1, out_channels])
      outputs_flat = tf.matmul(patches_flat, kernel_flat)

      outputs_, outputs_flat_ = sess.run([outputs, outputs_flat])
      self.assertAllClose(outputs_.flatten(), outputs_flat_.flatten())


class AccumulatorVariableTest(tf.test.TestCase):

  def test_assign_to_var(self):
    var_shape = (2, 3)
    var = tf.get_variable(
        shape=var_shape,
        name='orig_var',
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
    acc_var = utils.AccumulatorVariable(
        name='test_acc_var', var=var)
    values = [
        3. * tf.ones(shape=var.shape), 7. * tf.ones(shape=var.shape),
        11. * tf.ones(shape=var.shape)
    ]
    acc_ops = []
    accc_ops_after_reset = []
    for value in values:
      acc_ops.append(
          acc_var.accumulate(value, num_steps_for_update=len(values)))

    for value in values[:2]:
      accc_ops_after_reset.append(
          acc_var.accumulate(value, num_steps_for_update=2))

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init_op])
      for acc_op in acc_ops:
        sess.run(acc_op)

      acc_value, acc_var_value = sess.run([var, acc_var.value])
      self.assertAllEqual(acc_value, 7. * np.ones(shape=var_shape))
      self.assertAllEqual(acc_var_value, np.zeros(shape=var_shape))

      for acc_op in accc_ops_after_reset:
        sess.run(acc_op)

      acc_value, acc_var_value = sess.run([var, acc_var.value])
      self.assertAllEqual(acc_value, 5. * np.ones(shape=var_shape))
      self.assertAllEqual(acc_var_value, np.zeros(shape=var_shape))

  def test_accumulation(self):
    var_shape = (2, 3)
    acc_var = utils.AccumulatorVariable(
        name='test_acc_var', acc_var_shape=var_shape, acc_var_dtype=tf.float32)
    values = [
        2. * tf.ones(shape=var_shape), 4. * tf.ones(shape=var_shape),
        9. * tf.ones(shape=var_shape)
    ]
    acc_ops = []
    accc_ops_after_reset = []
    for value in values:
      acc_ops.append(
          acc_var.accumulate(value, num_steps_for_update=len(values)))

    for value in values[:2]:
      accc_ops_after_reset.append(
          acc_var.accumulate(value, num_steps_for_update=2))

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init_op])
      acc_value = None

      for acc_op in acc_ops:
        sess.run([acc_op])

      acc_value, acc_var_value = sess.run(
          [acc_var.accumulated_value, acc_var.value])
      self.assertAllEqual(acc_value, 5. * np.ones(shape=var_shape))
      self.assertAllEqual(acc_var_value, np.zeros(shape=var_shape))

      for acc_op in accc_ops_after_reset:
        sess.run([acc_op])

      acc_value, acc_var_value = sess.run(
          [acc_var.accumulated_value, acc_var.value])
      self.assertAllEqual(acc_value, 3. * np.ones(shape=var_shape))
      self.assertAllEqual(acc_var_value, np.zeros(shape=var_shape))


if __name__ == '__main__':
  tf.test.main()
