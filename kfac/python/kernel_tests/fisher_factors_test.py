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
"""Tests for kfac.fisher_factors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import numpy.random as npr

import tensorflow as tf

from tensorflow.python.framework import dtypes
from kfac.python.ops import fisher_blocks as fb
from kfac.python.ops import fisher_factors as ff

# We need to set these constants since the numerical values used in the tests
# were chosen when these used to be the defaults.
ff.set_global_constants(init_covariances_at_zero=False,
                        zero_debias=False,
                        init_inverses_at_zero=False,
                        max_num_patches_per_dimension=1.0)


def make_damping_func(damping):
  return fb._package_func(lambda: damping, damping)


class FisherFactorTestingDummy(ff.FisherFactor):
  """Dummy class to test the non-abstract methods on ff.FisherFactor."""

  @property
  def _var_scope(self):
    return 'dummy/a_b_c'

  @property
  def _cov_shape(self):
    raise NotImplementedError

  @property
  def _num_sources(self):
    return 1

  @property
  def _dtype(self):
    return tf.float32

  def _compute_new_cov(self):
    raise NotImplementedError

  def instantiate_covariance(self):
    pass

  def make_inverse_update_ops(self):
    return []

  @property
  def cov(self):
    return NotImplementedError

  def instantiate_inv_variables(self):
    return NotImplementedError

  def _num_towers(self):
    raise NotImplementedError

  def _get_data_device(self):
    raise NotImplementedError

  def register_matpower(self, exp, damping_func):
    raise NotImplementedError

  def register_cholesky(self, damping_func):
    raise NotImplementedError

  def register_cholesky_inverse(self, damping_func):
    raise NotImplementedError

  def get_matpower(self, exp, damping_func):
    raise NotImplementedError

  def get_cholesky(self, damping_func):
    raise NotImplementedError

  def get_cholesky_inverse(self, damping_func):
    raise NotImplementedError

  def get_cov_as_linear_operator(self):
    raise NotImplementedError


class DenseSquareMatrixFactorTestingDummy(ff.DenseSquareMatrixFactor):
  """Dummy class to test the non-abstract methods on ff.DenseSquareMatrixFactor.
  """

  def __init__(self, shape):
    self._shape = shape
    super(DenseSquareMatrixFactorTestingDummy, self).__init__()

  @property
  def _var_scope(self):
    return 'dummy/a_b_c'

  @property
  def _cov_shape(self):
    return self._shape

  @property
  def _num_sources(self):
    return 1

  @property
  def _dtype(self):
    return tf.float32

  def _compute_new_cov(self):
    raise NotImplementedError

  def instantiate_covariance(self):
    pass

  def _num_towers(self):
    raise NotImplementedError

  def _get_data_device(self):
    raise NotImplementedError


class NumericalUtilsTest(tf.test.TestCase):

  def testComputeCovAgainstNumpy(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)
      tf.set_random_seed(200)

      x = npr.randn(100, 3)
      cov = ff.compute_cov(tf.constant(x))
      np_cov = np.dot(x.T, x) / x.shape[0]

      self.assertAllClose(sess.run(cov), np_cov)

  def testComputeCovAgainstNumpyWithAlternativeNormalizer(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)
      tf.set_random_seed(200)

      normalizer = 10.
      x = npr.randn(100, 3)
      cov = ff.compute_cov(tf.constant(x), normalizer=normalizer)
      np_cov = np.dot(x.T, x) / normalizer

      self.assertAllClose(sess.run(cov), np_cov)

  def testAppendHomog(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      npr.seed(0)

      m, n = 3, 4
      a = npr.randn(m, n)
      a_homog = ff.append_homog(tf.constant(a))
      np_result = np.hstack([a, np.ones((m, 1))])

      self.assertAllClose(sess.run(a_homog), np_result)


class NameStringUtilFunctionTest(tf.test.TestCase):

  def _make_tensor(self):
    x = tf.placeholder(tf.float64, (3, 1))
    w = tf.constant(npr.RandomState(0).randn(3, 3))
    y = tf.matmul(w, x)
    g = tf.gradients(y, x)[0]
    return g

  def testScopeStringFromParamsSingleTensor(self):
    with tf.Graph().as_default():
      g = self._make_tensor()
      scope_string = ff.scope_string_from_params(g)
      self.assertEqual('gradients_MatMul_grad_MatMul_1_0', scope_string)

  def testScopeStringFromParamsMultipleTensors(self):
    with tf.Graph().as_default():
      x = tf.constant(1,)
      y = tf.constant(2,)
      scope_string = ff.scope_string_from_params((x, y))
      self.assertEqual('Const_0_Const_1_0', scope_string)

  def testScopeStringFromParamsMultipleTypes(self):
    with tf.Graph().as_default():
      x = tf.constant(1,)
      y = tf.constant(2,)
      scope_string = ff.scope_string_from_params([[1, 2, 3], 'foo', True, 4,
                                                  (x, y)])
      self.assertEqual('1-2-3_foo_True_4_Const_0__Const_1_0', scope_string)

  def testScopeStringFromParamsUnsupportedType(self):
    with tf.Graph().as_default():
      x = tf.constant(1,)
      y = tf.constant(2,)
      unsupported = 1.2  # Floats are not supported.
      with self.assertRaises(ValueError):
        ff.scope_string_from_params([[1, 2, 3], 'foo', True, 4, (x, y),
                                     unsupported])

  def testScopeStringFromName(self):
    with tf.Graph().as_default():
      g = self._make_tensor()
      scope_string = ff.scope_string_from_name(g)
      self.assertEqual('gradients_MatMul_grad_MatMul_1_0', scope_string)

  def testScalarOrTensorToString(self):
    with tf.Graph().as_default():
      self.assertEqual(ff.scalar_or_tensor_to_string(5.), repr(5.))

      g = self._make_tensor()
      scope_string = ff.scope_string_from_name(g)
      self.assertEqual(ff.scalar_or_tensor_to_string(g), scope_string)


class FisherFactorTest(tf.test.TestCase):

  def testMakeInverseUpdateOps(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      factor = FisherFactorTestingDummy()

      self.assertEqual(0, len(factor.make_inverse_update_ops()))


class InverseProvidingFactorTest(tf.test.TestCase):

  def testRegisterDampedInverse(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      shape = [2, 2]
      factor = DenseSquareMatrixFactorTestingDummy(shape)
      factor_var_scope = 'dummy/a_b_c'

      damping_funcs = [make_damping_func(0.1),
                       make_damping_func(0.1),
                       make_damping_func(1e-5),
                       make_damping_func(1e-5)]
      for damping_func in damping_funcs:
        factor.register_inverse(damping_func)

      factor.instantiate_inv_variables()

      inv = factor.get_inverse(damping_funcs[0]).to_dense().op.inputs[0]
      self.assertEqual(
          inv, factor.get_inverse(damping_funcs[1]).to_dense().op.inputs[0])
      self.assertNotEqual(
          inv, factor.get_inverse(damping_funcs[2]).to_dense().op.inputs[0])
      self.assertEqual(
          factor.get_inverse(damping_funcs[2]).to_dense().op.inputs[0],
          factor.get_inverse(damping_funcs[3]).to_dense().op.inputs[0])

      factor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      factor_var_scope)
      factor_tensors = (tf.convert_to_tensor(var).op.inputs[0]
                        for var in factor_vars)

      self.assertEqual(
          set([inv,
               factor.get_inverse(damping_funcs[2]).to_dense().op.inputs[0]]),
          set(factor_tensors))
      self.assertEqual(
          shape,
          factor.get_inverse(damping_funcs[0]).to_dense().get_shape())

  def testRegisterMatpower(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      shape = [3, 3]
      factor = DenseSquareMatrixFactorTestingDummy(shape)
      factor_var_scope = 'dummy/a_b_c'

      # TODO(b/74201126): Change to using the same func for both once
      # Topohash is in place.
      damping_func_1 = make_damping_func(0.5)
      damping_func_2 = make_damping_func(0.5)

      factor.register_matpower(-0.5, damping_func_1)
      factor.register_matpower(2, damping_func_2)

      factor.instantiate_inv_variables()

      factor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      factor_var_scope)

      factor_tensors = tuple(tf.convert_to_tensor(var).op.inputs[0]
                             for var in factor_vars)

      matpower1 = factor.get_matpower(-0.5, damping_func_1).to_dense()
      matpower2 = factor.get_matpower(2, damping_func_2).to_dense()

      self.assertEqual(set([matpower1.op.inputs[0], matpower2.op.inputs[0]]),
                       set(factor_tensors))

      self.assertEqual(shape, matpower1.get_shape())
      self.assertEqual(shape, matpower2.get_shape())

  def testMakeInverseUpdateOps(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      factor = FisherFactorTestingDummy()

      self.assertEqual(0, len(factor.make_inverse_update_ops()))

  def testMakeInverseUpdateOpsManyInversesEigenDecomp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      cov = np.array([[1., 2.], [3., 4.]])
      factor = DenseSquareMatrixFactorTestingDummy(cov.shape)
      factor._cov = tf.constant(cov, dtype=tf.float32)

      damping_funcs = []
      for i in range(1, ff.EIGENVALUE_DECOMPOSITION_THRESHOLD + 1):
        damping_funcs.append(make_damping_func(1./i))

      for i in range(ff.EIGENVALUE_DECOMPOSITION_THRESHOLD):
        factor.register_inverse(damping_funcs[i])

      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf.global_variables_initializer())
      new_invs = []
      sess.run(ops)
      for i in range(ff.EIGENVALUE_DECOMPOSITION_THRESHOLD):
        # The inverse op will assign the damped inverse of cov to the inv var.
        new_invs.append(
            sess.run(factor.get_inverse(damping_funcs[i]).to_dense()))

      # We want to see that the new invs are all different from each other.
      for i in range(len(new_invs)):
        for j in range(i + 1, len(new_invs)):
          # Just check the first element.
          self.assertNotEqual(new_invs[i][0][0], new_invs[j][0][0])

  def testMakeInverseUpdateOpsMatPowerEigenDecomp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      cov = np.array([[6., 2.], [2., 4.]])
      factor = DenseSquareMatrixFactorTestingDummy(cov.shape)
      factor._cov = tf.constant(cov, dtype=tf.float32)
      exp = 2  # NOTE(mattjj): must be int to test with np.linalg.matrix_power
      damping = 0.5
      damping_func = make_damping_func(damping)

      factor.register_matpower(exp, damping_func)
      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf.global_variables_initializer())
      sess.run(ops[0])
      matpower = sess.run(factor.get_matpower(exp, damping_func).to_dense())
      matpower_np = np.linalg.matrix_power(cov + np.eye(2) * damping, exp)
      self.assertAllClose(matpower, matpower_np)

  def testMakeInverseUpdateOpsNoEigenDecomp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      cov = np.array([[5., 2.], [2., 4.]])  # NOTE(mattjj): must be symmetric
      factor = DenseSquareMatrixFactorTestingDummy(cov.shape)
      factor._cov = tf.constant(cov, dtype=tf.float32)

      damping_func = make_damping_func(0)

      factor.register_inverse(damping_func)
      factor.instantiate_inv_variables()
      ops = factor.make_inverse_update_ops()
      self.assertEqual(1, len(ops))

      sess.run(tf.global_variables_initializer())
      # The inverse op will assign the damped inverse of cov to the inv var.
      old_inv = sess.run(factor.get_inverse(damping_func).to_dense())
      self.assertAllClose(
          sess.run(ff.inverse_initializer(cov.shape, tf.float32)), old_inv)

      sess.run(ops)
      new_inv = sess.run(factor.get_inverse(damping_func).to_dense())
      self.assertAllClose(new_inv, np.linalg.inv(cov))


class FullFactorTest(tf.test.TestCase):

  def testFullFactorInit(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), name='a/b/c')
      factor = ff.FullFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      self.assertEqual([6, 6], factor.cov.get_shape().as_list())

  def testFullFactorInitFloat64(self):
    with tf.Graph().as_default():
      dtype = dtypes.float64
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.FullFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([6, 6], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([1., 2.], name='a/b/c')
      factor = ff.FullFactor((tensor,), 2)
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[0.75, 0.5], [0.5, 1.5]], new_cov)


class NaiveDiagonalFactorTest(tf.test.TestCase):

  def testNaiveDiagonalFactorInit(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      self.assertEqual([6, 1], factor.cov.get_shape().as_list())

  def testNaiveDiagonalFactorInitFloat64(self):
    with tf.Graph().as_default():
      dtype = dtypes.float64
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 32)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([6, 1], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([1., 2.], name='a/b/c')
      factor = ff.NaiveDiagonalFactor((tensor,), 2)
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[0.75], [1.5]], new_cov)


class DiagonalKroneckerFactorTest(tf.test.TestCase):

  def testInitializationSparse(self):
    with tf.Graph().as_default():
      input_ids = tf.constant([[0], [1], [4]])
      vocab_size = 5
      factor = ff.DiagonalKroneckerFactor(((input_ids,),), vocab_size)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.shape.as_list(), [vocab_size])

  def testCovarianceUpdateOpSparse(self):
    with tf.Graph().as_default():
      input_ids = tf.constant([[0], [1], [4]])
      vocab_size = 5
      factor = ff.DiagonalKroneckerFactor(((input_ids,),), vocab_size)
      factor.instantiate_cov_variables()
      cov_update_op = factor.make_covariance_update_op(0.0)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(cov_update_op)
        new_cov = sess.run(factor.cov)
        self.assertAllClose(np.array([1., 1., 0., 0., 1.]) / 3., new_cov)

  def testInitializationDense(self):
    with tf.Graph().as_default():
      inputs = tf.constant([[0.0, 0.3, 0.4],
                            [1.0, 0.0, 0.0]])
      factor = ff.DiagonalKroneckerFactor(((inputs,),))
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.shape.as_list(), [3])

  def testCovarianceUpdateOpDense(self):
    with tf.Graph().as_default():
      inputs = tf.constant([[0.0, 0.3, 0.4],
                            [1.0, 0.0, 0.0],
                            [0.5, -1.0, 0.1]])
      factor = ff.DiagonalKroneckerFactor(((inputs,),))
      factor.instantiate_cov_variables()
      cov_update_op = factor.make_covariance_update_op(0.0)

      exact_cov = tf.matmul(inputs, inputs, transpose_a=True)
      diag_cov = tf.diag_part(exact_cov) / (
          tf.cast(tf.shape(inputs)[0], tf.float32))

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(cov_update_op)
        new_cov, expected = sess.run([factor.cov, diag_cov])
        self.assertAllClose(expected, new_cov)


class ConvDiagonalFactorTest(tf.test.TestCase):

  def setUp(self):
    self.batch_size = 10
    self.height = self.width = 32
    self.in_channels = 3
    self.out_channels = 1
    self.kernel_height = self.kernel_width = 3
    self.strides = [1, 2, 2, 1]
    self.data_format = 'NHWC'
    self.padding = 'SAME'
    self.kernel_shape = [
        self.kernel_height, self.kernel_width, self.in_channels,
        self.out_channels
    ]

  def testInit(self):
    with tf.Graph().as_default():
      inputs = tf.random_uniform(
          [self.batch_size, self.height, self.width, self.in_channels])
      outputs_grads = [
          tf.random_uniform([
              self.batch_size, self.height // self.strides[1],
              self.width // self.strides[2], self.out_channels
          ]) for _ in range(3)
      ]

      factor = ff.ConvDiagonalFactor(
          (inputs,),
          (outputs_grads,),
          self.kernel_shape,
          self.strides,
          self.padding,
          data_format=self.data_format)
      factor.instantiate_cov_variables()

      # Ensure covariance matrix's shape makes sense.
      self.assertEqual([
          self.kernel_height * self.kernel_width * self.in_channels,
          self.out_channels
      ],
                       factor.cov.shape.as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf.Graph().as_default():
      # Construct all arguments such that convolution kernel is applied in
      # exactly one spatial location.
      inputs = np.random.randn(
          1,  # batch_size
          self.kernel_height,
          self.kernel_width,
          self.in_channels)  # in_channels
      outputs_grad = np.random.randn(
          1,  # batch_size
          1,  # output_height
          1,  # output_width
          self.out_channels)

      factor = ff.ConvDiagonalFactor(
          (tf.constant(inputs),), ((tf.constant(outputs_grad),),),
          self.kernel_shape,
          strides=[1, 1, 1, 1],
          padding='VALID')
      factor.instantiate_cov_variables()

      # Completely forget initial value on first update.
      cov_update_op = factor.make_covariance_update_op(0.0)

      # Ensure new covariance value is same as outer-product of inputs/outputs
      # vectorized, squared.
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(cov_update_op)
        cov = sess.run(factor.cov)
        expected_cov = np.outer(inputs.flatten(), outputs_grad.flatten())**2
        self.assertAllClose(expected_cov, cov)

  def testHasBias(self):
    with tf.Graph().as_default():
      inputs = tf.random_uniform(
          [self.batch_size, self.height, self.width, self.in_channels])
      outputs_grads = [
          tf.random_uniform([
              self.batch_size, self.height // self.strides[1],
              self.width // self.strides[2], self.out_channels
          ]) for _ in range(3)
      ]

      factor = ff.ConvDiagonalFactor(
          (inputs,),
          (outputs_grads,),
          self.kernel_shape,
          self.strides,
          self.padding,
          data_format=self.data_format,
          has_bias=True)
      factor.instantiate_cov_variables()

      # Ensure shape accounts for bias.
      self.assertEqual([
          self.kernel_height * self.kernel_width * self.in_channels + 1,
          self.out_channels
      ],
                       factor.cov.shape.as_list())

      # Ensure update op doesn't crash.
      cov_update_op = factor.make_covariance_update_op(0.0)
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(cov_update_op)


class FullyConnectedKroneckerFactorTest(tf.test.TestCase):

  def _testFullyConnectedKroneckerFactorInit(self,
                                             has_bias,
                                             final_shape,
                                             dtype=dtypes.float32):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor(((tensor,),), has_bias=has_bias)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual(final_shape, cov.get_shape().as_list())

  def testFullyConnectedKroneckerFactorInitNoBias(self):
    for dtype in (dtypes.float32, dtypes.float64):
      self._testFullyConnectedKroneckerFactorInit(False, [3, 3], dtype=dtype)

  def testFullyConnectedKroneckerFactorInitWithBias(self):
    for dtype in (dtypes.float32, dtypes.float64):
      self._testFullyConnectedKroneckerFactorInit(True, [4, 4], dtype=dtype)

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor(((tensor,),), has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[3, 3.5, 1], [3.5, 5.5, 1.5], [1, 1.5, 1]], new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedKroneckerFactor(((tensor,),))
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[3, 3.5], [3.5, 5.5]], new_cov)


class ConvFactorTestCase(tf.test.TestCase):

  def assertMatrixRank(self, rank, matrix, atol=1e-5):
    assert rank <= matrix.shape[0], 'Rank cannot be larger than matrix size.'
    eigvals = np.linalg.eigvals(matrix)
    nnz_eigvals = np.sum(eigvals > atol)
    self.assertEqual(
        rank,
        nnz_eigvals,
        msg=('Found %d of %d expected non-zero eigenvalues: %s.' %
             (nnz_eigvals, rank, eigvals)))


class ConvInputKroneckerFactorTest(ConvFactorTestCase):

  def test3DConvolution(self):
    with tf.Graph().as_default():
      batch_size = 1
      width = 3
      in_channels = 3**3
      out_channels = 4

      factor = ff.ConvInputKroneckerFactor(
          inputs=(tf.random_uniform(
              (batch_size, width, width, width, in_channels), seed=0),),
          filter_shape=(width, width, width, in_channels, out_channels),
          padding='SAME',
          strides=(2, 2, 2),
          extract_patches_fn='extract_convolution_patches',
          has_bias=False)
      factor.instantiate_cov_variables()

      # Ensure shape of covariance matches input size of filter.
      input_size = in_channels * (width**3)
      self.assertEqual([input_size, input_size],
                       factor.cov.shape.as_list())

      # Ensure cov_update_op doesn't crash.
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(factor.make_covariance_update_op(0.0))
        cov = sess.run(factor.cov)

      # Cov should be rank-8, as the filter will be applied at each corner of
      # the 4-D cube.
      self.assertMatrixRank(8, cov)

  def testPointwiseConv2d(self):
    with tf.Graph().as_default():
      batch_size = 1
      width = 3
      in_channels = 3**2
      out_channels = 4

      factor = ff.ConvInputKroneckerFactor(
          inputs=(tf.random_uniform(
              (batch_size, width, width, in_channels), seed=0),),
          filter_shape=(1, 1, in_channels, out_channels),
          padding='SAME',
          strides=(1, 1, 1, 1),
          extract_patches_fn='extract_pointwise_conv2d_patches',
          has_bias=False)
      factor.instantiate_cov_variables()

      # Ensure shape of covariance matches input size of filter.
      self.assertEqual([in_channels, in_channels],
                       factor.cov.shape.as_list())

      # Ensure cov_update_op doesn't crash.
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(factor.make_covariance_update_op(0.0))
        cov = sess.run(factor.cov)

      # Cov should be rank-9, as the filter will be applied at each location.
      self.assertMatrixRank(9, cov)

  def testStrides(self):
    with tf.Graph().as_default():
      batch_size = 1
      width = 3
      in_channels = 3**2
      out_channels = 4

      factor = ff.ConvInputKroneckerFactor(
          inputs=(tf.random_uniform(
              (batch_size, width, width, in_channels), seed=0),),
          filter_shape=(1, 1, in_channels, out_channels),
          padding='SAME',
          strides=(1, 2, 1, 1),
          extract_patches_fn='extract_image_patches',
          has_bias=False)
      factor.instantiate_cov_variables()

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(factor.make_covariance_update_op(0.0))
        cov = sess.run(factor.cov)

      # Cov should be the sum of 3 * 2 = 6 outer products.
      self.assertMatrixRank(6, cov)

  def testDilationRate(self):
    with tf.Graph().as_default():
      batch_size = 1
      width = 3
      in_channels = 2
      out_channels = 4

      factor = ff.ConvInputKroneckerFactor(
          inputs=(tf.random_uniform(
              (batch_size, width, width, in_channels), seed=0),),
          filter_shape=(3, 3, in_channels, out_channels),
          padding='SAME',
          extract_patches_fn='extract_image_patches',
          strides=(1, 1, 1, 1),
          dilation_rate=(1, width, width, 1),
          has_bias=False)
      factor.instantiate_cov_variables()

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(factor.make_covariance_update_op(0.0))
        cov = sess.run(factor.cov)

      # Cov should be rank = in_channels, as only the center of the filter
      # receives non-zero input for each input channel.
      self.assertMatrixRank(in_channels, cov)

  def testConvInputKroneckerFactorInitNoBias(self):
    with tf.Graph().as_default():
      tensor = tf.ones((64, 1, 2, 3), name='a/b/c')
      factor = ff.ConvInputKroneckerFactor(
          inputs=(tensor,),
          filter_shape=(1, 2, 3, 4),
          padding='SAME',
          has_bias=False)
      factor.instantiate_cov_variables()
      self.assertEqual([1 * 2 * 3, 1 * 2 * 3],
                       factor.cov.get_shape().as_list())

  def testConvInputKroneckerFactorInit(self):
    with tf.Graph().as_default():
      tensor = tf.ones((64, 1, 2, 3), name='a/b/c')
      factor = ff.ConvInputKroneckerFactor(
          (tensor,), filter_shape=(1, 2, 3, 4), padding='SAME', has_bias=True)
      factor.instantiate_cov_variables()
      self.assertEqual([1 * 2 * 3 + 1, 1 * 2 * 3 + 1],
                       factor.cov.get_shape().as_list())

  def testConvInputKroneckerFactorInitFloat64(self):
    with tf.Graph().as_default():
      dtype = dtypes.float64
      tensor = tf.ones((64, 1, 2, 3), name='a/b/c', dtype=tf.float64)
      factor = ff.ConvInputKroneckerFactor(
          (tensor,), filter_shape=(1, 2, 3, 4), padding='SAME', has_bias=True)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([1 * 2 * 3 + 1, 1 * 2 * 3 + 1],
                       cov.get_shape().as_list())

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      input_shape = (2, 1, 1, 1)
      tensor = tf.constant(
          np.arange(1, 1 + np.prod(input_shape)).reshape(input_shape).astype(
              np.float32))
      factor = ff.ConvInputKroneckerFactor(
          (tensor,), filter_shape=(1, 1, 1, 1), padding='SAME', has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(0.))
      new_cov = sess.run(factor.cov)
      self.assertAllClose(
          [
              [(1. + 4.) / 2., (1. + 2.) / 2.],  #
              [(1. + 2.) / 2., (1. + 1.) / 2.]
          ],  #
          new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      input_shape = (2, 1, 1, 1)
      tensor = tf.constant(
          np.arange(1, 1 + np.prod(input_shape)).reshape(input_shape).astype(
              np.float32))
      factor = ff.ConvInputKroneckerFactor(
          (tensor,), filter_shape=(1, 1, 1, 1), padding='SAME')
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(0.))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[(1. + 4.) / 2.]], new_cov)

  def testSubSample(self):
    with tf.Graph().as_default():
      patches_1 = tf.constant(1, shape=(10, 2))
      patches_2 = tf.constant(1, shape=(10, 8))
      patches_3 = tf.constant(1, shape=(3, 3))
      patches_1_sub = ff._subsample_patches(patches_1)
      patches_2_sub = ff._subsample_patches(patches_2)
      patches_3_sub = ff._subsample_patches(patches_3)
      patches_1_sub_batch_size = patches_1_sub.shape.as_list()[0]
      patches_2_sub_batch_size = patches_2_sub.shape.as_list()[0]
      patches_3_sub_batch_size = patches_3_sub.shape.as_list()[0]
      self.assertEqual(2, patches_1_sub_batch_size)
      self.assertEqual(8, patches_2_sub_batch_size)
      self.assertEqual(3, patches_3_sub_batch_size)


class ConvOutputKroneckerFactorTest(ConvFactorTestCase):

  def test3DConvolution(self):
    with tf.Graph().as_default():
      batch_size = 1
      width = 3
      out_channels = width**3

      factor = ff.ConvOutputKroneckerFactor(outputs_grads=([
          tf.random_uniform(
              (batch_size, width, width, width, out_channels), seed=0)
      ],))
      factor.instantiate_cov_variables()

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(factor.make_covariance_update_op(0.0))
        cov = sess.run(factor.cov)

      # Cov should be rank 3^3, as each spatial position donates a rank-1
      # update.
      self.assertMatrixRank(width**3, cov)

  def testConvOutputKroneckerFactorInit(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3, 4, 5), name='a/b/c')
      factor = ff.ConvOutputKroneckerFactor(((tensor,),))
      factor.instantiate_cov_variables()
      self.assertEqual([5, 5], factor.cov.get_shape().as_list())

  def testConvOutputKroneckerFactorInitFloat64(self):
    with tf.Graph().as_default():
      dtype = dtypes.float64
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3, 4, 5), dtype=dtype, name='a/b/c')
      factor = ff.ConvOutputKroneckerFactor(((tensor,),))
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([5, 5], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOp(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = np.arange(1, 17).reshape(2, 2, 2, 2).astype(np.float32)
      factor = ff.ConvOutputKroneckerFactor(((tf.constant(tensor),),))
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[43, 46.5], [46.5, 51.5]], new_cov)


class FullyConnectedMultiKFTest(tf.test.TestCase):

  def testFullyConnectedMultiKFInit(self):
    with tf.Graph().as_default():
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), name='a/b/c')
      factor = ff.FullyConnectedMultiKF(((tensor,),), has_bias=False)
      factor.instantiate_cov_variables()
      self.assertEqual([3, 3], factor.cov.get_shape().as_list())

  def testFullyConnectedMultiKFInitFloat64(self):
    with tf.Graph().as_default():
      dtype = dtypes.float64
      tf.set_random_seed(200)
      tensor = tf.ones((2, 3), dtype=dtype, name='a/b/c')
      factor = ff.FullyConnectedMultiKF(((tensor,),), has_bias=False)
      factor.instantiate_cov_variables()
      cov = factor.cov
      self.assertEqual(cov.dtype, dtype)
      self.assertEqual([3, 3], cov.get_shape().as_list())

  def testMakeCovarianceUpdateOpWithBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedMultiKF(((tensor,),), has_bias=True)
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[3, 3.5, 1], [3.5, 5.5, 1.5], [1, 1.5, 1]], new_cov)

  def testMakeCovarianceUpdateOpNoBias(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      tf.set_random_seed(200)
      tensor = tf.constant([[1., 2.], [3., 4.]], name='a/b/c')
      factor = ff.FullyConnectedMultiKF(((tensor,),))
      factor.instantiate_cov_variables()

      sess.run(tf.global_variables_initializer())
      sess.run(factor.make_covariance_update_op(.5))
      new_cov = sess.run(factor.cov)
      self.assertAllClose([[3, 3.5], [3.5, 5.5]], new_cov)


class ConvInputSUAKroneckerFactorTest(ConvFactorTestCase):

  def setUp(self):
    tf.reset_default_graph()
    batch_size = 10
    in_dim = 3
    kernel_width = 2
    self.in_channels = 2
    out_channels = 1
    self.damping_val = 0.03
    self.kw_kh = kernel_width**2
    self.filter_shape = (kernel_width, kernel_width, self.in_channels,
                         out_channels)

    self.inputs = tf.random_uniform(
        (batch_size, in_dim, in_dim, self.in_channels), seed=0)
    damping_id = ('test', 'cov')

    def compute_damping():
      return (self.damping_val, self.damping_val)

    self.damping_func = fb._package_func(lambda: compute_damping()[0],
                                         damping_id + ('ref', 0))

  def testConv2dOpsNoBiasZeroMean(self):
    ff.set_global_constants(assume_zero_mean_activations=True)
    factor = ff.ConvInputSUAKroneckerFactor(
        inputs=(self.inputs,),
        filter_shape=self.filter_shape,
        has_bias=False)
    factor.instantiate_cov_variables()
    factor.register_matpower(-1., self.damping_func)

    factor.instantiate_inv_variables()
    self.assertEqual([self.in_channels, self.in_channels],
                     factor.cov.shape.as_list())
    cov_update_op = factor.make_covariance_update_op(0.0)
    inv_update_op = factor.make_inverse_update_ops()
    inv_matrix = factor.get_inv_vars()[0]

    fisher_inv = factor.get_matpower(-1., self.damping_func).to_dense()
    inv_dim = self.kw_kh * self.in_channels
    self.assertEqual([inv_dim, inv_dim], fisher_inv.shape.as_list())

    cov_damping = factor.get_matpower(1, self.damping_func).to_dense()
    cov_trace = factor.get_cov_as_linear_operator().trace()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, inputs_ = sess.run([cov_update_op, self.inputs])
      sess.run([inv_update_op])
      cov_ = sess.run(factor.cov)
      self.assertMatrixRank(self.in_channels, cov_)

      inputs_ = np.reshape(inputs_, (-1, self.in_channels))
      mean_ = (1. / inputs_.shape[0]) * np.sum(inputs_, axis=0, keepdims=True)
      expected_cov_ = (1. / (inputs_.shape[0])) * np.matmul(
          inputs_.transpose(), inputs_) - np.matmul(mean_.transpose(), mean_)
      self.assertAllClose(expected_cov_, cov_)

      expected_cov_damping_ = expected_cov_ + self.damping_val * np.eye(
          self.in_channels)
      expected_inv_matrix_ = np.linalg.inv(expected_cov_damping_)
      inv_matrix_ = sess.run(inv_matrix)
      self.assertAllClose(expected_inv_matrix_, inv_matrix_, rtol=1e-5)

      quant_1 = np.kron(expected_cov_damping_, np.eye(self.kw_kh))
      fisher_inv_ = sess.run(fisher_inv)
      expected_fisher_inv_ = np.linalg.inv(quant_1)
      self.assertAllClose(fisher_inv_, expected_fisher_inv_, rtol=1e-5)

      expected_cov_trace_ = np.kron(expected_cov_, np.eye(self.kw_kh)).trace()
      cov_trace_ = sess.run(cov_trace)
      self.assertAlmostEqual(cov_trace_, expected_cov_trace_, places=5)

      cov_damping_ = sess.run(cov_damping)
      self.assertAllClose(quant_1, cov_damping_)
      expected_id_ = np.matmul(cov_damping_, fisher_inv_)
      self.assertAllClose(expected_id_, np.eye(expected_id_.shape[0]))

  def testConv2dOpsBiasZeroMean(self):
    ff.set_global_constants(assume_zero_mean_activations=True)
    factor = ff.ConvInputSUAKroneckerFactor(
        inputs=(self.inputs,),
        filter_shape=self.filter_shape,
        has_bias=True)

    factor.instantiate_cov_variables()
    factor.register_matpower(-1., self.damping_func)

    factor.instantiate_inv_variables()
    self.assertEqual([self.in_channels, self.in_channels],
                     factor.cov.shape.as_list())

    cov_update_op = factor.make_covariance_update_op(0.0)
    inv_update_op = factor.make_inverse_update_ops()
    inv_matrix = factor.get_inv_vars()[0]

    fisher_inv = factor.get_matpower(-1., self.damping_func).to_dense()
    inv_dim = self.kw_kh * self.in_channels + 1
    self.assertEqual([inv_dim, inv_dim], fisher_inv.shape.as_list())

    cov_trace = factor.get_cov_as_linear_operator().trace()
    cov_damping = factor.get_matpower(1, self.damping_func).to_dense()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, inputs_ = sess.run([cov_update_op, self.inputs])
      sess.run([inv_update_op])
      cov_ = sess.run(factor.cov)
      self.assertMatrixRank(self.in_channels, cov_)

      inputs_ = np.reshape(inputs_, (-1, self.in_channels))
      mean_ = (1. / inputs_.shape[0]) * np.sum(inputs_, axis=0, keepdims=True)
      expected_cov_ = (1. / (inputs_.shape[0])) * np.matmul(
          inputs_.transpose(), inputs_) - np.matmul(mean_.transpose(), mean_)
      self.assertAllClose(expected_cov_, cov_)

      expected_cov_damping_ = expected_cov_ + self.damping_val * np.eye(
          self.in_channels)
      expected_inv_matrix_ = np.linalg.inv(expected_cov_damping_)
      inv_matrix_ = sess.run(inv_matrix)
      self.assertAllClose(expected_inv_matrix_, inv_matrix_, rtol=1e-5)

      quant_0 = np.kron(expected_cov_damping_, np.eye(self.kw_kh))
      quant_1 = np.zeros((inv_dim, inv_dim))
      quant_1[:-1, :-1] = quant_0
      quant_1[-1, -1] = self.damping_val

      fisher_inv_ = sess.run(fisher_inv)
      expected_fisher_inv_ = np.linalg.inv(quant_1)
      self.assertAllClose(fisher_inv_, expected_fisher_inv_, rtol=1e-5)

      expected_cov_trace_ = np.kron(expected_cov_, np.eye(self.kw_kh),).trace()
      cov_trace_ = sess.run(cov_trace)
      self.assertAlmostEqual(cov_trace_, expected_cov_trace_, places=5)

      cov_damping_ = sess.run(cov_damping)
      self.assertAllClose(quant_1, cov_damping_)
      expected_id_ = np.matmul(cov_damping_, fisher_inv_)
      self.assertAllClose(expected_id_, np.eye(expected_id_.shape[0]))

  def testConv2OpsNoBias(self):
    ff.set_global_constants(assume_zero_mean_activations=False)
    factor = ff.ConvInputSUAKroneckerFactor(
        inputs=(self.inputs,),
        filter_shape=self.filter_shape,
        has_bias=False)

    factor.instantiate_cov_variables()
    factor.register_matpower(-1., self.damping_func)

    factor.instantiate_inv_variables()
    self.assertEqual([self.in_channels, self.in_channels],
                     factor.cov.shape.as_list())

    cov_update_op = factor.make_covariance_update_op(0.0)
    inv_update_op = factor.make_inverse_update_ops()
    inv_matrix = factor.get_inv_vars()[0]

    fisher_inv_operator = factor.get_matpower(-1., self.damping_func)
    fisher_inv = fisher_inv_operator.to_dense()
    inv_dim = self.kw_kh * self.in_channels
    self.assertEqual([inv_dim, inv_dim], fisher_inv.shape.as_list())

    input_tensor = tf.random_uniform(shape=(inv_dim, 1), maxval=10.)
    output_tensor = fisher_inv_operator.matmul(input_tensor)

    cov_trace = factor.get_cov_as_linear_operator().trace()
    cov_damping = factor.get_matpower(1, self.damping_func).to_dense()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, inputs_ = sess.run([cov_update_op, self.inputs])
      sess.run([inv_update_op])
      cov_ = sess.run(factor.cov)
      self.assertMatrixRank(self.in_channels, cov_)

      inputs_ = np.reshape(inputs_, (-1, self.in_channels))
      mean_ = (1. / inputs_.shape[0]) * np.sum(inputs_, axis=0, keepdims=True)
      expected_cov_ = (1. / (inputs_.shape[0])) * np.matmul(
          inputs_.transpose(), inputs_) - np.matmul(mean_.transpose(), mean_)
      self.assertAllClose(expected_cov_, cov_)

      expected_cov_damping_ = expected_cov_ + self.damping_val * np.eye(
          self.in_channels)
      expected_inv_matrix_ = np.linalg.inv(expected_cov_damping_)
      inv_matrix_ = sess.run(inv_matrix)

      self.assertAllClose(expected_inv_matrix_, inv_matrix_, rtol=1e-5)

      quant_1 = np.kron(expected_cov_damping_, np.eye(self.kw_kh))
      quant_2 = np.kron((1. / inputs_.shape[0]) * np.sum(inputs_, axis=0),
                        np.ones(self.kw_kh))
      quant_3 = np.expand_dims(quant_2, axis=1)

      fisher_inv_ = sess.run(fisher_inv)
      fisher_ = quant_1 + np.matmul(quant_3, quant_3.transpose())
      expected_fisher_inv_ = np.linalg.inv(fisher_)
      self.assertAllClose(fisher_inv_, expected_fisher_inv_, rtol=1e-5)

      input_tensor_, output_tensor_ = sess.run([input_tensor, output_tensor])
      expected_output_tensor_ = np.matmul(expected_fisher_inv_, input_tensor_)
      self.assertAllClose(output_tensor_, expected_output_tensor_, rtol=1e-5,
                          atol=5e-5)

      expected_cov_trace_ = np.kron(expected_cov_, np.eye(
          self.kw_kh)).trace() + np.dot(quant_2, quant_2)
      cov_trace_ = sess.run(cov_trace)
      self.assertAlmostEqual(cov_trace_, expected_cov_trace_, places=5)

      cov_damping_ = sess.run(cov_damping)
      self.assertAllClose(fisher_, cov_damping_)
      expected_id_ = np.matmul(cov_damping_, fisher_inv_)
      self.assertAllClose(expected_id_, np.eye(expected_id_.shape[0]))

  def testConv2dOpsBias(self):
    ff.set_global_constants(assume_zero_mean_activations=False)
    factor = ff.ConvInputSUAKroneckerFactor(
        inputs=(self.inputs,),
        filter_shape=self.filter_shape,
        has_bias=True)

    factor.instantiate_cov_variables()
    factor.register_matpower(-1., self.damping_func)

    factor.instantiate_inv_variables()
    self.assertEqual([self.in_channels, self.in_channels],
                     factor.cov.shape.as_list())

    cov_update_op = factor.make_covariance_update_op(0.0)
    inv_update_op = factor.make_inverse_update_ops()
    inv_matrix = factor.get_inv_vars()[0]

    fisher_inv = factor.get_matpower(-1., self.damping_func).to_dense()
    inv_dim = self.kw_kh * self.in_channels + 1
    self.assertEqual([inv_dim, inv_dim], fisher_inv.shape.as_list())

    cov_trace = factor.get_cov_as_linear_operator().trace()
    cov_damping = factor.get_matpower(1, self.damping_func).to_dense()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _, inputs_ = sess.run([cov_update_op, self.inputs])
      sess.run([inv_update_op])
      cov_ = sess.run(factor.cov)
      self.assertMatrixRank(self.in_channels, cov_)

      inputs_ = np.reshape(inputs_, (-1, self.in_channels))
      mean_ = (1. / inputs_.shape[0]) * np.sum(inputs_, axis=0, keepdims=True)
      expected_cov_ = (1. / (inputs_.shape[0])) * np.matmul(
          inputs_.transpose(), inputs_) - np.matmul(mean_.transpose(), mean_)
      self.assertAllClose(expected_cov_, cov_)

      expected_cov_damping_ = expected_cov_ + self.damping_val * np.eye(
          self.in_channels)
      expected_inv_matrix_ = np.linalg.inv(expected_cov_damping_)
      inv_matrix_ = sess.run(inv_matrix)
      self.assertAllClose(expected_inv_matrix_, inv_matrix_, rtol=1e-5)

      quant_0 = np.kron(expected_cov_damping_, np.eye(self.kw_kh))
      quant_1 = np.zeros((inv_dim, inv_dim))
      quant_1[:-1, :-1] = quant_0
      quant_1[-1, -1] = self.damping_val

      quant_2 = np.kron((1. / inputs_.shape[0]) * np.sum(inputs_, axis=0),
                        np.ones(self.kw_kh))
      quant_3 = np.ones((inv_dim,))
      quant_3[:-1] = quant_2
      quant_3 = np.expand_dims(quant_3, axis=1)

      fisher_inv_ = sess.run(fisher_inv)
      fisher_ = quant_1 + np.matmul(quant_3, quant_3.transpose())
      expected_fisher_inv_ = np.linalg.inv(fisher_)
      self.assertAllClose(fisher_inv_, expected_fisher_inv_, rtol=1e-5)

      expected_cov_trace_ = np.kron(expected_cov_, np.eye(
          self.kw_kh)).trace() + np.dot(quant_2, quant_2) + 1.
      cov_trace_ = sess.run(cov_trace)
      self.assertAlmostEqual(cov_trace_, expected_cov_trace_, places=5)

      cov_damping_ = sess.run(cov_damping)
      self.assertAllClose(fisher_, cov_damping_)
      expected_id_ = np.matmul(cov_damping_, fisher_inv_)
      self.assertAllClose(
          expected_id_, np.eye(expected_id_.shape[0]), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  tf.test.main()
