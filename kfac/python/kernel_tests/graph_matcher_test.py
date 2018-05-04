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
"""Tests for tf.contrib.kfac.tensormatch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kfac.python.ops.tensormatch import graph_matcher as gm
from kfac.python.ops.tensormatch import graph_patterns as gp


class TestCase(tf.test.TestCase):

  def assertDictEqualIdenticalValues(self, d1, d2, msg=None):
    """Like assertDictEqual but compares values with 'is' rather than '=='."""
    self.assertDictEqual(d1, d2, msg)
    for k in d1:
      self.assertIs(d1[k], d2[k], msg)

  def assertDictContainsSubsetIdenticalValues(self, expected, actual, msg=None):
    self.assertIsInstance(actual, dict)
    actual_subset = {k: actual[k] for k in expected}
    return self.assertDictEqualIdenticalValues(expected, actual_subset, msg)


class BasicMatcherTests(TestCase):

  def test_basic_tensor(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      y = tf.matmul(x, w)

      tensor = gp.Tensor

      self.assertIs(gm.matcher(tensor)(x), True)
      self.assertIs(gm.matcher(tensor)(w), True)
      self.assertIs(gm.matcher(tensor)(y), True)

      self.assertDictEqualIdenticalValues(gm.matcher(tensor('a'))(x), {'a': x})
      self.assertDictEqualIdenticalValues(gm.matcher(tensor('a'))(w), {'a': w})
      self.assertDictEqualIdenticalValues(gm.matcher(tensor('a'))(y), {'a': y})

  def test_basic_placeholder(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      y = tf.matmul(x, w)

      placeholder = gp.Placeholder

      self.assertIs(gm.matcher(placeholder)(x), True)
      self.assertIs(gm.matcher(placeholder)(y), False)
      self.assertIs(gm.matcher(placeholder)(w), False)

      self.assertDictEqualIdenticalValues(
          gm.matcher(placeholder('a'))(x), {'a': x})
      self.assertIs(gm.matcher(placeholder('a'))(w), False)
      self.assertIs(gm.matcher(placeholder('a'))(y), False)

  def test_basic_variable(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      y = tf.matmul(x, w)

      variable = gp.Variable

      self.assertIs(gm.matcher(variable)(x), False)
      self.assertIs(gm.matcher(variable)(w), True)
      self.assertIs(gm.matcher(variable)(y), False)

      self.assertIs(gm.matcher(variable('a'))(x), False)
      self.assertDictEqualIdenticalValues(
          gm.matcher(variable('a'))(w), {'a': w})
      self.assertIs(gm.matcher(variable('a'))(y), False)

  def test_basic_op(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      y = tf.matmul(x, w)

      tensor, matmul, op = gp.Tensor, gp.MatMul, gp.Op

      self.assertIs(gm.matcher(tensor)(y.op), False)
      self.assertIs(gm.matcher(matmul)(y.op), True)
      self.assertIs(gm.matcher(op)(y.op), True)

      self.assertDictEqualIdenticalValues(
          gm.matcher(op('a'))(y.op), {'a': y.op})

  def test_basic_list(self):
    with tf.Graph().as_default():
      w = tf.Variable(tf.random_normal((3, 3)))
      tensors = tf.unstack(w, axis=1)

      tensor = gp.Tensor

      pat = ('List', tensor('a'), tensor('b'), tensor('c'))
      bindings = gm.matcher(pat)(tensors)

      self.assertDictEqualIdenticalValues(bindings, {
          'a': tensors[0],
          'b': tensors[1],
          'c': tensors[2]
      })

      pat = ('List', tensor('a'), tensor('b'), tensor('a'))
      bindings = gm.matcher(pat)(tensors)
      self.assertIs(bindings, False)


class CompoundPatternTests(TestCase):

  def test_basic_compound_pattern_inputs(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      z = tf.placeholder('float32', (10, 3))
      y = x + z

      tensor, add = gp.Tensor, gp.Add

      match = gm.matcher((tensor, ('In', add)))
      self.assertIs(match(y), True)

      match = gm.matcher((tensor, ('In', (add, ('In', tensor, tensor)))))
      self.assertIs(match(y), True)

      match = gm.matcher((tensor('y'), ('In', (add, ('In', tensor('x'),
                                                     tensor('z'))))))
      bindings = match(y)
      self.assertDictEqualIdenticalValues(bindings, {'y': y, 'x': x, 'z': z})

  def test_basic_compound_pattern_outputs(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      z = tf.placeholder('float32', (10, 3))
      y = x + z

      tensor, add = gp.Tensor, gp.Add

      match = gm.matcher((tensor, ('Out', add)))
      self.assertIs(match(x), True)

      match = gm.matcher((tensor, ('Out', (add, ('Out', tensor)))))
      self.assertIs(match(x), True)

      match = gm.matcher((tensor('x'), ('Out', (add, ('Out', tensor('y'))))))
      bindings = match(x)
      self.assertDictEqualIdenticalValues(bindings, {'x': x, 'y': y})

  def test_basic_compound_pattern_inputs_and_outputs(self):
    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, (10, 3))
      z = tf.placeholder(tf.float32, (10, 3))
      y = x + z

      tensor, add = gp.Tensor, gp.Add

      match = gm.matcher((tensor, ('Out', (add, ('In', tensor, tensor),
                                           ('Out', tensor)))))
      self.assertIs(match(x), True)

      match = gm.matcher((tensor('x'), ('Out', (add, ('In', tensor('x'),
                                                      tensor('z')),
                                                ('Out', tensor('y'))))))
      bindings = match(x)
      self.assertDictEqualIdenticalValues(bindings, {'y': y, 'x': x, 'z': z})

  def test_layer_require_single_consumer_of_result_tensor(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))

      # match a subexpresion of the form x*w+b that is used in exactly one op
      tensor, variable, add, op = gp.Tensor, gp.Variable, gp.Add, gp.Op
      pat = (tensor, ('In', (add, ('In', tensor, variable))),
             ('Out', op))  # this line only allows a single consumer
      match = gm.matcher(pat)

      s = tf.matmul(x, w) + b
      out = tf.tanh(s)  # pylint: disable=unused-variable
      self.assertIs(match(s), True)

      out2 = tf.nn.relu(s)  # pylint: disable=unused-variable
      self.assertIs(match(s), False)

  def test_layer_affine(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))

      match = gm.matcher(gp.Affine)

      self.assertTrue(match(tf.matmul(x, w)))
      self.assertTrue(match(tf.matmul(x, w) + b))
      self.assertTrue(match(b + tf.matmul(x, w)))
      self.assertFalse(match(b + x**2))
      self.assertFalse(match(b + tf.matmul(x, w) + b))

  def test_repeated_names(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      z = tf.placeholder('float32', (10, 3))

      tensor, add = gp.Tensor, gp.Add

      match = gm.matcher((tensor, ('In', (add, ('In', tensor('a'),
                                                tensor('a'))))))

      self.assertIs(match(x + z), False)
      self.assertDictEqualIdenticalValues(match(x + x), {'a': x})

  def test_multiple_names(self):
    with tf.Graph().as_default():
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))
      x = tf.placeholder('float32', (10, 3))
      y = tf.matmul(x, w) + b

      tensor, variable, add, matmul = gp.Tensor, gp.Variable, gp.Add, gp.MatMul

      pat = (tensor, ('In', (add('add'), ('In', tensor, variable('bias')))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {
          'add': y.op,
          'bias': b._ref()
      })

      pat = (tensor, ('In', (add, ('In', (tensor, ('In',
                                                   (matmul,
                                                    ('In', tensor,
                                                     tensor('weights'))))),
                                   variable('bias')))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {
          'weights': w._ref(),
          'bias': b._ref()
      })

  def test_compound_list(self):
    with tf.Graph().as_default():
      w = tf.Variable(tf.random_normal((3, 3)))
      tensors = tf.unstack(w, axis=1)

      tensor, unstack = gp.Tensor, gp.Unstack

      pat = (('List', tensor('a'), tensor('b'), tensor('c')), ('In', unstack))
      bindings = gm.matcher(pat)(tensors)
      self.assertDictEqualIdenticalValues(bindings, {
          'a': tensors[0],
          'b': tensors[1],
          'c': tensors[2]
      })

      pat = (('List', tensor('a'), tensor('a'), tensor('a')), ('In', unstack))
      bindings = gm.matcher(pat)(tensors)
      self.assertIs(bindings, False)


class ChoicePatternTests(TestCase):

  def test_basic_choice(self):
    with tf.Graph().as_default():
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))
      x = tf.placeholder('float32', (10, 3))
      y = tf.matmul(x, w) + b

      tensor, add, matmul = gp.Tensor, gp.Add, gp.MatMul

      pat = (tensor, ('In', ('?:choice', add('a'), matmul('a'))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {'a': y.op})

      pat = (tensor, ('In', (('?:choice', add('a'), matmul('a')), ('In', tensor,
                                                                   tensor))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {'a': y.op})

  def test_multiple_choice(self):
    with tf.Graph().as_default():
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))
      x = tf.placeholder('float32', (10, 3))
      y = tf.matmul(x, w) + b

      tensor, add, matmul = gp.Tensor, gp.Add, gp.MatMul

      pat = (tensor, ('In', (('?:choice', add('a'), matmul('a')),
                             ('In', (tensor, ('In', ('?:choice', add('b'),
                                                     matmul('b')))), tensor))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {
          'a': y.op,
          'b': y.op.inputs[0].op
      })

      pat = (tensor, ('In', (('?:choice', add('a'), matmul('a')),
                             ('In', (tensor, ('In', ('?:choice', add('b'),
                                                     matmul('b')))), tensor))))
      bindings = gm.matcher(pat)(y)
      self.assertDictEqualIdenticalValues(bindings, {
          'a': y.op,
          'b': y.op.inputs[0].op
      })


class NotPatternTests(TestCase):

  def test_not(self):
    x = tf.placeholder('float32', (10, 3))

    tensor, add, matmul = gp.Tensor, gp.Add, gp.MatMul

    pat = (tensor, ('In', (('?:not', matmul), ('In', tensor, tensor))))
    self.assertIs(gm.matcher(pat)(x + x), True)

    pat = (tensor, ('In', (('?:not', add), ('In', tensor, tensor))))
    self.assertIs(gm.matcher(pat)(x + x), False)


class AllMatcherTests(TestCase):

  def test_all_matcher(self):
    x = tf.placeholder('float32', (10, 3))
    tensor, placeholder, add = gp.Tensor, gp.Placeholder, gp.Add

    pat = (tensor, ('In', (add, ('In', tensor('a'), ('?:choice', tensor('a'),
                                                     placeholder('a'))))))
    results = gm.all_matcher(pat)(x + x)

    self.assertEqual(len(results), 2)
    self.assertDictEqualIdenticalValues(results[0], {'a': x})
    self.assertDictEqualIdenticalValues(results[1], {'a': x})


class LayerMatchingTests(TestCase):

  def test_fully_connected(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      b = tf.Variable(tf.random_normal((3,)))
      s = tf.matmul(x, w) + b
      a = tf.nn.relu(s)

      bindings = gm.matcher(gp.Layer)(a)

      expected = {
          'weights': w._ref(),
          'biases': b._ref(),
          'in': x,
          'pre_activations': s,
          'activations': a
      }

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)

  def test_fully_connected_nobias(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      w = tf.Variable(tf.random_normal((3, 3)))
      s = tf.matmul(x, w)
      a = tf.nn.relu(s)

      bindings = gm.matcher(gp.Layer)(a)

      expected = {
          'weights': w._ref(),
          'in': x,
          'pre_activations': s,
          'activations': a
      }

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)
      self.assertNotIn('biases', bindings)

  def test_conv(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3, 3, 2))
      w = tf.Variable(tf.random_normal((3, 2, 2, 4)))
      b = tf.Variable(tf.random_normal((3, 3, 4)))
      s = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
      a = tf.nn.relu(s)

      bindings = gm.matcher(gp.Layer)(a)

      expected = {
          'weights': w._ref(),
          'biases': b._ref(),
          'in': x,
          'pre_activations': s,
          'activations': a
      }

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)

  def test_batchnorm_layer(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3, 3, 2))
      w = tf.Variable(tf.random_normal((3, 2, 2, 4)))
      b = tf.Variable(tf.random_normal((3, 3, 4)))
      s = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

      mean, variance = tf.nn.moments(s, axes=[0])
      s_normed = tf.nn.batch_normalization(s, mean, variance, 0.5, 0.5, 1e-5)
      a = tf.nn.relu(s_normed)

      bindings = gm.matcher(gp.LayerWithBatchNorm)(a)

      expected = {
          'weights': w._ref(),
          'biases': b._ref(),
          'in': x,
          'pre_activations': s,
          'final_activations': a
      }

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)


class BatchNormTests(TestCase):
  variance_epsilon = 1e-5

  def test_batchnorm(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      mean, variance = tf.nn.moments(x, axes=[0])
      offset, scale = tf.nn.moments(x, axes=[0])
      output = tf.nn.batch_normalization(x, mean, variance, offset, scale,
                                         self.variance_epsilon)

      bindings = gm.matcher(gp.BatchNorm)(output)

      expected = {'out': output, 'offset': offset, 'scale': scale, 'in': x}

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)

  def test_batchchnorm_noscale(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      mean, variance = tf.nn.moments(x, axes=[0])
      offset, _ = tf.nn.moments(x, axes=[0])
      output = tf.nn.batch_normalization(x, mean, variance, offset, None,
                                         self.variance_epsilon)

      bindings = gm.matcher(gp.BatchNorm)(output)

      expected = {'out': output, 'offset': offset, 'in': x}

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)
      self.assertNotIn('scale', bindings)

  def test_batchnorm_nooffset(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      mean, variance = tf.nn.moments(x, axes=[0])
      _, scale = tf.nn.moments(x, axes=[0])
      output = tf.nn.batch_normalization(x, mean, variance, None, scale,
                                         self.variance_epsilon)

      bindings = gm.matcher(gp.BatchNorm)(output)

      expected = {'out': output, 'scale': scale, 'in': x}

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)
      self.assertNotIn('offset', bindings)

  def test_batchnorm_noscaleoffset(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3))
      mean, variance = tf.nn.moments(x, axes=[0])
      output = tf.nn.batch_normalization(x, mean, variance, None, None,
                                         self.variance_epsilon)

      bindings = gm.matcher(gp.BatchNorm)(output)

      expected = {'out': output, 'in': x}

      self.assertDictContainsSubsetIdenticalValues(expected, bindings)
      self.assertNotIn('offset', bindings)
      self.assertNotIn('scale', bindings)

  def test_fused_batchnorm_outputs(self):
    with tf.Graph().as_default():
      x = tf.placeholder('float32', (10, 3, 3, 3))
      scale = tf.placeholder('float32', (3,))
      offset = tf.placeholder('float32', (3,))
      output, _, _ = tf.nn.fused_batch_norm(x, scale, offset)

      match = gm.matcher(gp.FusedBatchNormOutput)
      bindings = match(output)

      expected = {'out': output, 'in': x, 'scale': scale, 'offset': offset}
      self.assertDictContainsSubsetIdenticalValues(expected, bindings)


if __name__ == '__main__':
  tf.test.main()
