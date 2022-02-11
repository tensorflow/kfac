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
"""Convenience functions for writing patterns in Python code.."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from kfac.python.ops.tensormatch import tensorflow_graph_util as util


## patterns


def Op(name=None):
  return ('?', name, util.is_op)


def Tensor(name=None):
  return ('?', name, util.is_tensor)


def Variable(name=None):
  return ('?', name, util.is_var)


def Const(name=None):
  return ('?', name, util.is_const)


def Placeholder(name=None):
  return ('?', name, util.is_placeholder)


util.import_ops_no_clobber(globals(), dir(tf.raw_ops))

# NOTE(mattjj): renamed in TF 1.0, but not registered as an op in 1.0.1
Unstack = util.make_op_pattern('Unpack')  # pylint: disable=invalid-name

## convenient compound patterns

# The op definitions are pulled in via the op_def_registry, which is
# why we disable the undefined variable check for e.g. Rsqrt, Mul, etc.
# Otherwise we would have to refer to them by name rather than object.
# pylint: disable=undefined-variable


def BatchNorm(in_pattern=Tensor('in'),
              scale_name='scale',
              offset_name='offset',
              output_name='out'):
  """Pattern constructor for matching tf.nn.batch_normalization subgraphs."""
  inv_pat = (Tensor('inv'), ('In', ('?:choice', Rsqrt,
                                    (Mul, ('In', (Tensor, ('In', Rsqrt)),
                                           Tensor(scale_name))))))
  without_offset_pat = (Mul, ('In', Tensor, Tensor('inv')))
  with_offset_pat = (Sub, ('In', Tensor(offset_name),
                           (Tensor, ('In', (Mul, ('In', Tensor,
                                                  Tensor('inv')))))))
  return (Tensor(output_name),
          ('In', (AddV2, ('In', (Tensor, ('In', (Mul, ('In', in_pattern,
                                                       inv_pat)))),
                          (Tensor, ('In', ('?:choice', with_offset_pat,
                                           without_offset_pat)))))))


def FusedBatchNormOutput(in_pattern=Tensor('in'),
                         scale_name='scale',
                         offset_name='offset',
                         output_name='out'):
  """Pattern constructor for matching tf.nn.fused_batch_norm subgraphs."""
  return (Tensor(output_name),
          ('In',
           (('?:choice', FusedBatchNorm, FusedBatchNormV2, FusedBatchNormV3),
            ('In', in_pattern, Tensor(scale_name), Tensor(offset_name), Tensor,
             Tensor))))


# TODO(mattjj): add more ops to this pattern
Nonlinearity = ('?:choice', Relu, Tanh)  # pylint: disable=invalid-name


def ScaleAndShift(in_pattern=Tensor('in'),
                  scale_name='scale',
                  shift_name='shift',
                  output_name='out'):
  """Pattern constructor for matching scale & shift operation subgraphs."""

  scale_pat_r = (Mul, ('In', in_pattern, Variable(scale_name)))
  scale_pat_l = (Mul, ('In', Variable(scale_name), in_pattern))

  scale_pat = ('?:choice', scale_pat_r, scale_pat_l)

  pat_r = (('?:choice', Add, AddV2),
           ('In', (Tensor, ('In', scale_pat)), Variable(shift_name)))
  pat_l = (('?:choice', Add, AddV2),
           ('In', Variable(shift_name), (Tensor, ('In', scale_pat))))

  return (Tensor(output_name), ('In', ('?:choice', pat_r, pat_l, scale_pat)))


def Affine(in_pattern=Tensor('in'),
           linear_op_name='linear_op',
           weights_name='weights',
           biases_name='biases',
           output_name='pre_activations'):
  """Pattern constructor for matching affine operation subgraphs."""
  linear_pat = (('?:choice', Conv2D(linear_op_name), MatMul(linear_op_name),
                 BatchMatMulV2(linear_op_name)),
                ('In', in_pattern, Variable(weights_name)))
  affine_pat_r = (('?:choice', Add, BiasAdd, AddV2),
                  ('In', (Tensor, ('In', linear_pat)), Variable(biases_name)))
  affine_pat_l = (('?:choice', Add, BiasAdd, AddV2),
                  ('In', Variable(biases_name), (Tensor, ('In', linear_pat))))
  affine_pat = ('?:choice', affine_pat_r, affine_pat_l)
  return (Tensor(output_name), ('In', ('?:choice', affine_pat, linear_pat)))


def Embed(in_pattern=Tensor('in'),
          linear_op_name='linear_op',
          weights_name='weights',
          axis_name='axis',
          output_name='pre_activations'):
  """Pattern constructor for matching embedding layer subgraphs."""
  embed_v1 = (('?:choice', Gather(linear_op_name),
               ResourceGather(linear_op_name)),
              ('In', Variable(weights_name), in_pattern))
  embed_v2 = (GatherV2(linear_op_name),
              ('In', Variable(weights_name), in_pattern, Tensor(axis_name)))
  embed = ('?:choice', embed_v1, embed_v2)

  return (Tensor(output_name), ('In', embed))


# Only used in tests:
def Layer(in_pattern=Tensor('in'), **kwargs):
  """Pattern constructor for matching a basic layer."""
  return (Tensor('activations'), ('In', (Nonlinearity, ('In', Affine(
      in_pattern, **kwargs)))))


# Only used in tests:
def LayerWithBatchNorm(in_pattern=Tensor('in')):
  """Pattern constructor for matching a layer with batch normalization."""
  return (Tensor('final_activations'),
          ('In', (Nonlinearity, ('In', BatchNorm(Affine(in_pattern))))))


# pylint: enable=undefined-variable
