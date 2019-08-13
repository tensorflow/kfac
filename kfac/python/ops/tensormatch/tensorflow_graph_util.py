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
"""Abstraction layer for working with the TensorFlow graph model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import six
import tensorflow as tf

from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import resource_variable_ops

from kfac.python.ops import utils


def is_op(node):
  return isinstance(node, tf.Operation)


def is_tensor(node):
  # return (isinstance(node, (tf.Tensor, tf.Variable))
  #         or resource_variable_ops.is_resource_variable(node))
  return tf_ops.is_dense_tensor_like(node)


def is_var(node):
  if not is_tensor(node):
    return False
  if node.op.type.startswith('Variable'):
    return True
  if ((resource_variable_ops.is_resource_variable(node) or
       utils.is_reference_variable(node))):
    return True
  if node.dtype == tf.resource and node.op.type == 'VarHandleOp':
    return True
  return False


def is_const(node):
  return is_tensor(node) and node.op.type == 'Const'


def is_placeholder(node):
  return is_tensor(node) and node.op.type == 'Placeholder'


def is_leaf(node):
  return is_var(node) or is_const(node) or is_placeholder(node)


def is_identity(node):
  if not is_op(node):
    return False
  # For ResourceVariables, a 'ReadVariableOp' has a single 'Enter' input, which
  # in turn has a Tensor with dtype == resource as input.
  return node.type in {'Identity', 'ReadVariableOp', 'Enter'}


def op_type_is(typename):

  def is_op_with_typename(node):
    return is_op(node) and node.type == typename

  return is_op_with_typename


def reduce_identity_ops(node):
  while is_tensor(node) and is_identity(node.op):
    assert len(node.op.inputs) == 1, 'identity op should have one input.'
    node = node.op.inputs[0]
  return node


def expand_inputs(node):
  """Return a list of input nodes for a given TF graph node (or node list)."""
  if is_op(node):
    return [reduce_identity_ops(tensor) for tensor in node.inputs[:]]
  elif is_tensor(node) and not is_leaf(node):
    return [reduce_identity_ops(node).op]
  elif isinstance(node, list) and all(is_tensor(elt) for elt in node):
    ops = {reduce_identity_ops(tensor).op for tensor in node}
    if len(ops) == 1:
      return [ops.pop()]
    raise ValueError
  return None


def expand_outputs(node):
  """Return a list of output nodes for a given TF graph node."""
  if is_op(node):
    return node.outputs[:]
  elif isinstance(node, tf.Variable):
    return node.value().consumers()
  elif is_tensor(node):
    return node.consumers()
  return None


def make_op_pattern(typename):
  """Makes a pattern that matches a given Op type."""

  def op_fun(name=None):
    return ('?', name, op_type_is(typename))

  op_fun_name = typename.encode('ascii', 'ignore')

  # In Python 3, str.encode() produces a bytes object. Convert this to an ASCII
  # str.
  if six.PY3:
    op_fun_name = op_fun_name.decode('ascii')

  op_fun.__name__ = op_fun_name
  return op_fun


def import_ops_no_clobber(dct, op_names):
  for name in op_names:
    if name not in dct:
      dct[name] = make_op_pattern(name)
