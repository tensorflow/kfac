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
"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.ops import resource_variable_ops

# Method used for inverting matrices.
POSDEF_INV_METHOD = "cholesky"
POSDEF_EIG_METHOD = "self_adjoint"


def set_global_constants(posdef_inv_method=None):
  """Sets various global constants used by the classes in this module."""
  global POSDEF_INV_METHOD

  if posdef_inv_method is not None:
    POSDEF_INV_METHOD = posdef_inv_method


class SequenceDict(object):
  """A dict convenience wrapper that allows getting/setting with sequences."""

  def __init__(self, iterable=None):
    self._dict = dict(iterable or [])

  def __getitem__(self, key_or_keys):
    if isinstance(key_or_keys, (tuple, list)):
      return list(map(self.__getitem__, key_or_keys))
    else:
      return self._dict[key_or_keys]

  def __setitem__(self, key_or_keys, val_or_vals):
    if isinstance(key_or_keys, (tuple, list)):
      for key, value in zip(key_or_keys, val_or_vals):
        self[key] = value
    else:
      self._dict[key_or_keys] = val_or_vals

  def items(self):
    return list(self._dict.items())


def tensors_to_column(tensors):
  """Converts a tensor or list of tensors to a column vector.

  Args:
    tensors: A tensor or list of tensors.

  Returns:
    The tensors reshaped into vectors and stacked on top of each other.
  """
  if isinstance(tensors, (tuple, list)):
    return tf.concat(
        tuple(tf.reshape(tensor, [-1, 1]) for tensor in tensors), axis=0)
  else:
    return tf.reshape(tensors, [-1, 1])


def column_to_tensors(tensors_template, colvec):
  """Converts a column vector back to the shape of the given template.

  Args:
    tensors_template: A tensor or list of tensors.
    colvec: A 2d column vector with the same shape as the value of
        tensors_to_column(tensors_template).

  Returns:
    X, where X is tensor or list of tensors with the properties:
     1) tensors_to_column(X) = colvec
     2) X (or its elements) have the same shape as tensors_template (or its
        elements)
  """
  if isinstance(tensors_template, (tuple, list)):
    offset = 0
    tensors = []
    for tensor_template in tensors_template:
      sz = np.prod(tensor_template.shape.as_list(), dtype=np.int32)
      tensor = tf.reshape(colvec[offset:(offset + sz)], tensor_template.shape)
      tensors.append(tensor)
      offset += sz

    tensors = tuple(tensors)
  else:
    tensors = tf.reshape(colvec, tensors_template.shape)

  return tensors


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product two matrices."""
  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def layer_params_to_mat2d(vector):
  """Converts a vector shaped like layer parameters to a 2D matrix.

  In particular, we reshape the weights/filter component of the vector to be
  2D, flattening all leading (input) dimensions. If there is a bias component,
  we concatenate it to the reshaped weights/filter component.

  Args:
    vector: A Tensor or pair of Tensors shaped like layer parameters.

  Returns:
    A 2D Tensor with the same coefficients and the same output dimension.
  """
  if isinstance(vector, (tuple, list)):
    w_part, b_part = vector
    w_part_reshaped = tf.reshape(w_part, [-1, w_part.shape.as_list()[-1]])
    return tf.concat((w_part_reshaped, tf.reshape(b_part, [1, -1])), axis=0)
  elif isinstance(vector, tf.IndexedSlices):
    return vector
  else:  # Tensor or Tensor-like.
    return tf.reshape(vector, [-1, vector.shape.as_list()[-1]])


def mat2d_to_layer_params(vector_template, mat2d):
  """Converts a canonical 2D matrix representation back to a vector.

  Args:
    vector_template: A Tensor or pair of Tensors shaped like layer parameters.
    mat2d: A 2D Tensor with the same shape as the value of
        layer_params_to_mat2d(vector_template).

  Returns:
    A Tensor or pair of Tensors with the same coefficients as mat2d and the same
        shape as vector_template.
  """
  if isinstance(vector_template, (tuple, list)):
    w_part, b_part = mat2d[:-1], mat2d[-1]
    return tf.reshape(w_part, vector_template[0].shape), b_part
  elif isinstance(vector_template, tf.IndexedSlices):
    if not isinstance(mat2d, tf.IndexedSlices):
      raise TypeError(
          "If vector_template is an IndexedSlices, so should mat2d.")
    return mat2d
  else:
    return tf.reshape(mat2d, vector_template.shape)


def posdef_inv(tensor, damping):
  """Computes the inverse of tensor + damping * identity."""
  identity = tf.eye(tensor.shape.as_list()[0], dtype=tensor.dtype)
  damping = tf.cast(damping, dtype=tensor.dtype)
  return posdef_inv_functions[POSDEF_INV_METHOD](tensor, identity, damping)


def posdef_inv_matrix_inverse(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) directly."""
  return tf.matrix_inverse(tensor + damping * identity)


def posdef_inv_cholesky(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with Cholesky."""
  chol = tf.cholesky(tensor + damping * identity)
  return tf.cholesky_solve(chol, identity)


def posdef_inv_eig(tensor, identity, damping):
  """Computes inverse(tensor + damping * identity) with eigendecomposition."""
  eigenvalues, eigenvectors = tf.self_adjoint_eig(tensor + damping * identity)
  return tf.matmul(eigenvectors / eigenvalues, eigenvectors, transpose_b=True)


posdef_inv_functions = {
    "matrix_inverse": posdef_inv_matrix_inverse,
    "cholesky": posdef_inv_cholesky,
    "eig": posdef_inv_eig,
}


def posdef_eig(mat):
  """Computes the eigendecomposition of a positive semidefinite matrix."""
  return posdef_eig_functions[POSDEF_EIG_METHOD](mat)


def posdef_eig_svd(mat):
  """Computes the singular values and left singular vectors of a matrix."""
  evals, evecs, _ = tf.svd(mat)

  return evals, evecs


def posdef_eig_self_adjoint(mat):
  """Computes eigendecomposition using self_adjoint_eig."""
  evals, evecs = tf.self_adjoint_eig(mat)
  evals = tf.abs(evals)  # Should be equivalent to svd approach.

  return evals, evecs


posdef_eig_functions = {
    "self_adjoint": posdef_eig_self_adjoint,
    "svd": posdef_eig_svd,
}


class SubGraph(object):
  """Defines a subgraph given by all the dependencies of a given set of outputs.
  """

  def __init__(self, outputs):
    # Set of all ancestor Tensors, Ops to 'outputs'.
    self._members = set()

    self._iter_add(outputs)

  def _iter_add(self, root):
    """Iteratively adds all of nodes' ancestors using depth first search."""
    stack = [root]
    while stack:
      nodes = stack.pop()
      for node in nodes:
        if node in self._members:
          continue
        self._members.add(node)

        if isinstance(node, tf.Tensor):
          stack.append((node.op,))
        elif isinstance(node, tf.Operation):
          stack.append(node.inputs)

  def is_member(self, node):
    """Check if 'node' is in this subgraph."""
    return node in self._members

  def variable_uses(self, var):
    """Computes number of times a variable is used.

    Args:
      var: Variable or ResourceVariable instance.

    Returns:
      Number of times a variable is used within this subgraph.

    Raises:
      ValueError: If 'var' is not a variable type.
    """
    if isinstance(var, resource_variable_ops.ResourceVariable):
      var = var.handle
    elif isinstance(var, tf.Variable):
      var = var.value()
    else:
      raise ValueError("%s does not appear to be a variable." % str(var))

    return len(self._members.intersection(set(var.consumers())))

  def filter_list(self, node_list):
    """Filters 'node_list' to nodes in this subgraph."""
    filtered_list = []
    for node in node_list:
      if self.is_member(node):
        filtered_list.append(node)
    return filtered_list


def generate_random_signs(shape, dtype=tf.float32):
  """Generate a random tensor with {-1, +1} entries."""
  ints = tf.random_uniform(shape, maxval=2, dtype=tf.int32)
  return 2 * tf.cast(ints, dtype=dtype) - 1


def fwd_gradients(ys, xs, grad_xs=None, stop_gradients=None):
  """Compute forward-mode gradients."""
  # See b/37888268.

  # This version of forward-mode autodiff is based on code by Tim Cooijmans
  # and handles list arguments and certain special cases such as when the
  # ys doesn't depend on one or more of the xs, and when tf.IndexedSlices are
  # generated by the first tf.gradients call.

  us = [tf.zeros_like(y) + float("nan") for y in ys]
  dydxs = tf.gradients(ys, xs, grad_ys=us, stop_gradients=stop_gradients)

  # Deal with strange types that tf.gradients returns but can't
  # deal with.
  dydxs = [
      tf.convert_to_tensor(dydx) if isinstance(dydx, tf.IndexedSlices) else dydx
      for dydx in dydxs
  ]
  dydxs = [
      tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
  ]

  dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs)

  return dysdx


def on_tpu():
  """Returns True when building a TPU computation."""
  return tpu_function.get_tpu_context().number_of_shards is not None


def cross_replica_mean(tensor, name=None):
  """Takes mean value of a Tensor across all TPU cores.

  Args:
    tensor: Tensor to be synchronized.
    name: None or string. Name of Op.

  Returns:
    Average of Tensor across all TPU cores.

  Raises:
    ValueError: If called outside of TPU context.
  """
  with tf.name_scope(name, "cross_replica_mean", [tensor]):
    num_shards = tpu_function.get_tpu_context().number_of_shards
    if num_shards is None:
      raise ValueError(
          "Cannot take cross_replica_mean() outside of TPU Context.")
    if num_shards == 1:
      return tensor
    return tf.contrib.tpu.cross_replica_sum(tensor / num_shards)


def ensure_sequence(obj):
  """If `obj` isn't a tuple or list, return a tuple containing `obj`."""
  if isinstance(obj, (tuple, list)):
    return obj
  else:
    return (obj,)


def batch_execute(global_step, thunks, batch_size, name=None):
  """Executes a subset of ops per global step.

  Given a list of thunks, each of which produces a single stateful op,
  ensures that exactly 'batch_size' ops are run per global step. Ops are
  scheduled in a round-robin fashion. For example, with 3 ops

    global_step | op0 | op1 | op2
    ------------+-----+-----+-----
        0       |  x  |  x  |
    ------------+-----+-----+-----
        1       |  x  |     |  x
    ------------+-----+-----+-----
        2       |     |  x  |  x
    ------------+-----+-----+-----
        3       |  x  |  x  |
    ------------+-----+-----+-----
        4       |  x  |     |  x

  Does not guarantee order of op execution within a single global step.

  Args:
    global_step: Tensor indicating time. Determines which ops run.
    thunks: List of thunks. Each thunk encapsulates one op. Return values are
      ignored.
    batch_size: int. Number of ops to execute per global_step.
    name: string or None. Name scope for newly added ops.

  Returns:
    List of ops. Exactly 'batch_size' ops are guaranteed to have an effect
    every global step.
  """

  def true_fn(thunk):
    """Ensures thunk is executed and returns an Op (not a Tensor)."""

    def result():
      with tf.control_dependencies([thunk()]):
        return tf.no_op()

    return result

  def false_fn(_):
    """Executes a no-op."""

    def result():
      return tf.no_op()

    return result

  with tf.name_scope(name, "batch_execute"):
    true_fns = [true_fn(thunk) for thunk in thunks]
    false_fns = [false_fn(thunk) for thunk in thunks]
    num_thunks = len(thunks)
    conditions = [
        tf.less(
            tf.mod(batch_size - 1 + global_step * batch_size - j, num_thunks),
            batch_size) for j in range(num_thunks)
    ]
    result = [
        tf.cond(condition, true_fn, false_fn)
        for (condition, true_fn,
             false_fn) in zip(conditions, true_fns, false_fns)
    ]
    return result


def extract_convolution_patches(inputs,
                                filter_shape,
                                padding,
                                strides=None,
                                dilation_rate=None,
                                name=None,
                                data_format=None):
  """Extracts inputs to each output coordinate in tf.nn.convolution.

  This is a generalization of tf.extract_image_patches() to tf.nn.convolution(),
  where the number of spatial dimensions may be something other than 2.

  Assumes,
  - First dimension of inputs is batch_size
  - Convolution filter is applied to all input channels.

  Args:
    inputs: Tensor of shape [batch_size, ..spatial_image_shape..,
      ..spatial_filter_shape.., in_channels]. Inputs to tf.nn.convolution().
    filter_shape: List of ints. Shape of filter passed to tf.nn.convolution().
    padding: string. Padding method. One of "VALID", "SAME".
    strides: None or list of ints. Strides along spatial dimensions.
    dilation_rate: None or list of ints. Dilation along spatial dimensions.
    name: None or str. Name of Op.
    data_format: None or str. Format of data.

  Returns:
    Tensor of shape [batch_size, ..spatial_image_shape..,
      ..spatial_filter_shape.., in_channels]

  Raises:
    ValueError: If data_format does not put channel last.
    ValueError: If inputs and filter disagree on in_channels.
  """
  if not is_data_format_channel_last(data_format):
    raise ValueError("Channel must be last dimension.")
  with tf.name_scope(name, "extract_convolution_patches",
                     [inputs, filter_shape, padding, strides, dilation_rate]):
    batch_size = inputs.shape.as_list()[0]
    in_channels = inputs.shape.as_list()[-1]

    # filter_shape = spatial_filter_shape + [in_channels, out_channels]
    spatial_filter_shape = filter_shape[:-2]
    if in_channels != filter_shape[-2]:
      raise ValueError("inputs and filter_shape must agree on in_channels.")

    # Map each input feature to a location in the output.
    out_channels = np.prod(spatial_filter_shape) * in_channels
    filters = tf.eye(out_channels)
    filters = tf.reshape(
        filters,
        list(spatial_filter_shape) + [in_channels, out_channels])

    result = tf.nn.convolution(
        inputs,
        filters,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate)
    spatial_output_shape = result.shape.as_list()[1:-1]
    result = tf.reshape(result, [batch_size or -1] + spatial_output_shape +
                        list(spatial_filter_shape) + [in_channels])

    return result


def extract_pointwise_conv2d_patches(inputs,
                                     filter_shape,
                                     name=None,
                                     data_format=None):
  """Extract patches for a 1x1 conv2d.

  Args:
    inputs: 4-D Tensor of shape [batch_size, height, width, in_channels].
    filter_shape: List of 4 ints. Shape of filter to apply with conv2d()
    name: None or str. Name for Op.
    data_format: None or str. Format for data. See 'data_format' in
      tf.nn.conv2d() for details.

  Returns:
    Tensor of shape [batch_size, ..spatial_input_shape..,
    ..spatial_filter_shape.., in_channels]

  Raises:
    ValueError: if inputs is not 4-D.
    ValueError: if filter_shape is not [1, 1, ?, ?]
    ValueError: if data_format is not channels-last.
  """
  if inputs.shape.ndims != 4:
    raise ValueError("inputs must have 4 dims.")
  if len(filter_shape) != 4:
    raise ValueError("filter_shape must have 4 dims.")
  if filter_shape[0] != 1 or filter_shape[1] != 1:
    raise ValueError("filter_shape must have shape 1 along spatial dimensions.")
  if not is_data_format_channel_last(data_format):
    raise ValueError("data_format must be channels last.")
  with tf.name_scope(name, "extract_pointwise_conv2d_patches",
                     [inputs, filter_shape]):
    ksizes = [1, 1, 1, 1]  # Spatial shape is 1x1.
    strides = [1, 1, 1, 1]  # Operate on all pixels.
    rates = [1, 1, 1, 1]  # Dilation has no meaning with spatial shape = 1.
    padding = "VALID"  # Doesn't matter.
    result = tf.extract_image_patches(inputs, ksizes, strides, rates, padding)

    batch_size, input_height, input_width, in_channels = inputs.shape.as_list()
    filter_height, filter_width, in_channels, _ = filter_shape
    return tf.reshape(result, [
        batch_size, input_height, input_width, filter_height, filter_width,
        in_channels
    ])


def is_data_format_channel_last(data_format):
  """True if data_format puts channel last."""
  if data_format is None:
    return True
  return data_format.endswith("C")


def matmul_sparse_dense(A, B, name=None):  # pylint: disable=invalid-name
  """Computes matmul(A, B) where A is sparse, B is dense.

  Args:
    A: tf.IndexedSlices with dense shape [m, n].
    B: tf.Tensor with shape [n, k].
    name: str. Name of op.

  Returns:
    tf.IndexedSlices resulting from matmul(A, B).

  Raises:
    ValueError: If A doesn't represent a matrix.
    ValueError: If B is not rank-2.
  """
  with tf.name_scope(name, "matmul_sparse_dense", [A, B]):
    if A.indices.shape.ndims != 1 or A.values.shape.ndims != 2:
      raise ValueError("A must represent a matrix. Found: %s." % A)
    if B.shape.ndims != 2:
      raise ValueError("B must be a matrix.")
    new_values = tf.matmul(A.values, B)
    return tf.IndexedSlices(
        new_values,
        A.indices,
        dense_shape=tf.stack([A.dense_shape[0], new_values.shape[1]]))


def matmul_diag_sparse(A_diag, B, name=None):  # pylint: disable=invalid-name
  """Computes matmul(A, B) where A is a diagonal matrix, B is sparse.

  Args:
    A_diag: diagonal entries of matrix A of shape [m, m].
    B: tf.IndexedSlices. Represents matrix of shape [m, n].
    name: str. Name of op.

  Returns:
    tf.IndexedSlices resulting from matmul(A, B).

  Raises:
    ValueError: If A_diag is not rank-1.
    ValueError: If B doesn't represent a matrix.
  """
  with tf.name_scope(name, "matmul_diag_sparse", [A_diag, B]):
    A_diag = tf.convert_to_tensor(A_diag)
    if A_diag.shape.ndims != 1:
      raise ValueError("A_diag must be a rank-1 Tensor.")
    if B.indices.shape.ndims != 1 or B.values.shape.ndims != 2:
      raise ValueError("B must represent a matrix. Found: %s." % B)
    a = tf.gather(A_diag, B.indices)
    a = tf.reshape(a, list(a.shape) + [1] * (B.values.shape.ndims - 1))
    return tf.IndexedSlices(a * B.values, B.indices, dense_shape=B.dense_shape)


class PartitionedTensor(object):
  """A Tensor partitioned across its 0-th dimension."""

  def __init__(self, tensors):
    """Initializes PartitionedTensor.

    Args:
      tensors: List of Tensors. All Tensors must agree on shape (excepting
        batch dimension) and dtype.

    Raises:
      ValueError: If 'tensors' has length zero.
      ValueError: if contents of 'tensors' don't agree on shape or dtype.
    """
    if not tensors:
      raise ValueError("tensors must be a list of 1+ Tensors.")

    dtype = tensors[0].dtype
    if not all(tensor.dtype == dtype for tensor in tensors):
      raise ValueError("all tensors must have dtype = %s." % dtype)

    shape = tensors[0].shape[1:]
    if not all(tensor.shape[1:] == shape for tensor in tensors):
      raise ValueError("All tensors must have shape = %s (excluding batch "
                       "dimension)." % shape)

    self.tensors = tensors
    self._concats = {}  # {device: Tensor}

  @property
  def shape(self):
    feature_shape = self.tensors[0].shape[1:]
    batch_size = sum([tensor.shape[0] for tensor in self.tensors],
                     tf.Dimension(0))
    return tf.TensorShape([batch_size]).concatenate(feature_shape)

  def get_shape(self):
    return self.shape

  @property
  def dtype(self):
    return self.tensors[0].dtype

  def __str__(self):
    return "PartitionedTensor([%s, ...], dtype=%s, shape=%s)" % (
        self.tensors[0].name, self.dtype.name, tuple(self.shape.as_list()))

  def __hash__(self):
    return hash(tuple(self.tensors))

  def __eq__(self, other):
    if not isinstance(other, PartitionedTensor):
      return False
    return self.tensors == other.tensors

  def __ne__(self, other):
    return not self == other  # pylint: disable=g-comparison-negation

  def __getitem__(self, key):
    return self.as_tensor()[key]

  def as_tensor(self, dtype=None, name=None, as_ref=False):
    with tf.name_scope(name, "PartitionedTensor.as_tensor", self.tensors):
      assert not as_ref
      assert dtype in [None, self.dtype]
      result = tf.concat(self.tensors, axis=0)

      # Cache 'result' if we haven't already cached a value for this device.
      if result.device not in self._concats:
        self._concats[result.device] = result
      return self._concats[result.device]

  @property
  def device(self):
    # PartitionedTensors in general do not live on a single device.  If the
    # device cannot be determined unambiguously this property will return None.
    device = self.tensors[0].device
    if all(tensor.device == device for tensor in self.tensors):
      return device
    return None


tf.register_tensor_conversion_function(
    PartitionedTensor,
    lambda val, dtype, name, as_ref: val.as_tensor(dtype, name, as_ref))


# TODO(b/69623235): Add a function for finding tensors that share gradients
# to eliminate redundant fisher factor computations.
