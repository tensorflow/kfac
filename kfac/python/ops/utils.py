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

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import moving_averages


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


def cholesky(tensor, damping):
  """Computes the inverse of tensor + damping * identity."""
  identity = tf.eye(tensor.shape.as_list()[0], dtype=tensor.dtype)
  damping = tf.cast(damping, dtype=tensor.dtype)
  return tf.cholesky(tensor + damping * identity)


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
    def _add_tensor_consumers_to_set(tensor, consumers_set):
      """Finds consumers of a tensor and add them to the current consumers set.
      """
      for consumer in set(tensor.consumers()):
        # These are the type of ops which relay a tensor to other ops without
        # doing anything to the tensor value, so recursively find the actual
        # consumers.
        if consumer.type in [
            "Identity", "ReadVariableOp", "Enter", "ResourceGather"]:
          for output in consumer.outputs:
            _add_tensor_consumers_to_set(output, consumers_set)
        else:
          consumers_set.add(consumer)

    if isinstance(var, resource_variable_ops.ResourceVariable):
      var = var.handle
    elif isinstance(var, tf.Variable):
      var = var.value()
    else:
      raise ValueError("%s does not appear to be a variable." % str(var))

    consumers = set()
    _add_tensor_consumers_to_set(var, consumers)
    return len(self._members.intersection(consumers))

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


def fwd_gradients(ys, xs, grad_xs=None, stop_gradients=None,
                  colocate_gradients_with_ops=True):
  """Compute forward-mode gradients."""
  # See b/37888268.

  # This version of forward-mode autodiff is based on code by Tim Cooijmans
  # and handles list arguments and certain special cases such as when the
  # ys doesn't depend on one or more of the xs, and when tf.IndexedSlices are
  # generated by the first tf.gradients call.

  us = [tf.zeros_like(y) + float("nan") for y in ys]
  dydxs = tf.gradients(ys, xs, grad_ys=us, stop_gradients=stop_gradients,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Deal with strange types that tf.gradients returns but can't
  # deal with.
  dydxs = [
      tf.convert_to_tensor(dydx) if isinstance(dydx, tf.IndexedSlices) else dydx
      for dydx in dydxs
  ]
  dydxs = [
      tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
  ]
  dysdx = tf.gradients(dydxs, us, grad_ys=grad_xs,
                       colocate_gradients_with_ops=colocate_gradients_with_ops)
  return dysdx


def on_tpu():
  """Returns True when building a TPU computation."""
  return tpu_function.get_tpu_context().number_of_shards is not None


def get_num_tpu_shards():
  return tpu_function.get_tpu_context().number_of_shards


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
    num_shards = get_num_tpu_shards()
    if num_shards is None:
      raise ValueError(
          "Cannot take cross_replica_mean() outside of TPU Context.")
    if num_shards == 1:
      return tensor
    return tf.contrib.tpu.cross_replica_sum(tensor / num_shards)


def cross_replica_sum(tensor, name=None):
  """Takes sum of values of a Tensor across all TPU cores.

  Args:
    tensor: Tensor to be synchronized.
    name: None or string. Name of Op.

  Returns:
    Sum of Tensor across all TPU cores.

  Raises:
    ValueError: If called outside of TPU context.
  """
  with tf.name_scope(name, "cross_replica_sum", [tensor]):
    num_shards = get_num_tpu_shards()
    if num_shards is None:
      raise ValueError(
          "Cannot take cross_replica_sum() outside of TPU Context.")
    if num_shards == 1:
      return tensor
    return tf.contrib.tpu.cross_replica_sum(tensor)


def get_replica_id():
  """Returns an id number for the current replica, counting from 0."""
  # This code is based on TensorTracer._add_replica_id_to_graph().

  # I'm assuming replicas and shards are always equal until someone tells me
  # different.
  num_replicas = get_num_tpu_shards()

  if not num_replicas:
    return None

  with tf.control_dependencies(None):
    # Uses None as dependency to run outside of TPU graph rewrites.
    return tpu_ops.tpu_replicated_input(
        list(range(num_replicas)),
        name="replica_id")


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

    if strides is not None and len(strides) == len(inputs.shape):
      strides = strides[1:-1]  # remove batch and channel dimension

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


def matmul_sparse_dense(A, B, name=None, transpose_a=False, transpose_b=False):  # pylint: disable=invalid-name
  """Computes matmul(A, B) where A is sparse, B is dense.

  Args:
    A: tf.IndexedSlices with dense shape [m, n].
    B: tf.Tensor with shape [n, k].
    name: str. Name of op.
    transpose_a: Bool. If true we transpose A before multiplying it by B.
      (Default: False)
    transpose_b: Bool. If true we transpose B before multiplying it by A.
      (Default: False)

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
    new_values = tf.matmul(
        A.values, B, transpose_a=transpose_a, transpose_b=transpose_b)
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


class AccumulatorVariable(object):
  """Accumulates values over multiple sess.run calls."""

  def __init__(self,
               name,
               var=None,
               acc_var_shape=None,
               acc_var_dtype=None):
    """Constructs a new `AccumulatorVariable`.

    If `var` is specified then accumulated value will be assigned to it after
    every 'num_steps_for_update' runs.  Otherwise `acc_var_shape` and
    `acc_var_dtype` must be specificed and the average accumulated value can be
    read by invoking `self.accumulated_value` property.

    Args:
      name: `string`. Name of the accumulator variable.
      var: tf.Variable. A value for this variable will be assigned after
        `num_steps` times `assign()` function is invoked.
      acc_var_shape: Shape of the variable to be accumulated. Required only if
        `var` is not passed.
      acc_var_dtype: Data type of the variable to be accumulated. Required only
        if `var` is not passed.

    Raises:
     ValueError: If `var` is not passed and `acc_var_shape` or `acc_var_dtype`
       is None.
    """

    if var is None and (acc_var_shape is None or acc_var_dtype is None):
      raise ValueError("If var is not specified then both acc_var_shape and"
                       "acc_var_dtype must be passed.")

    with tf.variable_scope("acc_var", reuse=tf.AUTO_REUSE):
      if var is None:
        self._var = tf.get_variable(
            name + "_orig_var",
            initializer=tf.zeros_initializer(),
            shape=acc_var_shape,
            dtype=acc_var_dtype,
            trainable=False,
            use_resource=True)
      else:
        self._var = var

      self._acc_var = tf.get_variable(
          name,
          initializer=tf.zeros_initializer(),
          shape=var.shape if var is not None else acc_var_shape,
          dtype=var.dtype if var is not None else acc_var_dtype,
          trainable=False,
          use_resource=True)

      self._counter = tf.get_variable(
          shape=(),
          dtype=tf.int32,
          name=name+"_counter",
          initializer=tf.zeros_initializer(),
          trainable=False,
          use_resource=True)

  def accumulate(self,
                 value,
                 ema_decay=0.,
                 num_steps_for_update=1,
                 zero_debias=False):
    """Adds `value` to the accumulator var and assigns to `var` conditionally.

    Adds `value` to accumulator variable. If the function is called
    `num_steps_for_update` number of times then the accumulated value, divided
    by `num_steps_for_update`, is assigned to `var` and accumulated value is
    reset to zero and new accumulation cycle is started.

    Args:
      value: A tensor, of the same shape and type as `var`
      (or `acc_var_shape` and `acc_var_dtype`)passed to the initializer.
      ema_decay: float, If the vale is zero, then accumulated value is assigned
        directly used otherwise moving avergae of the accumulated value is
        computed. Optional (Default: 0.)
      num_steps_for_update: int, The number of steps to accumulate in each
        cycle before resetting. Optional, Default: 1.
      zero_debias: boolean, Whether to zero-debias the moving averages.
        Optional, Default: False.

    Returns:
      An op which does accumulation and manages accumulation cycles.
    """
    inc_step_op = tf.assign_add(self._counter, 1)
    acc_op = tf.assign_add(self._acc_var, value)

    with tf.control_dependencies([inc_step_op]):
      should_reset = tf.equal(tf.mod(self._counter, num_steps_for_update), 0)

    with tf.control_dependencies([acc_op, inc_step_op]):
      var_assign_op = tf.cond(
          should_reset,
          self._assign_acc_value_to_var(ema_decay, zero_debias,
                                        num_steps_for_update), tf.no_op)

      with tf.control_dependencies([var_assign_op]):
        return tf.cond(should_reset, self._reset_var, tf.no_op)

  @property
  def value(self):
    """Returns the value of accumulator variable which is reset."""
    return tf.identity(self._acc_var)

  @property
  def accumulated_value(self):
    """Returns the accumulated value."""
    return tf.identity(self._var)

  def _assign_acc_value_to_var(self, ema_decay, zero_debias,
                               num_steps_for_update):
    """Assigns average accumulate value to self._var."""
    def _assign():
      """No docstring needed, pylint."""
      avg_acc_val = (1. / num_steps_for_update) * self._acc_var
      ema_decay_tensor = tf.convert_to_tensor(ema_decay)

      def _assign_moving_average():
        # These moving averages use "slots" internally, which aren't implemented
        # with resource variables. Thus I don't trust them.
        # TODO(b/121265708): Someone should look into this!

        # I'm adding this scope to try to force the use of resource variables
        # by the "slots", but it probably won't work.
        with tf.variable_scope("moving_avg", use_resource=True):
          return tf.group(
              moving_averages.assign_moving_average(
                  self._var, avg_acc_val, ema_decay, zero_debias=zero_debias))

      return tf.cond(tf.greater(ema_decay_tensor, 0.),
                     _assign_moving_average,
                     lambda: tf.group(tf.assign(self._var, avg_acc_val)))

    return _assign

  def _reset_var(self):
    """Resets step counter and accumulator variable."""
    var_zero_assign_op = tf.assign(
        self._acc_var, tf.zeros(self._acc_var.shape, dtype=self._acc_var.dtype))
    with tf.control_dependencies([var_zero_assign_op]):
      return tf.group(tf.assign(self._counter, tf.constant(0)))


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
      raise ValueError(
          "all tensors must have the same dtype. The tensors are {}".format(
              tensors))

    shape = tensors[0].shape[1:]
    if not all(tensor.shape[1:] == shape for tensor in tensors):
      raise ValueError("All tensors must have shape = %s (excluding batch "
                       "dimension)." % shape)

    self.tensors = tensors

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
      return tf.concat(self.tensors, axis=0)

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


def _check_match_lists_of_pairs(list1, list2):
  for (_, var1), (_, var2) in zip(list1, list2):
    if var1 is not var2:
      raise ValueError("The variables referenced by the two arguments "
                       "must match.")


def sprod(scalar, list_):
  # Product of scalar with list of items.
  return tuple(scalar*item for item in list_)


def sprod_p(scalar, list_):
  # Product of scalar with list of (item, var) pairs.
  return tuple((scalar*item, var) for (item, var) in list_)


def sum_(list1, list2):
  # Element-wise sum of lists of tensors.
  return tuple(item1 + item2 for item1, item2 in zip(list1, list2))


def sum_p(list1, list2):
  # Element-wise sum of lists of (tensor, var) pairs.
  _check_match_lists_of_pairs(list1, list2)
  return tuple((item1 + item2, var1)
               for (item1, var1), (item2, var2) in zip(list1, list2))


def ip(list1, list2):
  # Inner product of lists of tensors.
  return tf.add_n(tuple(tf.reduce_sum(tensor1 * tensor2)
                        for tensor1, tensor2 in zip(list1, list2)))


def ip_p(list1, list2):
  # Inner product of lists of (tensor, var) pairs.
  _check_match_lists_of_pairs(list1, list2)

  return ip(tuple(tensor for (tensor, _) in list1),
            tuple(tensor for (tensor, _) in list2))


def assert_variables_match_pairs_list(a_and_vars,
                                      b_and_vars,
                                      error_message=None):
  """Assert the variables in two lists of (tensor, var) pairs are the same.

  Args:
    a_and_vars: a list of (tensor, variable) pairs.
    b_and_vars: a list of (tensor, variable) pairs.
    error_message: an optional string prepended to the error message.

  Raises:
    ValueError: if any variables in the input pair lists are not the same.
  """
  _, a_variables = zip(*a_and_vars)
  _, b_variables = zip(*b_and_vars)
  variable_mismatch_indices = []
  for vi, (a_var, b_var) in enumerate(zip(a_variables, b_variables)):
    if a_var is not b_var:
      variable_mismatch_indices.append(vi)

  if variable_mismatch_indices:
    mismatch_indices_str = ", ".join(map(str, variable_mismatch_indices))
    a_variables_str = ", ".join(map(str, a_variables))
    b_variables_str = ", ".join(map(str, b_variables))
    error_str = "Mismatch on variable lists at indices {}.\n{}\n{}".format(
        mismatch_indices_str, a_variables_str, b_variables_str)
    if error_message:
      error_str = "{} {}".format(error_message, error_str)
    raise ValueError(error_str)
