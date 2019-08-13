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
"""Curvature matrix-vector multiplication."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow.python.util import nest
from kfac.python.ops import utils


class CurvatureMatrixVectorProductComputer(object):
  """Class for computing matrix-vector products for Fishers and GGNs.

  In other words we compute M*v where M is the matrix, v is the vector, and
  * refers to standard matrix/vector multiplication (not element-wise
  multiplication).

  The matrices are defined in terms of some differential quantity of the total
  loss function with respect to a provided list of tensors ("wrt_tensors").
  For example, the Fisher associated with a log-prob loss w.r.t. the
  parameters.

  The 'vecs' argument to each method are lists of tensors that must be the
  size as the corresponding ones from "wrt_tensors".  They represent
  the vector being multiplied.

  "factors" of the matrix M are defined as matrices B such that B*B^T = M.
  Methods that multiply by the factor B take a 'loss_inner_vecs' argument
  instead of 'vecs', which must be a list of tensors with shapes given by the
  corresponding XXX_inner_shapes property.

  Note that matrix-vector products are not normalized by the batch size, nor
  are any damping terms added to the results.  These things can be easily
  applied externally, if desired.

  See for example: www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf
  and https://arxiv.org/abs/1412.1193 for more information about the
  generalized Gauss-Newton, Fisher, etc., and how to compute matrix-vector
  products.
  """

  def __init__(self, layer_collection, wrt_tensors,
               colocate_gradients_with_ops=True):
    """Create a CurvatureMatrixVectorProductComputer object.

    Args:
      layer_collection: A LayerCollection object where the desired loss
        functions are registered (possibly with weighing factors).
      wrt_tensors: A list of Tensors to compute the differential quantities
        (defining the matrices) with respect to.  See class description for more
        info.
      colocate_gradients_with_ops: Whether we should request gradients be
          colocated with their respective ops. (Default: True)
    """
    self._layer_collection = layer_collection
    self._wrt_tensors = wrt_tensors
    self._colocate_gradients_with_ops = colocate_gradients_with_ops

  @property
  def _loss_colocation_ops(self):
    return self._layer_collection.loss_colocation_ops

  @property
  def _losses(self):
    return self._layer_collection.losses

  @property
  def _inputs_to_losses(self):
    return list(loss.inputs for loss in self._losses)

  @property
  def _inputs_to_losses_flat(self):
    return nest.flatten(self._inputs_to_losses)

  @property
  def _total_loss(self):
    vals = []
    for loss in self._losses:
      with tf.colocate_with(self._loss_colocation_ops[loss]):
        vals.append(loss.evaluate())
    return tf.add_n(tuple(vals))

  # Jacobian multiplication functions:
  def _multiply_jacobian(self, vecs):
    """Multiply vecs by the Jacobian of losses."""
    # We stop gradients at wrt_tensors to produce partial derivatives (which is
    # what we want for Jacobians).
    jacobian_vecs_flat = utils.fwd_gradients(
        self._inputs_to_losses_flat, self._wrt_tensors, grad_xs=vecs,
        stop_gradients=self._wrt_tensors,
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    return nest.pack_sequence_as(self._inputs_to_losses, jacobian_vecs_flat)

  def _multiply_jacobian_transpose(self, loss_vecs):
    """Multiply vecs by the transpose Jacobian of losses."""
    loss_vecs_flat = nest.flatten(loss_vecs)
    # We stop gradients at wrt_tensors to produce partial derivatives (which is
    # what we want for Jacobians).
    return tf.gradients(
        self._inputs_to_losses_flat,
        self._wrt_tensors,
        grad_ys=loss_vecs_flat,
        stop_gradients=self._wrt_tensors,
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)

  # Losses Fisher/GGN multiplication functions:
  def _multiply_loss_fisher(self, loss_vecs):
    """Multiply loss_vecs by Fisher of total loss."""
    return tuple(
        loss.multiply_fisher(loss_vec)
        for loss, loss_vec in zip(self._losses, loss_vecs))

  def _multiply_across_losses(self, mult_func, vecs):
    products = []
    for loss, vec in zip(self._losses, vecs):
      with tf.colocate_with(self._loss_colocation_ops[loss]):
        products.append(mult_func(loss, vec))
    return tuple(products)

  def _multiply_loss_fisher_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of Fisher of total loss."""
    mult_func = lambda loss, vec: loss.multiply_fisher_factor(vec)
    return self._multiply_across_losses(mult_func, loss_inner_vecs)

  def _multiply_loss_fisher_factor_transpose(self, loss_vecs):
    """Multiply loss_vecs by transpose factor of Fisher of total loss."""
    mult_func = lambda loss, vec: loss.multiply_fisher_factor_transpose(vec)
    return self._multiply_across_losses(mult_func, loss_vecs)

  def _multiply_loss_ggn(self, loss_vecs):
    """Multiply loss_vecs by GGN of total loss."""
    mult_func = lambda loss, vec: loss.multiply_ggn(vec)
    return self._multiply_across_losses(mult_func, loss_vecs)

  def _multiply_loss_ggn_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of GGN of total loss."""
    mult_func = lambda loss, vec: loss.multiply_ggn_factor(vec)
    return self._multiply_across_losses(mult_func, loss_inner_vecs)

  def _multiply_loss_ggn_factor_transpose(self, loss_vecs):
    """Multiply loss_vecs by transpose factor of GGN of total loss."""
    mult_func = lambda loss, vec: loss.multiply_ggn_factor_transpose(vec)
    return self._multiply_across_losses(mult_func, loss_vecs)

  # Matrix-vector product functions:
  def multiply_fisher(self, vecs):
    """Multiply vecs by Fisher of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    loss_fisher_jacobian_vecs = self._multiply_loss_fisher(jacobian_vecs)
    return self._multiply_jacobian_transpose(loss_fisher_jacobian_vecs)

  def multiply_fisher_factor_transpose(self, vecs):
    """Multiply vecs by transpose of factor of Fisher of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    return self._multiply_loss_fisher_factor_transpose(jacobian_vecs)

  def multiply_fisher_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of Fisher of total loss."""
    fisher_factor_transpose_vecs = self._multiply_loss_fisher_factor(
        loss_inner_vecs)
    return self._multiply_jacobian_transpose(fisher_factor_transpose_vecs)

  def multiply_hessian(self, vecs):
    """Multiply vecs by Hessian of total loss."""
    return tf.gradients(
        tf.gradients(
            self._total_loss,
            self._wrt_tensors,
            colocate_gradients_with_ops=self._colocate_gradients_with_ops),
        self._wrt_tensors,
        grad_ys=vecs,
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)

  def multiply_ggn(self, vecs):
    """Multiply vecs by generalized Gauss-Newton of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    loss_ggn_jacobian_vecs = self._multiply_loss_ggn(jacobian_vecs)
    return self._multiply_jacobian_transpose(loss_ggn_jacobian_vecs)

  def multiply_ggn_factor_transpose(self, vecs):
    """Multiply vecs by transpose of factor of GGN of total loss."""
    jacobian_vecs = self._multiply_jacobian(vecs)
    return self._multiply_loss_ggn_factor_transpose(jacobian_vecs)

  def multiply_ggn_factor(self, loss_inner_vecs):
    """Multiply loss_inner_vecs by factor of GGN of total loss."""
    ggn_factor_transpose_vecs = (
        self._multiply_loss_ggn_factor(loss_inner_vecs))
    return self._multiply_jacobian_transpose(ggn_factor_transpose_vecs)

  # Shape properties for multiply_XXX_factor methods:
  @property
  def fisher_factor_inner_shapes(self):
    """Shapes required by multiply_fisher_factor."""
    return tuple(loss.fisher_factor_inner_shape for loss in self._losses)

  @property
  def fisher_factor_inner_static_shapes(self):
    """Shapes required by multiply_fisher_factor."""
    return tuple(loss.fisher_factor_inner_static_shape for loss in self._losses)

  @property
  def ggn_factor_inner_shapes(self):
    """Shapes required by multiply_generalized_gauss_newton_factor."""
    return tuple(loss.ggn_factor_inner_shape for loss in self._losses)

  @property
  def ggn_factor_inner_static_shapes(self):
    """Shapes required by multiply_generalized_gauss_newton_factor."""
    return tuple(loss.ggn_factor_inner_static_shape
                 for loss in self._losses)
