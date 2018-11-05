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
"""Defines the high-level Fisher estimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

# Dependency imports
import numpy as np
import six
import tensorflow as tf

from tensorflow.python.util import nest
from kfac.python.ops import placement
from kfac.python.ops import utils


# The linter is confused.
# pylint: disable=abstract-class-instantiated
def make_fisher_estimator(placement_strategy=None, **kwargs):
  """Creates Fisher estimator instances based on the placement strategy.

  For example if the `placement_strategy` is 'round_robin' then
  `FisherEstimatorRoundRobin` instance is returned.

  Args:
    placement_strategy: `string`, Strategy to be used for placing covariance
      variables, covariance ops and inverse ops. Check
      `placement.FisherEstimatorRoundRobin` for a concrete example.
   **kwargs: Arguments to be passed into `FisherEstimator` class initializer.

  Returns:
    An instance of class which inherits from `FisherEstimator` and the mixin
    which implements specific placement strategy. See,
    `FisherEstimatorRoundRobin` which inherits from `FisherEstimator` and
    `RoundRobinPlacementMixin`.

  Raises:
    ValueError: If the `placement_strategy` is not equal to 'round_robin'.
  """
  if placement_strategy in [None, "round_robin"]:
    return FisherEstimatorRoundRobin(**kwargs)
  else:
    raise ValueError(
        "Unimplemented vars and ops placement strategy : {}".format(
            placement_strategy))
# pylint: enable=abstract-class-instantiated


@six.add_metaclass(abc.ABCMeta)
class FisherEstimator(object):
  """Fisher estimator class supporting various approximations of the Fisher.

  This is an abstract base class which does not implement a strategy for
  placing covariance variables, covariance update ops and inverse update ops.
  The placement strategies are implemented in `placement.py`. See
  `FisherEstimatorRoundRobin` for example of a concrete subclass with
  a round-robin placement strategy.
  """

  def __init__(self,
               variables,
               cov_ema_decay,
               damping,
               layer_collection,
               exps=(-1,),
               estimation_mode="gradients",
               colocate_gradients_with_ops=True,
               name="FisherEstimator",
               compute_cholesky=False,
               compute_cholesky_inverse=False):
    """Create a FisherEstimator object.

    Args:
      variables: A `list` of variables or `callable` which returns the variables
          for which to estimate the Fisher. This must match the variables
          registered in layer_collection (if it is not None).
      cov_ema_decay: The decay factor used when calculating the covariance
          estimate moving averages.
      damping: float. The damping factor used to stabilize training due to
          errors in the local approximation with the Fisher information matrix,
          and to regularize the update direction by making it closer to the
          gradient. (Higher damping means the update looks more like a standard
          gradient update - see Tikhonov regularization.)
      layer_collection: Either layer collection object or a function which
          returns an instance to `LayerCollection` object, which holds for the
          Fisher blocks, Kronecker factors, and losses associated with the
          graph.
      exps: List of floats or ints. These represent the different matrix
          powers of the approximate Fisher that the FisherEstimator will be able
          to multiply vectors by. If the user asks for a matrix power other
          one of these (or 1, which is always supported), there will be a
          failure. (Default: (-1,))
      estimation_mode: The type of estimator to use for the Fishers/GGNs. Can be
          'gradients', 'empirical', 'curvature_prop', 'curvature_prop_GGN',
          'exact', or 'exact_GGN'. (Default: 'gradients'). 'gradients' is the
          basic estimation approach from the original K-FAC paper.
          'empirical' computes the 'empirical' Fisher information matrix (which
          uses the data's distribution for the targets, as opposed to the true
          Fisher which uses the model's distribution) and requires that each
          registered loss have specified targets. 'curvature_propagation' is a
          method which estimates the Fisher using self-products of random 1/-1
          vectors times "half-factors" of the Fisher, as described here:
          https://arxiv.org/abs/1206.6464 . 'exact' is the obvious
          generalization of Curvature Propagation to compute the exact Fisher
          (modulo any additional diagonal or Kronecker approximations) by
          looping over one-hot vectors for each coordinate of the output
          instead of using 1/-1 vectors.  It is more expensive to compute than
          the other three options by a factor equal to the output dimension,
          roughly speaking. Finally, 'curvature_prop_GGN' and 'exact_GGN' are
          analogous to 'curvature_prop' and 'exact', but estimate the
          Generalized Gauss-Newton matrix (GGN).
      colocate_gradients_with_ops: Whether we should request gradients be
          colocated with their respective ops. (Default: True)
      name: A string. A name given to this estimator, which is added to the
          variable scope when constructing variables and ops.
          (Default: "FisherEstimator")
      compute_cholesky: Bool. Whether or not the FisherEstimator will be
          able to multiply vectors by the Cholesky factor.
          (Default: False)
      compute_cholesky_inverse: Bool. Whether or not the FisherEstimator
          will be able to multiply vectors by the Cholesky factor inverse.
          (Default: False)
    Raises:
      ValueError: If no losses have been registered with layer_collection.
    """
    self._variables = variables
    self._cov_ema_decay = cov_ema_decay
    self._damping = damping
    self._estimation_mode = estimation_mode
    self._layer_collection = layer_collection
    self._gradient_fns = {
        "gradients": self._get_grads_lists_gradients,
        "empirical": self._get_grads_lists_empirical,
        "curvature_prop": self._get_grads_lists_curvature_prop,
        "curvature_prop_GGN": self._get_grads_lists_curvature_prop,
        "exact": self._get_grads_lists_exact,
        "exact_GGN": self._get_grads_lists_exact
    }
    self._gradient_fns_extra_args = {
        "gradients": {},
        "empirical": {},
        "curvature_prop": {"mode": "fisher"},
        "curvature_prop_GGN": {"mode": "GGN"},
        "exact": {"mode": "fisher"},
        "exact_GGN": {"mode": "GGN"},
    }
    self._colocate_gradients_with_ops = colocate_gradients_with_ops

    self._exps = exps
    self._compute_cholesky = compute_cholesky
    self._compute_cholesky_inverse = compute_cholesky_inverse

    self._name = name

  @property
  def variables(self):
    if callable(self._variables):
      return self._variables()
    else:
      return self._variables

  @property
  def damping(self):
    return self._damping

  @property
  def blocks(self):
    """All registered FisherBlocks."""
    return self.layers.get_blocks()

  @property
  def factors(self):
    """All registered FisherFactors."""
    return self.layers.get_factors()

  @property
  def name(self):
    return self._name

  @property
  def layers(self):
    if callable(self._layer_collection):
      return self._layer_collection()
    else:
      return self._layer_collection

  @abc.abstractmethod
  def make_vars_and_create_op_thunks(self, scope=None):
    """Make vars and create op thunks with a specific placement strategy.

    For each factor, all of that factor's cov variables and their associated
    update ops will be placed on a particular device.  A new device is chosen
    for each factor by cycling through list of devices in the cov_devices
    argument. If cov_devices is None then no explicit device placement occurs.

    An analogous strategy is followed for inverse update ops, with the list of
    devices being given by the inv_devices argument.

    Inverse variables on the other hand are not placed on any specific device
    (they will just use the current the device placement context, whatever
    that happens to be).  The idea is that the inverse variable belong where
    they will be accessed most often, which is the device that actually applies
    the preconditioner to the gradient. The user will be responsible for setting
    the device context for this.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all thunks will execute, inside of a variable scope of the given
        name. (Default: None)

    Returns:
      cov_update_thunks: List of cov update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
      inv_update_thunks: List of inv update thunks. Corresponds one-to-one with
        the list of factors given by the "factors" property.
    """
    pass

  def _apply_transformation(self, vecs_and_vars, transform):
    """Applies an block-wise transformation to the corresponding vectors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      transform: A function of the form f(fb, vec), where vec is the vector
          to transform and fb is its corresponding block in the matrix, that
          returns the transformed vector.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    vecs = utils.SequenceDict((var, vec) for vec, var in vecs_and_vars)

    trans_vecs = utils.SequenceDict()

    for params, fb in self.layers.fisher_blocks.items():
      trans_vecs[params] = transform(fb, vecs[params])

    return [(trans_vecs[var], var) for _, var in vecs_and_vars]

  def multiply_inverse(self, vecs_and_vars):
    """Multiplies the vecs by the corresponding (damped) inverses of the blocks.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """
    return self.multiply_matpower(-1, vecs_and_vars)

  def multiply(self, vecs_and_vars):
    """Multiplies the vectors by the corresponding (damped) blocks.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """
    return self.multiply_matpower(1, vecs_and_vars)

  def multiply_matpower(self, exp, vecs_and_vars):
    """Multiplies the vecs by the corresponding matrix powers of the blocks.

    Args:
      exp: A float representing the power to raise the blocks by before
        multiplying it by the vector.
      vecs_and_vars: List of (vector, variable) pairs.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """
    fcn = lambda fb, vec: fb.multiply_matpower(vec, exp)
    return self._apply_transformation(vecs_and_vars, fcn)

  def multiply_cholesky(self, vecs_and_vars, transpose=False):
    """Multiplies the vecs by the corresponding Cholesky factors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      transpose: Bool. If true the Cholesky factors are transposed before
        multiplying the vecs. (Default: False)

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    fcn = lambda fb, vec: fb.multiply_cholesky(vec, transpose=transpose)
    return self._apply_transformation(vecs_and_vars, fcn)

  def multiply_cholesky_inverse(self, vecs_and_vars, transpose=False):
    """Mults the vecs by the inverses of the corresponding Cholesky factors.

      Note: if you are using Cholesky inverse multiplication to sample from
      a matrix-variate Gaussian you will want to multiply by the transpose.
      Let L be the Cholesky factor of F and observe that

        L^-T * L^-1 = (L * L^T)^-1 = F^-1 .

      Thus we want to multiply by L^-T in order to sample from Gaussian with
      covariance F^-1.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      transpose: Bool. If true the Cholesky factor inverses are transposed
        before multiplying the vecs. (Default: False)

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    fcn = lambda fb, vec: fb.multiply_cholesky_inverse(vec, transpose=transpose)
    return self._apply_transformation(vecs_and_vars, fcn)

  def _instantiate_factors(self):
    """Instantiates FisherFactors' variables.

    Raises:
      ValueError: If estimation_mode was improperly specified at construction.
    """
    blocks = self.blocks
    tensors_to_compute_grads = [
        block.tensors_to_compute_grads() for block in blocks
    ]

    try:
      grads_lists = self._gradient_fns[self._estimation_mode](
          tensors_to_compute_grads,
          **self._gradient_fns_extra_args[self._estimation_mode])
    except KeyError:
      raise ValueError("Unrecognized value {} for estimation_mode.".format(
          self._estimation_mode))

    for grads_list, block in zip(grads_lists, blocks):
      block.instantiate_factors(grads_list, self.damping)

  def _register_matrix_functions(self):
    for block in self.blocks:
      for exp in self._exps:
        block.register_matpower(exp)
      if self._compute_cholesky:
        block.register_cholesky()
      if self._compute_cholesky_inverse:
        block.register_cholesky_inverse()

  def _finalize_layer_collection(self):
    self.layers.create_subgraph()
    self.layers.check_registration(self.variables)
    self._instantiate_factors()
    self._register_matrix_functions()

  def create_ops_and_vars_thunks(self, scope=None):
    """Create thunks that make the ops and vars on demand.

    This function returns 4 lists of thunks: cov_variable_thunks,
    cov_update_thunks, inv_variable_thunks, and inv_update_thunks.

    The length of each list is the number of factors and the i-th element of
    each list corresponds to the i-th factor (given by the "factors" property).

    Note that the execution of these thunks must happen in a certain
    partial order.  The i-th element of cov_variable_thunks must execute
    before the i-th element of cov_update_thunks (and also the i-th element
    of inv_update_thunks).  Similarly, the i-th element of inv_variable_thunks
    must execute before the i-th element of inv_update_thunks.

    TL;DR (oversimplified): Execute the thunks according to the order that
    they are returned.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All thunks will execute inside
        of a variable scope of the given name. (Default: None)
    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """

    self._finalize_layer_collection()

    scope = self.name if scope is None else scope

    cov_variable_thunks = [
        self._create_cov_variable_thunk(factor, scope)
        for factor in self.factors
    ]
    cov_update_thunks = [
        self._create_cov_update_thunk(factor, scope) for factor in self.factors
    ]
    inv_variable_thunks = [
        self._create_inv_variable_thunk(factor, scope)
        for factor in self.factors
    ]
    inv_update_thunks = [
        self._create_inv_update_thunk(factor, scope) for factor in self.factors
    ]

    return (cov_variable_thunks, cov_update_thunks,
            inv_variable_thunks, inv_update_thunks)

  def get_cov_vars(self):
    """Returns all covariance variables associated with each Fisher factor.

    Note the returned list also includes additional factor specific covariance
    variables.

    Returns: List of list. The number of inner lists is equal to number of
      factors. And each inner list contains all covariance
      variables for that factor.
    """
    return tuple(factor.get_cov_vars() for factor in self.factors)

  def get_inv_vars(self):
    """Returns all covariance variables associated with each Fisher factor.

    Note the returned list also includes additional factor specific covariance
    variables.

    Returns: List of list. The number of inner lists is equal to number of
      factors. And each inner list contains all inverse computation related
      variables for that factor.
    """
    return tuple(factor.get_inv_vars() for factor in self.factors)

  def _create_cov_variable_thunk(self, factor, scope):
    """Constructs a covariance variable thunk for a single FisherFactor."""

    def thunk():
      with tf.variable_scope(scope):
        return factor.instantiate_cov_variables()

    return thunk

  def _create_cov_update_thunk(self, factor, scope):
    """Constructs a covariance update thunk for a single FisherFactor."""

    def thunk():
      with tf.variable_scope(scope):
        return factor.make_covariance_update_op(self._cov_ema_decay)

    return thunk

  def _create_inv_variable_thunk(self, factor, scope):
    """Constructs a inverse variable thunk for a single FisherFactor."""

    def thunk():
      with tf.variable_scope(scope):
        return factor.instantiate_inv_variables()

    return thunk

  def _create_inv_update_thunk(self, factor, scope):
    """Constructs an inverse update thunk for a single FisherFactor."""

    def thunk():
      with tf.variable_scope(scope):
        return tf.group(factor.make_inverse_update_ops())

    return thunk

  def _get_grads_lists_gradients(self, tensors):
    # Passing in a list of loss values is better than passing in the sum as
    # the latter creates unnecessary ops on the default device
    grads_flat = tf.gradients(
        self.layers.eval_losses(target_mode="sample", coeff_mode="sqrt"),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_empirical(self, tensors):
    # Passing in a list of loss values is better than passing in the sum as
    # the latter creates unnessesary ops on the default device
    grads_flat = tf.gradients(
        self.layers.eval_losses(target_mode="data", coeff_mode="regular"),
        nest.flatten(tensors),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_transformed_random_signs(self, mode="fisher"):
    """No docstring required."""
    if mode == "fisher":
      mult_func = lambda loss, index: loss.multiply_fisher_factor(index)
      inner_shape_func = lambda loss: loss.fisher_factor_inner_shape
    elif mode == "GGN":
      mult_func = lambda loss, index: loss.multiply_hessian_factor(index)
      inner_shape_func = lambda loss: loss.hessian_factor_inner_shape

    transformed_random_signs = []
    for loss in self.layers.losses:
      with tf.colocate_with(self.layers.loss_colocation_ops[loss]):
        transformed_random_signs.append(
            tf.sqrt(self.layers.loss_coeffs[loss])*mult_func(
                loss, utils.generate_random_signs(inner_shape_func(loss))))
    return transformed_random_signs

  def _get_grads_lists_curvature_prop(self, tensors, mode="fisher"):
    loss_inputs = list(loss.inputs for loss in self.layers.losses)
    transformed_random_signs = self._get_transformed_random_signs(mode=mode)
    grads_flat = tf.gradients(
        nest.flatten(loss_inputs),
        nest.flatten(tensors),
        grad_ys=nest.flatten(transformed_random_signs),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_exact(self, tensors, mode="fisher"):
    """No docstring required."""
    if mode == "fisher":
      # pylint: disable=g-long-lambda
      mult_func = (lambda loss, index:
                   loss.multiply_fisher_factor_replicated_one_hot(index))
      inner_shape_func = lambda loss: loss.fisher_factor_inner_static_shape
    elif mode == "GGN":
      # pylint: disable=g-long-lambda
      mult_func = (lambda loss, index:
                   loss.multiply_hessian_factor_replicated_one_hot(index))
      inner_shape_func = lambda loss: loss.fisher_hessian_inner_static_shape

    # Loop over all coordinates of all losses.
    grads_all = []
    for loss in self.layers.losses:
      with tf.colocate_with(self.layers.loss_colocation_ops[loss]):
        for index in np.ndindex(*inner_shape_func(loss)[1:]):
          transformed_one_hot = (tf.sqrt(self.layers.loss_coeffs[loss]) *
                                 mult_func(loss, index))
          grads_flat = tf.gradients(
              loss.inputs,
              nest.flatten(tensors),
              grad_ys=transformed_one_hot,
              colocate_gradients_with_ops=self._colocate_gradients_with_ops)
          grads_all.append(nest.pack_sequence_as(tensors, grads_flat))
    return zip(*grads_all)


class FisherEstimatorRoundRobin(placement.RoundRobinPlacementMixin,
                                FisherEstimator):
  """Fisher estimator which provides round robin device placement strategy."""
  pass
