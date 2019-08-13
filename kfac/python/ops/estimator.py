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
    `RoundRobinPlacementMixin`, as an example.

  Raises:
    ValueError: If the `placement_strategy` argument is not one of the
    recognized options.
  """
  if placement_strategy in [None, "round_robin"]:
    return FisherEstimatorRoundRobin(**kwargs)
  elif placement_strategy == "replica_round_robin":
    return FisherEstimatorReplicaRoundRobin(**kwargs)
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
               compute_cholesky_inverse=False,
               compute_params_stats=False,
               batch_size=None):
    """Create a FisherEstimator object.

    Args:
      variables: A `list` of variables for which to estimate the Fisher. This
        must match the variables registered in layer_collection (if it is not
        None).
      cov_ema_decay: The decay factor used when calculating the covariance
        estimate moving averages.
      damping: float or 0D Tensor. This quantity times the identity matrix is
        (approximately) added to the matrix being estimated.
      layer_collection: A LayerCollection object which holds for the
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
      compute_params_stats: Bool. If True, we compute the first order version
        of the statistics computed to estimate the Fisher/GGN. These correspond
        to the `variables` method in a one-to-one fashion.  They are available
        via the `params_stats` property.  When estimation_mode is 'empirical',
        this will correspond to the standard parameter gradient on the loss.
        (Default: False)
      batch_size: The size of the mini-batch. Only needed when
        `compute_params_stats` is True. Note that when using data parallelism
        where the model graph and optimizer are replicated across multiple
        devices, this should be the per-replica batch size. An example of
        this is sharded data on the TPU, where batch_size should be set to
        the total batch size divided by the number of shards. (Default: None)

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
    self._mat_type_table = {
        "gradients": "Fisher",
        "empirical": "Empirical_Fisher",
        "curvature_prop": "Fisher",
        "curvature_prop_GGN": "GGN",
        "exact": "Fisher",
        "exact_GGN": "GGN",
    }

    self._colocate_gradients_with_ops = colocate_gradients_with_ops

    self._exps = exps
    self._compute_cholesky = compute_cholesky
    self._compute_cholesky_inverse = compute_cholesky_inverse

    self._name = name

    self._compute_params_stats = compute_params_stats
    self._batch_size = batch_size

    if compute_params_stats and batch_size is None:
      raise ValueError("Batch size needs to be passed in when "
                       "compute_params_stats is True.")

    self._finalized = False

  @property
  def variables(self):
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
    return self._layer_collection

  @property
  def mat_type(self):
    return self._mat_type_table[self._estimation_mode]

  @property
  def params_stats(self):
    return self._params_stats

  @abc.abstractmethod
  def _place_and_compute_transformation_thunks(self, thunks, params_list):
    """Computes transformation thunks with device placement.

    Device placement will be determined by the strategy asked for when this
    estimator was constructed.

    Args:
      thunks: A list of thunks to run. Must be in one to one correspondence
        with the `blocks` property.
      params_list: A list of the corresponding parameters. Must be in one to one
        correspondence with the `blocks` property.

    Returns:
      A list (in the same order) of the returned results of the thunks,
      with possible device placement applied.
    """
    pass

  def _compute_transformation(self, vecs_and_vars, transform):
    """Computes a block-wise transformation of a list of vectors.

    Args:
      vecs_and_vars: List of (vector, variable) pairs.
      transform: A function of the form f(fb, vec), that
          returns the transformed vector, where vec is the vector
          to transform and fb is its corresponding block.

    Returns:
      A list of (transformed vector, var) pairs in the same order as
      vecs_and_vars.
    """

    vecs = utils.SequenceDict((var, vec) for vec, var in vecs_and_vars)

    def make_thunk(fb, params):
      return lambda: transform(fb, vecs[params])

    thunks = tuple(make_thunk(fb, params)
                   for params, fb in self.layers.fisher_blocks.items())

    params_list = tuple(params
                        for params, _ in self.layers.fisher_blocks.items())

    results = self._place_and_compute_transformation_thunks(thunks, params_list)

    trans_vecs = utils.SequenceDict()
    for params, result in zip(self.layers.fisher_blocks, results):
      trans_vecs[params] = result

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
    return self._compute_transformation(vecs_and_vars, fcn)

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
    return self._compute_transformation(vecs_and_vars, fcn)

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
    return self._compute_transformation(vecs_and_vars, fcn)

  def _instantiate_factors(self):
    """Instantiates FisherFactors' variables.

    Raises:
      ValueError: If estimation_mode was improperly specified at construction.
    """
    blocks = self.blocks
    tensors_to_compute_grads = [
        block.tensors_to_compute_grads() for block in blocks
    ]

    if self._compute_params_stats:
      tensors_to_compute_grads = tensors_to_compute_grads + self.variables

    try:
      grads_lists = self._gradient_fns[self._estimation_mode](
          tensors_to_compute_grads)
    except KeyError:
      raise ValueError("Unrecognized value {} for estimation_mode.".format(
          self._estimation_mode))

    if any(grad is None for grad in nest.flatten(grads_lists)):
      tensors_flat = nest.flatten(tensors_to_compute_grads)
      grads_flat = nest.flatten(grads_lists)
      bad_tensors = tuple(
          tensor for tensor, grad in zip(tensors_flat, grads_flat)
          if grad is None)
      bad_string = ""
      for tensor in bad_tensors:
        bad_string += "\t{}\n".format(tensor)
      raise ValueError("It looks like you registered one of more tensors that "
                       "the registered loss/losses don't depend on. (These "
                       "returned None from tf.gradients.) The tensors were:"
                       "\n\n" + bad_string)

    if self._compute_params_stats:
      idx = len(blocks)
      params_stats_unnorm = tuple(tf.add_n(grad_list)
                                  for grad_list in grads_lists[idx:])

      scalar = 1. / tf.cast(self._batch_size,
                            dtype=params_stats_unnorm[0].dtype)
      params_stats = utils.sprod(scalar, params_stats_unnorm)

      # batch_size should be the per-replica batch size and thus we do a
      # cross-replica mean instead of a sum here
      self._params_stats = tuple(utils.all_average(tensor)
                                 for tensor in params_stats)

      grads_lists = grads_lists[:idx]

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

  def _finalize(self):
    if not self._finalized:
      self.layers.finalize()
      self.layers.check_registration(self.variables)
      self._instantiate_factors()
      self._register_matrix_functions()

    self._finalized = True

  def _check_batch_sizes(self, factor):
    """Checks that the batch size(s) for a factor matches the reference value."""

    # Should make these messages use quote characters instead of parentheses
    # when the bug with quote character rendering in assertion messages is
    # fixed. See b/129476712
    if self._batch_size is None:
      batch_size = self.factors[0].batch_size()
      string = ("Batch size {} for factor (" + factor.name + ") of type "
                + utils.cls_name(factor) + " did not match value {} used by "
                "factor (" + self.factors[0].name + ") of type "
                + utils.cls_name(self.factors[0]) + ".")
    else:
      batch_size = self._batch_size
      string = ("Batch size {} for factor (" + factor.name + ") of type "
                + utils.cls_name(factor) + " did not match value {} which was "
                "passed to optimizer/estimator.")

    factor_batch_size = factor.batch_size()

    if isinstance(batch_size, int) and isinstance(factor_batch_size, int):
      if factor_batch_size != batch_size:
        raise ValueError(string.format(factor_batch_size, batch_size))
      return factor.check_partial_batch_sizes()

    else:
      message = string.format("(x)", "(y)")
      with tf.control_dependencies([factor.check_partial_batch_sizes()]):
        return tf.assert_equal(factor_batch_size, batch_size, message=message)

  def _create_ops_and_vars_thunks(self, scope=None):
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

    self._finalize()

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

  @abc.abstractmethod
  def create_ops_and_vars_thunks(self, scope=None):
    """Create thunks that make the ops and vars on demand with device placement.

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

    Device placement will be determined by the strategy asked for when this
    estimator was constructed.

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
    pass

  def make_vars_and_create_op_thunks(self, scope=None):
    """Make vars and create op thunks with device placement.

    Similar to create_ops_and_vars_thunks but actually makes the variables
    instead of returning thunks that make them.

    Device placement will be determined by the strategy asked for when this
    estimator was constructed.

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
    (cov_variable_thunks, cov_update_thunks, inv_variable_thunks,
     inv_update_thunks) = self.create_ops_and_vars_thunks(scope=scope)

    for thunk in cov_variable_thunks:
      thunk()

    for thunk in inv_variable_thunks:
      thunk()

    return cov_update_thunks, inv_update_thunks

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

    def thunk(should_write=True,
              ema_decay=None,
              ema_weight=None):

      if ema_decay is None:
        ema_decay = self._cov_ema_decay

      if ema_weight is None:
        ema_weight = 1.0 - self._cov_ema_decay

      with tf.variable_scope(scope):
        with tf.control_dependencies([self._check_batch_sizes(factor)]):
          return factor.make_covariance_update_op(
              ema_decay, ema_weight, should_write=should_write)

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

  def _get_transformed_random_signs(self):
    if self.mat_type == "Fisher":
      mult_func = lambda loss, index: loss.multiply_fisher_factor(index)
      inner_shape_func = lambda loss: loss.fisher_factor_inner_shape
    elif self.mat_type == "GGN":
      mult_func = lambda loss, index: loss.multiply_ggn_factor(index)
      inner_shape_func = lambda loss: loss.ggn_factor_inner_shape

    transformed_random_signs = []
    for loss in self.layers.losses:
      with tf.colocate_with(self.layers.loss_colocation_ops[loss]):
        value = mult_func(loss,
                          utils.generate_random_signs(inner_shape_func(loss)))
        coeff = tf.cast(self.layers.loss_coeffs[loss], dtype=value.dtype)
        transformed_random_signs.append(tf.sqrt(coeff) * value)
    return transformed_random_signs

  def _get_grads_lists_curvature_prop(self, tensors):
    loss_inputs = list(loss.inputs for loss in self.layers.losses)
    transformed_random_signs = self._get_transformed_random_signs()
    grads_flat = tf.gradients(
        nest.flatten(loss_inputs),
        nest.flatten(tensors),
        grad_ys=nest.flatten(transformed_random_signs),
        colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    grads_all = nest.pack_sequence_as(tensors, grads_flat)
    return tuple((grad,) for grad in grads_all)

  def _get_grads_lists_exact(self, tensors):
    if self.mat_type == "Fisher":
      # pylint: disable=g-long-lambda
      mult_func = (lambda loss, index:
                   loss.multiply_fisher_factor_replicated_one_hot(index))
      inner_shape_func = lambda loss: loss.fisher_factor_inner_static_shape
    elif self.mat_type == "GGN":
      # pylint: disable=g-long-lambda
      mult_func = (lambda loss, index:
                   loss.multiply_ggn_factor_replicated_one_hot(index))
      inner_shape_func = lambda loss: loss.fisher_ggn_inner_static_shape

    # Loop over all coordinates of all losses.
    grads_all = []
    for loss in self.layers.losses:
      with tf.colocate_with(self.layers.loss_colocation_ops[loss]):
        for index in np.ndindex(*inner_shape_func(loss)[1:]):
          value = mult_func(loss, index)
          coeff = tf.cast(self.layers.loss_coeffs[loss], dtype=value.dtype)
          transformed_one_hot = tf.sqrt(coeff) * value
          grads_flat = tf.gradients(
              loss.inputs,
              nest.flatten(tensors),
              grad_ys=transformed_one_hot,
              colocate_gradients_with_ops=self._colocate_gradients_with_ops)
          grads_all.append(nest.pack_sequence_as(tensors, grads_flat))
    return tuple(zip(*grads_all))


class FisherEstimatorRoundRobin(placement.RoundRobinPlacementMixin,
                                FisherEstimator):
  """FisherEstimator which provides round robin device placement strategy."""
  pass


class FisherEstimatorReplicaRoundRobin(
    placement.ReplicaRoundRobinPlacementMixin,
    FisherEstimator):
  """FisherEstimator which provides round robin replica placement strategy."""
  pass
