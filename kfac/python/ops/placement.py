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
"""Implements placement strategies for various ops and variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import tensorflow as tf

from tensorflow.python.util import nest
from kfac.python.ops import utils as utils


def _make_thunk_on_device(func, device):
  def thunk(*args, **kwargs):
    with tf.device(device):
      return func(*args, **kwargs)
  return thunk


class RoundRobinPlacementMixin(object):
  """Implements round robin placement strategy for ops and variables."""

  def __init__(self, cov_devices=None, inv_devices=None, trans_devices=None,
               **kwargs):
    """Create a RoundRobinPlacementMixin object.

    Args:
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
        computations will be placed on these devices in a round-robin fashion.
        Can be None or empty, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
        computations will be placed on these devices in a round-robin fashion.
        Can be None or empty, which means that no devices are specified.
      trans_devices: Iterable of device strings (e.g. '/gpu:0'). Transformation
        computations (e.g. multiplying different blocks by the inverse Fisher)
        will be placed on these devices in a round-robin fashion. Can be None
        or empty, which means that no devices are specified.
      **kwargs: Pass through arguments.
    """
    super(RoundRobinPlacementMixin, self).__init__(**kwargs)
    self._cov_devices = cov_devices
    self._inv_devices = inv_devices
    self._trans_devices = trans_devices

  def _place_and_compute_transformation_thunks(self, thunks, params_list):
    """Computes transformation thunks with round-robin device placement.

    Device placement done in round-robin fashion according to the order of
    the `blocks` property, using the list `trans_devices` passed in to the
    constructor.

    Args:
      thunks: A list of thunks to run. Must be in one to one correspondence
        with the `blocks` property.
      params_list: A list of the corresponding parameters. Must be in one to one
        correspondence with the `blocks` property.

    Returns:
      A list (in the same order) of the returned results of the thunks, with
      round-robin device placement applied.
    """
    del params_list

    if self._trans_devices:
      results = []
      for thunk, device in zip(thunks, itertools.cycle(self._trans_devices)):
        with tf.device(device):
          results.append(thunk())
      return results
    else:
      return tuple(thunk() for thunk in thunks)

  def create_ops_and_vars_thunks(self, scope=None):
    """Create thunks that make the ops and vars on demand with device placement.

    For each factor, all of that factor's cov variables and their associated
    update ops will be placed on a particular device.  A new device is chosen
    for each factor by cycling through list of devices in the
    `self._cov_devices` attribute. If `self._cov_devices` is `None` then no
    explicit device placement occurs.

    An analogous strategy is followed for inverse update ops, with the list of
    devices being given by the `self._inv_devices` attribute.

    Inverse variables on the other hand are not placed on any specific device
    (they will just use the current the device placement context, whatever
    that happens to be).  The idea is that the inverse variable belong where
    they will be accessed most often, which is the device that actually applies
    the preconditioner to the gradient. The user will be responsible for setting
    the device context for this.

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
        estimator (given by the name property). All variables will be created,
        and all thunks will execute, inside of a variable scope of the given
        name. (Default: None)

    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """
    (cov_variable_thunks_raw, cov_update_thunks_raw, inv_variable_thunks_raw,
     inv_update_thunks_raw) = self._create_ops_and_vars_thunks(scope=scope)

    if self._cov_devices:
      cov_variable_thunks = []
      cov_update_thunks = []
      for cov_variable_thunk, cov_update_thunk, device in zip(
          cov_variable_thunks_raw, cov_update_thunks_raw,
          itertools.cycle(self._cov_devices)):

        cov_variable_thunks.append(_make_thunk_on_device(cov_variable_thunk,
                                                         device))
        cov_update_thunks.append(_make_thunk_on_device(cov_update_thunk,
                                                       device))
    else:
      cov_variable_thunks = cov_variable_thunks_raw
      cov_update_thunks = cov_update_thunks_raw

    inv_variable_thunks = inv_variable_thunks_raw

    if self._inv_devices:
      inv_update_thunks = []
      for inv_update_thunk, device in zip(inv_update_thunks_raw,
                                          itertools.cycle(self._inv_devices)):
        inv_update_thunks.append(_make_thunk_on_device(inv_update_thunk,
                                                       device))
    else:
      inv_update_thunks = inv_update_thunks_raw

    return (cov_variable_thunks, cov_update_thunks,
            inv_variable_thunks, inv_update_thunks)


class ReplicaRoundRobinPlacementMixin(object):
  """Implements round robin placement strategy for certain ops on replicas.

  This placement strategy can be used in certain TPU training systems, where
  there are multiple "replicas" of the graph, such as in TPUEstimator or
  TF-Replicator. The execution of inverse and transformation ops, which by
  default occurs redundantly on all replicas, are instead distributed over
  replicas in a round-robin fashion. This is achieved by using tf.cond
  statements to check the replica id number.

  This placement strategy doesn't need to be used with TPU training, and may
  not work with all possible setups (such as TF Replicator). When it does work
  however, it may provide a substantial improvement in wall-clock time.
  """

  def __init__(self, distribute_transformations=True, **kwargs):
    """Create a ReplicaRoundRobinPlacementMixin object.

    Args:
      distribute_transformations: Bool. If True we distribute certain vector
        transformations, such as multiplication by the preconditioner, across
        different replicas. Because this is a cheaper operation it may not
        always be worth the increase communication cost to do this.
        (Default: True)
      **kwargs: Pass through arguments.
    """

    if not utils.is_replicated():
      raise ValueError("This placement mode should only be used with certain "
                       "kinds of 'replicated' setups, such as TPUEstimator "
                       "or TF-Replicator.")

    self._distribute_transformations = distribute_transformations

    super(ReplicaRoundRobinPlacementMixin, self).__init__(**kwargs)

  def _place_and_compute_transformation_thunks(self, thunks, params_list):
    """Computes transformation thunks with round-robin replica placement.

    Replica placement done in round-robin fashion according to the order of
    the `blocks` property, cycling through the replicas in numerical order.

    Args:
      thunks: A list of thunks to run. Must be in one to one correspondence
        with the `blocks` property.
      params_list: A list of the corresponding parameters. Must be in one to one
        correspondence with the `blocks` property.

    Returns:
      A list (in the same order) of the returned results of the thunks, with
      round-robin replica placement applied.
    """
    del params_list

    return utils.map_gather(thunks)

  def create_ops_and_vars_thunks(self, scope=None):
    """Create op/var-making thunks with replica placement for inverse ops.

    For each factor in the list of factors, the associated inverse ops will
    execute on a single replica which is chosen in round-robin fashion.

    Cov ops are run on all replicas, with the appropriate averaging done by
    using a few cross_replica_mean's that have been injected into the
    FisherFactor classes (and execute regardless if this mixin is being used).

    This function returns 4 lists of thunks: cov_variable_thunks,
    cov_update_thunks, inv_variable_thunks, and inv_update_thunks.

    The length of each list is the number of factors and the i-th element of
    each list corresponds to the i-th factor (given by the "factors" property).
    (Actually, for inv_update_thunks this class in particular returns only one
    thunk inside inv_update_thunks that updates all the factors.)

    Note that the execution of these thunks must happen in a certain
    partial order.  The i-th element of cov_variable_thunks must execute
    before the i-th element of cov_update_thunks (and also the i-th element
    of inv_update_thunks).  Similarly, the i-th element of inv_variable_thunks
    must execute before the i-th element of inv_update_thunks.

    TL;DR (oversimplified): Execute the thunks according to the order that
    they are returned.

    Args:
      scope: A string or None.  If None it will be set to the name of this
        estimator (given by the name property). All variables will be created,
        and all thunks will execute, inside of a variable scope of the given
        name. (Default: None)

    Returns:
      cov_variable_thunks: A list of thunks that make the cov variables.
      cov_update_thunks: A list of thunks that make the cov update ops.
      inv_variable_thunks: A list of thunks that make the inv variables.
      inv_update_thunks: A list of thunks that make the inv update ops.
    """

    (cov_variable_thunks_raw, cov_update_thunks_raw, inv_variable_thunks_raw,
     inv_update_thunks_raw) = self._create_ops_and_vars_thunks(scope=scope)

    cov_variable_thunks = cov_variable_thunks_raw

    # all_averages of cov values are performed internally in the FisherFactor
    # classes, so we don't need to do anything for the cov updates here.
    cov_update_thunks = cov_update_thunks_raw
    inv_variable_thunks = inv_variable_thunks_raw

    # The thunks made here execute the supplied inverse update thunk and then
    # retrieve the values from the corresponding inverse variables.
    def make_thunk(inv_update_thunk, inv_vars):
      def thunk():
        with tf.control_dependencies([inv_update_thunk()]):
          return nest.map_structure(tf.identity, inv_vars)
      return thunk

    # This single thunk calls map_gather to distribute the work, and then
    # saves the results back to the corresponding inverse variables.
    def inv_update_thunk():

      assert len(inv_update_thunks_raw) == len(self.factors)

      # Create a list of factors and thunks that only include the factors
      # that have inverse variables.  Note that not executing the inverse ops of
      # those that don't shouldn't matter.
      factors_and_thunks = tuple(
          (factor, thunk)
          for factor, thunk in zip(self.factors, inv_update_thunks_raw)
          if factor.get_inv_vars())

      factors, _ = zip(*factors_and_thunks)

      thunks = tuple(
          make_thunk(inv_update_thunk, factor.get_inv_vars())
          for factor, inv_update_thunk in factors_and_thunks)

      results = utils.map_gather(thunks)

      # These assigns save the values back to the variables.
      ops = (utils.smart_assign(var, val)
             for factor, result in zip(factors, results)  # pylint: disable=g-complex-comprehension
             for val, var in zip(result, factor.get_inv_vars()))
      return tf.group(*ops)

    # Note that we have to return one big inv_update_thunk instead of one for
    # each factor. This is because utils.map_gather doesn't support returning
    # thunks (because TFReplicator's map_gather doesn't).
    inv_update_thunks = [inv_update_thunk]

    return (cov_variable_thunks, cov_update_thunks,
            inv_variable_thunks, inv_update_thunks)

