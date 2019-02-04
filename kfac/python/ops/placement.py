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
"""Implements placement strategies for various ops and cov variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import tensorflow as tf

from tensorflow.python.util import nest
from kfac.python.ops import utils as utils


def _make_thunk_on_device(func, device):
  def thunk():
    with tf.device(device):
      return func()
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


class TPURoundRobinPlacementMixin(object):
  """Implements round robin placement strategy for certain ops on replicas.

  This placement strategy can be used in certain TPU training systems, where
  there are multiple "replicas" of the graph, such as in TPUEstimator. The
  execution of inverse and transformation ops, which by default occurs
  redundantly on all replicas, are instead distributed over replicas in a round-
  robin fashion. This is achieved by using tf.cond statements to check the
  replica id number.

  This placement strategy doesn't need to be used with TPU training, and may
  not work with all possible setups (such as TF Replicator). When it does work
  however, it may provide a substantial improvement in wall-clock time.
  """

  def __init__(self, distribute_transformations=True, **kwargs):
    """Create a TPURoundRobinPlacementMixin object.

    Args:
      distribute_transformations: Bool. If True we distribute certain vector
        transformations, such as multiplication by the preconditioner, across
        different replicas. Because this is a cheaper operation it may not
        always be worth the increase communication cost to do this.
        (Default: True)
      **kwargs: Pass through arguments.
    """

    if not utils.on_tpu():
      raise ValueError("This placement mode should only be used with certain "
                       "kinds of TPU setups, such as TPUEstimator.")

    self._replica_id = utils.get_replica_id()

    # I'm assuming replicas and shards are the same thing until someone tells
    # me different
    self._num_replicas = utils.get_num_tpu_shards()

    self._distribute_transformations = distribute_transformations

    super(TPURoundRobinPlacementMixin, self).__init__(**kwargs)

  def _place_and_compute_transformation_thunks(self, thunks, params_list):
    """Computes transformation thunks with round-robin replica placement.

    Replica placement done in round-robin fashion according to the order of
    the `blocks` property, cycling through the replicas in numerical order.

    In more detail, only one replica will compute each transformation thunk,
    while the rest just compute zeros of the same shape. The results are then
    shared via utils.cross_replica_sum.

    Args:
      thunks: A list of thunks to run. Must be in one to one correspondence
        with the `blocks` property.
      params_list: A list of the corresponding parameters. Must be in one to one
        correspondence with the `blocks` property.

    Returns:
      A list (in the same order) of the returned results of the thunks, with
      round-robin replica placement applied.
    """
    if self._distribute_transformations:
      # compute_thunk computes its thunk only when self._replica_id ==
      # idx % self._num_replicas, where idx is the index of the thunk, otherwise
      # returning zeros. It then performs a cross_replica_sum on the output to
      # share the non-zero outputs with every replica, and in particular the
      # ones that didn't execute the thunk (which should be all but one of
      # them). It's inefficient insofar as it's communicating a bunch of zero
      # tensors together with the non-zero ones instead of only the latter.
      def compute_thunk(thunk, params, idx):
        def compute_zeros():
          return nest.map_structure(tf.zeros_like, params)

        return nest.map_structure(
            utils.cross_replica_sum,
            tf.cond(tf.equal(self._replica_id, idx % self._num_replicas),
                    thunk, compute_zeros, strict=True))

      return tuple(compute_thunk(thunk, params, idx)
                   for idx, (thunk, params) in enumerate(zip(thunks,
                                                             params_list)))
    else:
      return tuple(thunk() for thunk in thunks)

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

    # cross_replica_mean of the cov values is performed internally in the
    # FisherFactor classes, so we don't need to do anything for the cov updates
    # here.
    cov_update_thunks = cov_update_thunks_raw

    inv_variable_thunks = inv_variable_thunks_raw

    # The packaged inv update thunk will execute the inverse update thunk if
    # self._replica_id == idx % self._num_replicas, where idx is the index of
    # the thunk.  It will then set values_or_zeros to be (a list of) the values
    # of the corresponding inverse variables if self._replica_id ==
    # idx % self._num_replicas and zeros otherwise, and cross_replica_sum
    # each element of values_or_zeros, and finally write the result back to the
    # inverse variables. As with _place_and_compute_transformation_thunks,
    # there is a lot of needless communication happening here.
    def package_inv_update_thunk(inv_update_thunk, idx):

      def packaged_inv_update_thunk():
        maybe_update_inv = tf.cond(
            tf.equal(self._replica_id, idx % self._num_replicas),
            inv_update_thunk, tf.no_op)

        with tf.control_dependencies([maybe_update_inv]):
          inv_vars = self.factors[idx].get_inv_vars()

          if inv_vars:
            values_or_zeros = tf.cond(tf.equal(self._replica_id,
                                               idx % self._num_replicas),
                                      lambda: map(tf.identity, inv_vars),
                                      lambda: map(tf.zeros_like, inv_vars),
                                      strict=True)

            values = map(utils.cross_replica_sum, values_or_zeros)

            return tf.group(*(var.assign(val)
                              for val, var in zip(values, inv_vars)))
          else:
            return tf.no_op()

      return packaged_inv_update_thunk

    inv_update_thunks = tuple(
        package_inv_update_thunk(inv_update_thunk_raw, idx)
        for idx, inv_update_thunk_raw in enumerate(inv_update_thunks_raw))

    return (cov_variable_thunks, cov_update_thunks,
            inv_variable_thunks, inv_update_thunks)

