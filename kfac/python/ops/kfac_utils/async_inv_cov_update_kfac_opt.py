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
"""Implementation of KFAC which runs cov and inv ops asynchronously."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
# Dependency imports
import tensorflow as tf

from kfac.python.ops import optimizer

_MAX_NUM_COV_INV_UPDATE_THREADS = 10


class AsyncInvCovUpdateKfacOpt(optimizer.KfacOptimizer):
  """Provides functionality to run cov and inv ops asynchronously.

  The update ops are placed on devices in a round robin manner. These ops are
  run asynchronously in the sense that the training op and cov and inv matrix
  matrix computations are run independently of each other. The cov and inv
  ops are run in background by threads.

  Example usage:
   opt = DedicatedInvCovUpdateKfacOpt(cov_devices=["/gpu:0"],
           inv_devices=["/gpu:1"])
   train_op = opt.minimize(loss)
   with tf.Session() as sess:
     opt.run_cov_inv_ops(sess)
     for _ in range(100):
       sess.run([train_op])
     opt.stop_cov_inv_ops(sess)
  """

  def __init__(self,
               cov_devices,
               inv_devices,
               num_cov_inv_update_threads=None,
               **kwargs):
    """Initializes AsyncInvCovUpdateKfacOpt.

    See the docstring for `KfacOptimizer` class (in optimizer.py) for
    complete list of arguments (there are many!).

    Args:
      cov_devices: Iterable of device strings (e.g. '/gpu:0'). Covariance
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      inv_devices: Iterable of device strings (e.g. '/gpu:0'). Inversion
        computations will be placed on these devices in a round-robin fashion.
        Can be None, which means that no devices are specified.
      num_cov_inv_update_threads: `int`, Number of parallel computations of
        inverse and covariance ops. If a value is not passed then the number of
        threads will be set to half of length of number of ops to run
        asynchronously (Capped at `_MAX_NUM_COV_INV_UPDATE_THREADS`).
        (Default: None)
      **kwargs: Arguments to `KfacOptimizer` class.
    """
    self.next_op = None
    self._coord = None
    self._num_cov_inv_update_threads = num_cov_inv_update_threads
    self._threads = None
    super(AsyncInvCovUpdateKfacOpt, self).__init__(
        placement_strategy="round_robin", **kwargs)

  def _make_ops(self, update_thunks):
    return [thunk() for thunk in update_thunks]

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    cov_update_thunks, inv_update_thunks = self.make_vars_and_create_op_thunks()
    apply_grads = super(AsyncInvCovUpdateKfacOpt,
                        self).apply_gradients(
                            grads_and_vars=grads_and_vars,
                            global_step=global_step,
                            name=name)
    self._set_up_op_name_queue(
        self._make_ops(cov_update_thunks + inv_update_thunks))
    return apply_grads

  def run_cov_inv_ops(self, sess):
    """Starts threads to run covariance and inverse ops."""
    self._coord = tf.train.Coordinator()
    self._threads = [
        threading.Thread(target=self._run_ops, args=(
            (sess,)
        )) for _ in range(self._num_cov_inv_update_threads)
    ]
    for t in self._threads:
      t.start()

  def _run_ops(self, sess):
    """Runs the covariance and inverse ops.

    Each thread gets the next op name to run from the shared dataset that is
    created in `_set_up_op_name_queue` method. The opname is mapped to the
    op which is run in thread context.

    Args:
      sess: `tf.Session` instance.
    """
    while not self._coord.should_stop():
      next_op_name = sess.run(self._next_op_name).decode("ascii")
      next_op = self._ops_by_name[next_op_name]
      sess.run(next_op)

  def stop_cov_inv_ops(self, sess):
    """Signals coordinator to stop and waits for threads to terminate."""
    self._coord.request_stop()
    self._coord.join(self._threads)

  def _set_up_op_name_queue(self, ops_to_run):
    """Sets up a queue of op names.

    Convert the names of ops to run to tensors and creates a dataset of names.
    The op name tensors in the Dataset are repeated indefinitely. Running
    `self._next_op_name` returns the name of the next op to execute.

    Args:
      ops_to_run: `List` of ops to run asynchronously.
    """
    self._num_cov_inv_update_threads = self._num_cov_inv_update_threads or max(
        int(len(ops_to_run) / 2), _MAX_NUM_COV_INV_UPDATE_THREADS)
    self._ops_by_name = {op.name: op for op in ops_to_run}
    op_names = tf.convert_to_tensor(list(sorted(op.name for op in ops_to_run)))
    op_names_dataset = tf.data.Dataset.from_tensor_slices(op_names).repeat()
    self._next_op_name = op_names_dataset.make_one_shot_iterator().get_next()
