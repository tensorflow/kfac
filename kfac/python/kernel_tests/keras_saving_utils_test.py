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
#,============================================================================
"""Tests for keras/saving_utils.py.

These tests were forked from the hdf5_format_test.py tests in Keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import test_util
from kfac.python.keras import optimizers
from kfac.python.keras import saving_utils

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

keras = tf.keras
_KFAC_KWARGS = {
    'learning_rate': 0.0001,
    'damping': 0.01,
    'momentum': 0.85,
    'fisher_approx': {
        keras.layers.Dense: 'kron_in_diag',
    },
    'loss': 'mse',
    # This seed is necessary to keep the optimizer updates deterministic, since
    # we're approximating the true Fisher by sampling the targets. Since for
    # many tests we only do one training step, the approximations can vary
    # significantly without a set seed.
    'seed': 1234,
}


class SavingUtilsTest(tf.test.TestCase):

  @test_util.run_v1_only('b/120994067')
  def test_sequential_model_saving(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(2,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(3))
      model.compile(
          loss=keras.losses.MSE,
          optimizer=optimizers.Kfac(model=model, **_KFAC_KWARGS),
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy()
          ])

      x = np.random.random((1, 2))
      y = np.random.random((1, 3))

      # TODO(b/136561651): Since we use TFP distributions to sample from the
      # output distribution, optimizer's won't match exactly unless they are run
      # for the same number of steps. Even with a random seed, the internal
      # state of TFP changes with each call. We must switch to a stateless
      # sampler. Uncomment the train line below once this is implemented.
      # model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      new_model = saving_utils.load_model(fname, optimizer_name='new')
      os.close(fd)
      os.remove(fname)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # test that new updates are the same with both models
      x = np.random.random((1, 2))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)
      new_model.train_on_batch(x, y)

      x = np.random.random((1, 2))
      y = np.random.random((1, 3))
      eval_out = model.evaluate(x, y)
      eval_out2 = new_model.evaluate(x, y)
      self.assertArrayNear(eval_out, eval_out2, 1e-03)

      out = model.predict(x)
      out2 = new_model.predict(x)

      self.assertAllClose(out, out2, atol=1e-05)

  @test_util.run_deprecated_v1
  def test_functional_model_saving(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(
          loss=keras.losses.MSE,
          optimizer=optimizers.Kfac(model=model, **_KFAC_KWARGS),
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy()
          ],
          weighted_metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy()
          ])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = saving_utils.load_model(fname, optimizer_name='new')
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_model_with_long_layer_names(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      # This layer name will make the `layers_name` HDF5 attribute blow
      # out of proportion. Note that it fits into the internal HDF5
      # attribute memory limit on its own but because h5py converts
      # the list of layer names into numpy array, which uses the same
      # amount of memory for every item, it increases the memory
      # requirements substantially.
      x = keras.Input(shape=(2,), name='input_' + ('x' * (2**15)))
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='dense_%d' % (i,))(f)
      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(optimizers.Kfac(model=model, **_KFAC_KWARGS),
                    loss=keras.losses.MeanSquaredError(),
                    metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = saving_utils.load_model(fname, optimizer_name='new')

      # Check that the HDF5 files contains chunked array
      # of layer names.
      with h5py.File(fname, 'r') as h5file:
        num_names_arrays = len([attr for attr in h5file['model_weights'].attrs
                                if attr.startswith('layer_names')])
      # The chunking of layer names array should have happened.
      self.assertGreater(num_names_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_saving_model_with_long_weights_names(self):
    self.skipTest('KFAC does not support nested models yet.')
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      x = keras.Input(shape=(2,), name='nested_model_input')
      f = x
      for i in range(4):
        f = keras.layers.Dense(2, name='nested_model_dense_%d' % (i,))(f)
      # This layer name will make the `weights_name`
      # HDF5 attribute blow out of proportion.
      f = keras.layers.Dense(2, name='nested_model_output' + ('x' * (2**14)))(f)
      nested_model = keras.Model(inputs=[x], outputs=[f], name='nested_model')

      x = keras.Input(shape=(2,), name='outer_model_input')
      f = nested_model(x)
      f = keras.layers.Dense(2, name='outer_model_output')(f)

      model = keras.Model(inputs=[x], outputs=[f])
      model.compile(loss='mse',
                    optimizer=optimizers.Kfac(model=model, **_KFAC_KWARGS),
                    metrics=['acc'])

      x = np.random.random((1, 2))
      y = np.random.random((1, 2))
      model.train_on_batch(x, y)
      out = model.predict(x)

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = saving_utils.load_model(fname, optimizer_name='new')

      # Check that the HDF5 files contains chunked array
      # of weight names.
      with h5py.File(fname, 'r') as h5file:
        num_weight_arrays = len(
            [attr for attr in h5file['model_weights']['nested_model'].attrs
             if attr.startswith('weight_names')])
      # The chunking of layer names array should have happened.
      self.assertGreater(num_weight_arrays, 0)
      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  @test_util.run_deprecated_v1
  def test_model_saving_to_pre_created_h5py_file(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      inputs = keras.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      outputs = keras.layers.Dense(3)(x)

      model = keras.Model(inputs, outputs)
      model.compile(
          loss=keras.losses.MSE,
          optimizer=optimizers.Kfac(model=model, **_KFAC_KWARGS),
          metrics=[
              keras.metrics.categorical_accuracy,
              keras.metrics.CategoricalAccuracy()
          ])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      fd, fname = tempfile.mkstemp('.h5')
      with h5py.File(fname, mode='r+') as h5file:
        keras.models.save_model(model, h5file)
        loaded_model = saving_utils.load_model(h5file, optimizer_name='new')
        out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Test non-default options in h5
      with h5py.File(
          '-', driver='core', mode='w', backing_store=False) as h5file:
        keras.models.save_model(model, h5file)
        loaded_model = saving_utils.load_model(h5file, optimizer_name='new2')
        out2 = loaded_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # Cleanup
      os.close(fd)
      os.remove(fname)

  def test_saving_constant_initializer_with_numpy(self):
    if h5py is None:
      self.skipTest('h5py required to run this test')

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              2,
              input_shape=(3,),
              kernel_initializer=keras.initializers.Constant(np.ones((3, 2)))))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse',
                    optimizer=optimizers.Kfac(model=model, **_KFAC_KWARGS),
                    metrics=['acc'])
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = saving_utils.load_model(fname, optimizer_name='new')
      os.close(fd)
      os.remove(fname)

if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
