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
"""Saving/loading utilities for models created with the KFAC Optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import warnings
import tensorflow as tf

from tensorflow.python.keras.saving import hdf5_format
from kfac.python.keras import optimizers

# This optional h5py import allows users to import all of tensorflow_kfac
# without h5py. The ImportError is raised manually if they try to use load_model
# without h5py. This follows the Keras save.py style:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/save.py
try:
  import h5py  # pylint: disable=g-import-not-at-top
except ImportError:
  h5py = None


def _compile_args_from_training_config(training_config, custom_objects=None):
  """Return model.compile arguments from training config."""
  if custom_objects is None:
    custom_objects = {}

  optimizer_config = training_config['optimizer_config']
  optimizer = tf.keras.optimizers.deserialize(
      optimizer_config, custom_objects=custom_objects)

  # Recover loss functions and metrics.
  loss_config = training_config['loss']  # Deserialize loss class.
  if isinstance(loss_config, dict) and 'class_name' in loss_config:
    loss_config = tf.keras.losses.get(loss_config)
  loss = tf.nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), loss_config)
  metrics = tf.nest.map_structure(
      lambda obj: custom_objects.get(obj, obj), training_config['metrics'])
  weighted_metrics = tf.nest.map_structure(
      lambda obj: custom_objects.get(obj, obj),
      training_config.get('weighted_metrics', None))
  sample_weight_mode = training_config['sample_weight_mode']
  loss_weights = training_config['loss_weights']

  return dict(optimizer=optimizer,
              loss=loss,
              metrics=metrics,
              weighted_metrics=weighted_metrics,
              loss_weights=loss_weights,
              sample_weight_mode=sample_weight_mode)


def load_model(filepath, custom_objects=None, optimizer_name=None):
  """Loads and compiles a Keras model saved as an HDF5 file.

  Same as tf.keras.model.load_model, except it will always compile the model
  and instantiate the Kfac optimizer correctly. If you do not want the model to
  be compiled, or saved without the optimizer, use tf.keras.models.load_model
  instead.

  Example:
  ```python:
  import tensorflow as tf
  import kfac

  model = tf.keras.Model(...)
  loss = tf.keras.losses.MSE()  # could be a serialized loss function
  optimizer = kfac.keras.optimizers.Kfac(0.001, 0.01, model=model, loss=loss)
  model.compile(optimizer, loss)
  model.fit(...)
  model.save('saved_model.hdf5')  # or use tf.keras.models.save_model
  ...
  loaded_model = kfac.keras.saving_utils.load_model('saved_model.hdf5')
  loaded_model.fit(...)
  ```

  Args:
    filepath: One of the following:
        - String, path to the saved model
        - `h5py.File` object from which to load the model
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization. Kfac will
      be added to this dictionary automatically.
    optimizer_name: Optional string that specifies what variable scope you want
      the KFAC variables to be created in. Useful if you have multiple KFAC
      optimizers on one graph.

  Raises:
    ImportError: If h5py was not imported.

  Returns:
    A compiled Keras model with the Kfac optimizer correctly initialized.
  """
  if h5py is None:
    raise ImportError('`load_model` requires h5py.')
  if not custom_objects:
    custom_objects = {}
  custom_objects['Kfac'] = optimizers.Kfac

  should_open_file = not isinstance(filepath, h5py.File)
  model_file = h5py.File(filepath, mode='r') if should_open_file else filepath

  model = tf.keras.models.load_model(
      model_file, custom_objects=custom_objects, compile=False)

  # Code below is current as of 2019-06-20 and may break due to future changes.
  # github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/hdf5_format.py
  try:
    training_config = model_file.attrs.get('training_config')
    if training_config is None:
      raise ValueError('No training configuration found in save file, meaning '
                       'the model was not compiled. Please use '
                       'tf.keras.models.load_model instead.')
    training_config = json.loads(training_config.decode('utf-8'))

    model.compile(**_compile_args_from_training_config(training_config,
                                                       custom_objects))
    model.optimizer.register_layers(model)
    if optimizer_name:
      model.optimizer.name = optimizer_name

    if 'optimizer_weights' in model_file:
      # Build train function (to get weight updates).
      # Models that aren't graph networks must wait until they are called
      # with data to _make_train_function() and so can't load optimizer
      # weights.
      model._make_train_function()  # pylint: disable=protected-access
      opt_weight_vals = hdf5_format.load_optimizer_weights_from_hdf5_group(
          model_file)
      try:
        model.optimizer.set_weights(opt_weight_vals)
      except ValueError:
        warnings.warn('Error in loading the saved optimizer state. As a '
                      'result, your model is starting with a freshly '
                      'initialized optimizer.')
  finally:
    if should_open_file:
      model_file.close()

  return model
