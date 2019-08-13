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
"""Hyperparameter Scheduling Callbacks for Keras K-FAC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class HyperparameterDecay(tf.keras.callbacks.Callback):
  """Base class for global_step/iterations-based optimizer decay callbacks."""

  def __init__(self, hyperparameter, num_delay_steps=0, verbose=0):
    """Construct a new HyperparameterDecay.

    Args:
      hyperparameter: String specifying the optimizer attribute to decay.
      num_delay_steps: Integer specifying how many steps to wait before decaying
        the attribute.
      verbose: Integer. When > 1, the hyperparameter value is printed every
        epoch.
    """

    self._hyperparameter = hyperparameter
    self._num_delay_steps = num_delay_steps
    self.verbose = verbose

  def on_train_begin(self, logs=None):
    self._optimizer = self.model.optimizer
    if not hasattr(self._optimizer, self._hyperparameter):
      raise ValueError('Optimizer must have a "{}" attribute.'
                       .format(self._hyperparameter))
    if not hasattr(self._optimizer, 'iterations'):
      raise ValueError('Optimizer must have a "iterations" attribute.')

  def on_epoch_begin(self, epoch, logs=None):
    if self.verbose > 0:
      value = float(tf.keras.backend.get_value(getattr(self._optimizer,
                                                       self._hyperparameter)))
      print('\nEpoch {:05}: Current {} is {}.'
            .format(epoch + 1, self._hyperparameter, value))

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None:
      logs[self._hyperparameter] = tf.keras.backend.get_value(
          getattr(self._optimizer, self._hyperparameter))

  def _get_global_step(self):
    return (tf.keras.backend.get_value(self._optimizer.iterations)
            - self._num_delay_steps)


class PolynomialDecay(HyperparameterDecay):
  """Polynomial Optimizer Hyperparameter Schedule.

  Based on https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay

  The decay applies as follows for num_decay_steps steps when the global_step
  (i.e. optimizer.iterations) exceeds the num_delay_steps.

    step = global_step - num_delay_steps
    decayed_value = (init_value - final_value) *
                    (1 - step / num_decay_steps) ^ (power) + final_value
  """

  def __init__(self,
               hyperparameter,
               init_value,
               final_value,
               power,
               num_decay_steps,
               **kwargs):
    """Construct a new PolynomialDecay Callback.

    Args:
      hyperparameter: String specifying the optimizer attribute to decay.
      init_value: Float specifying initial value of the attribute.
      final_value: Float specifying value of attribute at the end of the decay.
      power: Float specifying power (exponent) of the polynomial decay.
      num_decay_steps: Integer, number of steps to decay the attribute.
      **kwargs: Keyword arguments for HyperparameterDecay. This includes
        num_delay_steps and verbose.
    """
    super(PolynomialDecay, self).__init__(hyperparameter, **kwargs)
    self._init_value = init_value
    self._final_value = final_value
    self._power = power
    self._num_decay_steps = num_decay_steps

  def on_batch_begin(self, batch, logs=None):
    step = self._get_global_step()
    if step > 0 and step <= self._num_decay_steps:
      decayed_value = ((self._init_value - self._final_value) *
                       (1 - step / self._num_decay_steps) ** (self._power) +
                       self._final_value)
      setattr(self._optimizer, self._hyperparameter, decayed_value)


class ExponentialDecay(HyperparameterDecay):
  """Exponential Optimizer Hyperparameter Decay Schedule.

  The decay applies as follows for num_decay_steps steps when the global_step
  (i.e. optimizer.iterations) exceeds the num_delay_steps. If num_decay_steps
  is not provided, it will keep decaying for the duration of training.

  When a decay rate and num_decay_steps is provided:
    step = min(global_step - num_delay_steps, num_decay_steps)
    decayed_value = init_value * decay_rate^step

  When a decay_rate and final_value are provided:
    step = global_step - num_delay_steps
    decayed_value = max(init_value * decay_rate^step, final_value)

  When a final value and num_decay_steps is provided:
    step = global_step - num_delay_steps
    decayed_value = init_value *
                   (final_value / init_value) ^ (step / num_decay_steps)
  """

  def __init__(self,
               hyperparameter,
               init_value,
               final_value=None,
               decay_rate=None,
               num_decay_steps=None,
               **kwargs):
    """Construct a new ExponentialDecay Callback.

    You must specify exactly two of final_value, decay_rate, and
    num_decay_steps.


    Args:
      hyperparameter: String specifying the optimizer attribute to decay.
      init_value: Float specifying initial value of the attribute.
      final_value: Float specifying value of attribute at the end of the decay.
      decay_rate: Float specifying the decay rate of the decay.
      num_decay_steps: Integer, number of steps to decay the attribute.
      **kwargs: Keyword arguments for HyperparameterDecay. This includes
        num_delay_steps and verbose.
    """
    super(ExponentialDecay, self).__init__(hyperparameter, **kwargs)
    self._num_decay_steps = num_decay_steps

    # In theory, we could support more different combinations of final_value,
    # num_decay_steps, and decay_rate, but for the sake of clarity we will limit
    # this callback to the below combinations.
    if final_value and decay_rate and num_decay_steps:
      raise ValueError('You must specify exactly two of final_value, decay_rate'
                       ', and num_decay_steps.')
    if final_value and decay_rate:
      self._decay_func = lambda step: max(  # pylint: disable=g-long-lambda
          (init_value * (decay_rate ** step)), final_value)
    elif decay_rate and num_decay_steps:
      self._decay_func = lambda step: (init_value * decay_rate ** step)
    elif final_value and num_decay_steps:
      self._decay_func = lambda step: (  # pylint: disable=g-long-lambda
          init_value * (final_value / init_value) **
          (float(step) / num_decay_steps))
    else:
      raise ValueError('You must specify exactly two of final_value, decay_rate'
                       ', and num_decay_steps.')

  def on_batch_begin(self, batch, logs=None):
    global_step = self._get_global_step()
    if (global_step > 0 and
        (not self._num_decay_steps or global_step <= self._num_decay_steps)):
      decayed_value = self._decay_func(global_step)
      setattr(self._optimizer, self._hyperparameter, decayed_value)
