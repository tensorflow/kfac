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
"""Tests for keras/callbacks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from kfac.python.keras import callbacks
from kfac.python.keras import optimizers

layers = tf.keras.layers
_SEED = 1234


class HyperParamTracker(tf.keras.callbacks.Callback):
  EPOCH, BATCH = range(2)

  def __init__(self, hyper, record_list, frequency):
    self.hyper = hyper
    self.record_list = record_list
    self.frequency = frequency

  def on_batch_end(self, batch, logs=None):
    if self.frequency == HyperParamTracker.BATCH:
      val = tf.keras.backend.get_value(getattr(self.model.optimizer,
                                               self.hyper))
      self.record_list.append(val)

  def on_epoch_end(self, epoch, logs=None):
    if self.frequency == HyperParamTracker.EPOCH:
      val = tf.keras.backend.get_value(getattr(self.model.optimizer,
                                               self.hyper))
      self.record_list.append(val)


class CallbacksTest(parameterized.TestCase, tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(CallbacksTest, self).__init__(*args, **kwargs)
    self.batch_size = 16
    self.num_steps = 20
    self.data = np.random.random((self.batch_size*self.num_steps))
    self.labels = np.random.random((self.batch_size*self.num_steps))

  def setUp(self):
    super(CallbacksTest, self).setUp()
    self.model = tf.keras.Sequential([layers.Dense(1, input_shape=(1,))])
    tf.random.set_random_seed(_SEED)

  def testPolynomialDecayValues(self):
    init_value = 0.01
    final_value = 0.0002
    power = 0.6
    num_decay_steps = 11
    num_delay_steps = 3
    opt = tf.keras.optimizers.Adam(learning_rate=init_value)
    self.model.compile(opt, 'mse')
    lr_list = []
    cbs = [
        callbacks.PolynomialDecay(hyperparameter='learning_rate',
                                  init_value=init_value,
                                  final_value=final_value,
                                  power=power,
                                  num_decay_steps=num_decay_steps,
                                  num_delay_steps=num_delay_steps,
                                  verbose=1),
        HyperParamTracker('learning_rate', lr_list, HyperParamTracker.BATCH)
    ]
    self.model.fit(
        self.data, self.labels, batch_size=self.batch_size, callbacks=cbs)
    expected_list = [init_value] * num_delay_steps + [
        (init_value - final_value) *
        (1 - min(i, num_decay_steps) / float(num_decay_steps)) ** power +
        final_value for i in range(self.num_steps - num_delay_steps)
    ]
    self.assertAllClose(lr_list, expected_list)

  def testExponentialDampingValuesWithDecayRate(self):
    init_value = 0.01
    decay_rate = 0.3
    num_decay_steps = 4
    num_delay_steps = 3
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=init_value, model=self.model, loss='mse')
    self.model.compile(opt, 'mse')
    damping_list = []
    cbs = [
        callbacks.ExponentialDecay(hyperparameter='damping',
                                   init_value=init_value,
                                   decay_rate=decay_rate,
                                   num_decay_steps=num_decay_steps,
                                   num_delay_steps=num_delay_steps,
                                   verbose=1),
        HyperParamTracker('damping', damping_list, HyperParamTracker.BATCH)
    ]
    self.model.fit(
        self.data, self.labels, batch_size=self.batch_size, callbacks=cbs)

    expected_list = [init_value] * num_delay_steps + [
        init_value * decay_rate ** min(i, num_decay_steps)
        for i in range(self.num_steps - num_delay_steps)
    ]
    self.assertAllClose(damping_list, expected_list)

  def testExponentialDampingValuesWithFinalValue(self):
    init_value = 0.01
    final_value = 0.0001
    num_decay_steps = 4
    num_delay_steps = 3
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=init_value, model=self.model, loss='mse')
    self.model.compile(opt, 'mse')
    damping_list = []
    cbs = [
        callbacks.ExponentialDecay(hyperparameter='damping',
                                   init_value=init_value,
                                   final_value=final_value,
                                   num_decay_steps=num_decay_steps,
                                   num_delay_steps=num_delay_steps,
                                   verbose=1),
        HyperParamTracker('damping', damping_list, HyperParamTracker.BATCH)
    ]
    self.model.fit(
        self.data, self.labels, batch_size=self.batch_size, callbacks=cbs)

    expected_list = [init_value] * num_delay_steps + [
        init_value * (final_value/init_value) **
        (min(i, num_decay_steps)*1./num_decay_steps)
        for i in range(self.num_steps - num_delay_steps)
    ]
    self.assertAllClose(damping_list, expected_list)
    self.assertNear(damping_list[-1], final_value, err=1e-5)

  def testExponentialDampingValuesWithFinalValueAndRate(self):
    init_value = 0.01
    final_value = 0.0001
    decay_rate = 0.6
    num_delay_steps = 3
    opt = optimizers.Kfac(
        learning_rate=0.01, damping=init_value, model=self.model, loss='mse')
    self.model.compile(opt, 'mse')
    damping_list = []
    cbs = [
        callbacks.ExponentialDecay(hyperparameter='damping',
                                   init_value=init_value,
                                   final_value=final_value,
                                   decay_rate=decay_rate,
                                   num_delay_steps=num_delay_steps,
                                   verbose=1),
        HyperParamTracker('damping', damping_list, HyperParamTracker.BATCH)
    ]
    self.model.fit(
        self.data, self.labels, batch_size=self.batch_size, callbacks=cbs)

    expected_list = [init_value] * num_delay_steps + [
        max((init_value * decay_rate ** i), final_value)
        for i in range(self.num_steps - num_delay_steps)
    ]
    self.assertAllClose(damping_list, expected_list)
    self.assertNear(damping_list[-1], final_value, err=1e-5)

  @parameterized.named_parameters(
      ('_Exponential', 'damping',
       callbacks.ExponentialDecay(hyperparameter='damping',
                                  init_value=0.01,
                                  decay_rate=0.3,
                                  num_decay_steps=30)),
      ('_Polynomial', 'learning_rate',
       callbacks.PolynomialDecay(hyperparameter='learning_rate',
                                 init_value=0.001,
                                 final_value=0.002,
                                 power=0.6,
                                 num_decay_steps=30)))
  def testTrainHistory(self, hyper, callback):
    opt = optimizers.Kfac(learning_rate=0.001, damping=0.01,
                          model=self.model, loss='mse', num_burnin_steps=5)
    self.model.compile(opt, 'mse')
    lst = []
    cbs = [callback, HyperParamTracker(hyper, lst, HyperParamTracker.EPOCH)]
    hist = self.model.fit(self.data, self.labels,
                          batch_size=self.batch_size, epochs=3, callbacks=cbs)
    self.assertAllClose(lst, hist.history[hyper])

  def testDampingDecayFailsWithNoDamping(self):
    with self.assertRaisesRegex(ValueError, '.*must have a "damping".*'):
      self.model.compile('adam', 'mse')
      cb = callbacks.ExponentialDecay(hyperparameter='damping',
                                      init_value=0.01,
                                      decay_rate=0.3,
                                      num_decay_steps=4)
      self.model.fit(self.data, self.data, callbacks=[cb])

  def testExponentialDampingFailsNoRateOrFinalValue(self):
    with self.assertRaisesRegex(ValueError, '.*must specify exactly two of.*'):
      callbacks.ExponentialDecay(hyperparameter='damping',
                                 init_value=0.01)

  def testExponentialDampingFailsWithAllOptionals(self):
    with self.assertRaisesRegex(ValueError, '.*must specify exactly two of.*'):
      callbacks.ExponentialDecay(hyperparameter='learning_rate',
                                 init_value=0.01,
                                 final_value=0.001,
                                 decay_rate=0.99,
                                 num_decay_steps=50)


if __name__ == '__main__':
  tf.test.main()
