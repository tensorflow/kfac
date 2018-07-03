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
"""Tests for tf.contrib.kfac.tensormatch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from kfac.python.ops import fisher_blocks as fb
from kfac.python.ops import layer_collection as lc
from kfac.python.ops import optimizer

from kfac.python.ops.tensormatch import graph_search as gs

# TODO(b/69055612): Remove once tests are enabled.
# pylint: disable=invalid-name


def _build_model():
  w = tf.get_variable('W', [10, 10])
  b_1 = tf.get_variable('b_1', [
      10,
  ])
  b_0 = tf.get_variable('b_0', [
      10,
  ])
  x = tf.placeholder(tf.float32, shape=(32, 10))
  y = tf.placeholder(tf.float32, shape=(32, 10))

  pre_bias_0 = tf.matmul(x, w)
  pre_bias_1 = tf.matmul(y, w)

  out_0 = pre_bias_0 + b_0  # pylint: disable=unused-variable
  out_1 = pre_bias_1 + b_1  # pylint: disable=unused-variable

  tensor_dict = {}

  tensor_dict['w'] = w
  tensor_dict['b_0'] = b_0
  tensor_dict['b_1'] = b_1
  tensor_dict['x'] = x
  tensor_dict['y'] = y
  tensor_dict['pre_bias_0'] = pre_bias_0
  tensor_dict['pre_bias_1'] = pre_bias_1
  tensor_dict['out_0'] = out_0
  tensor_dict['out_1'] = out_1

  return tensor_dict


def _build_mock_records():
  tensor_dict = _build_model()
  weight_record = gs.MatchRecord(
      record_type=gs.RecordType.fully_connected,
      params=tensor_dict['w'],
      tensor_set={
          tensor_dict['x'], tensor_dict['w'], tensor_dict['pre_bias_0']
      })
  weight_and_bias_0_record = gs.MatchRecord(
      record_type=gs.RecordType.fully_connected,
      params=(tensor_dict['w'], tensor_dict['b_0']),
      tensor_set={
          tensor_dict['x'], tensor_dict['w'], tensor_dict['pre_bias_0'],
          tensor_dict['b_0'], tensor_dict['out_0']
      })
  bias_0_record = gs.MatchRecord(
      record_type=gs.RecordType.fully_connected,
      params=tensor_dict['b_0'],
      tensor_set={
          tensor_dict['pre_bias_0'], tensor_dict['b_0'], tensor_dict['out_0']
      })
  weight_and_bias_1_record = gs.MatchRecord(
      record_type=gs.RecordType.fully_connected,
      params=(tensor_dict['w'], tensor_dict['b_1']),
      tensor_set={
          tensor_dict['y'], tensor_dict['w'], tensor_dict['pre_bias_1'],
          tensor_dict['b_1'], tensor_dict['out_1']
      })
  record_list_dict = collections.defaultdict(list)
  for record in [
      weight_record, weight_and_bias_0_record, bias_0_record,
      weight_and_bias_1_record
  ]:
    record_list_dict[record.params].append(record)
  return tensor_dict, dict(record_list_dict)


def assert_fisher_blocks_match(test_case, layer_collection_a,
                               layer_collection_b):
  """Check that two `LayerCollection`s have matching fisher_blocks."""

  fisher_blocks_a = layer_collection_a.fisher_blocks
  fisher_blocks_b = layer_collection_b.fisher_blocks

  test_case.assertSetEqual(
      set(fisher_blocks_a.keys()), set(fisher_blocks_b.keys()))

  for parameters, block_a in fisher_blocks_a.items():
    block_b = fisher_blocks_b[parameters]
    test_case.assertEqual(type(block_a), type(block_b))
    if hasattr(block_a, '_inputs'):
      test_case.assertEqual(block_a._inputs, block_b._inputs)  # pylint: disable=protected-access
      test_case.assertEqual(block_a._outputs, block_b._outputs)  # pylint: disable=protected-access
    else:
      test_case.assertEqual(block_a._params, block_b._params)  # pylint: disable=protected-access


def sparse_softmax_cross_entropy(labels,
                                 logits,
                                 num_classes,
                                 weights=1.0,
                                 label_smoothing=0.1):
  """Softmax cross entropy with example weights, label smoothing."""
  assert_valid_label = [
      tf.assert_greater_equal(labels, tf.cast(0, dtype=tf.int64)),
      tf.assert_less(labels, tf.cast(num_classes, dtype=tf.int64))
  ]
  with tf.control_dependencies(assert_valid_label):
    labels = tf.reshape(labels, [-1])
    dense_labels = tf.one_hot(labels, num_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=dense_labels,
        logits=logits,
        weights=weights,
        label_smoothing=label_smoothing)
  return loss


class GraphSearchTestCase(tf.test.TestCase):

  def testEmptyGraph(self):
    """Ensure nothing is registered if there are no variables/losses."""
    with tf.Graph().as_default():
      layer_collection = lc.LayerCollection()
      gs.register_layers(layer_collection, tf.trainable_variables())
      self.assertEqual(0, len(layer_collection.fisher_blocks))
      self.assertEqual(0, len(layer_collection.losses))

  def testRegisterLayers(self):
    """Ensure graph search can find a single layer network."""
    with tf.Graph().as_default():
      layer_collection = lc.LayerCollection()

      # Construct a 1-layer model.
      inputs = tf.ones((2, 1)) * 2
      weights = tf.get_variable(
          'w',
          shape=(1, 1),
          dtype=tf.float32,
          initializer=tf.random_normal_initializer)
      bias = tf.get_variable(
          'b', initializer=tf.zeros_initializer(), shape=(1, 1))
      non_variable_bias = tf.ones((1, 1))
      output = tf.matmul(inputs, weights) + bias + non_variable_bias
      logits = tf.tanh(output)

      # Register posterior distribution. Graph search will infer variables
      # needed to construct this.
      layer_collection.register_categorical_predictive_distribution(logits)

      # Register variables.
      gs.register_layers(layer_collection, tf.trainable_variables())

      # Ensure 1-layer got registered.
      self.assertEqual(
          [(weights, bias)],
          list(layer_collection.fisher_blocks.keys()))
      self.assertEqual(1, len(layer_collection.losses))

  def test_register_records_order(self):
    """Ensure records are always registered in the same order."""
    with tf.Graph().as_default():
      data = {'inputs': tf.zeros([10, 4]), 'outputs': tf.zeros([10, 3])}
      params1 = tf.get_variable('w1', [4, 3])
      record1 = gs.MatchRecord(
          gs.RecordType.fully_connected, params1, set(), data=data)

      params2 = (tf.get_variable('w2', [4, 3]),
                 tf.get_variable('b2', [3]))
      record2 = gs.MatchRecord(
          gs.RecordType.fully_connected, params2, set(), data=data)

      # Create a dict of records.
      records = collections.OrderedDict()
      records[params1] = [record1]
      records[params2] = [record2]

      # Register variables.
      layer_collection = lc.LayerCollection(name='lc1')
      gs.register_records(layer_collection, records)

      # Ensure order matches lexicographic order.
      self.assertEqual([params2, params1],
                       list(layer_collection.fisher_blocks.keys()))

      # Create a dict of records in a different order.
      records = collections.OrderedDict()
      records[params2] = [record2]
      records[params1] = [record1]

      # Register variables.
      layer_collection = lc.LayerCollection(name='lc2')
      gs.register_records(layer_collection, records)

      # Ensure order matches lexicographic order.
      self.assertEqual([params2, params1],
                       list(layer_collection.fisher_blocks.keys()))

  def test_multitower_examples_model(self):
    """Ensure graph search runs properly on a multitower setup.

    This test uses linear_model from examples/convnets.
    """
    with tf.Graph().as_default():
      def linear_model(images, labels, num_classes):
        """Creates a linear model.

        Args:
          images: The input image tensors, a tensor of size
              (batch_size x height_in x width_in x channels).
          labels: The sparse target labels, a tensor of size (batch_size x 1).
          num_classes: The number of classes, needed for one-hot encoding (int).

        Returns:
          loss: The total loss for this model (0-D tensor).
          logits: Predictions for this model (batch_size x num_classes).
        """
        images = tf.reshape(images, [images.shape[0], -1])
        logits = tf.layers.dense(images, num_classes, name='logits')
        loss = sparse_softmax_cross_entropy(labels, logits, num_classes)
        return loss, logits

      model = linear_model
      layer_collection = lc.LayerCollection()
      num_towers = 2
      batch_size = num_towers
      num_classes = 2

      # Set up data.
      images = tf.random_uniform(shape=[batch_size, 32, 32, 1])
      labels = tf.random_uniform(
          dtype=tf.int64, shape=[batch_size, 1], maxval=num_classes)

      tower_images = tf.split(images, num_towers)
      tower_labels = tf.split(labels, num_towers)

      # Build model.
      losses = []
      logits = []
      for tower_id in range(num_towers):
        tower_name = 'tower%d' % tower_id
        with tf.name_scope(tower_name):
          with tf.variable_scope(tf.get_variable_scope(), reuse=(tower_id > 0)):
            current_loss, current_logits = model(
                tower_images[tower_id], tower_labels[tower_id], num_classes + 1)
            layer_collection.register_categorical_predictive_distribution(
                current_logits, name='logits')
            losses.append(current_loss)
            logits.append(current_logits)

      # Run the graph scanner.
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        gs.register_layers(layer_collection, tf.trainable_variables())
      self.assertEqual(len(layer_collection.fisher_blocks), 1)
      fisher_block = list(layer_collection.fisher_blocks.values())[0]
      self.assertIsInstance(fisher_block, fb.FullyConnectedKFACBasicFB)
      self.assertEqual(fisher_block.num_registered_towers, num_towers)

      global_step = tf.train.get_or_create_global_step()
      opt = optimizer.KfacOptimizer(
          learning_rate=0.1,
          cov_ema_decay=0.1,
          damping=0.1,
          layer_collection=layer_collection,
          momentum=0.1)
      cost = tf.reduce_mean(losses)
      (cov_update_thunks,
       inv_update_thunks) = opt.make_vars_and_create_op_thunks()
      cov_update_op = tf.group(*(thunk() for thunk in cov_update_thunks))
      inv_update_op = tf.group(*(thunk() for thunk in inv_update_thunks))
      train_op = opt.minimize(cost, global_step=global_step)
      init = tf.global_variables_initializer()

      # Run a single training step.
      with self.test_session() as sess:
        sess.run(init)
        sess.run([cov_update_op])
        sess.run([inv_update_op])
        sess.run([train_op])

  def test_multitower_multi_loss_function(self):
    """Test multitower setup with multiple loss functions.

    The automatic graph scanner should handle multiple loss functions per tower,
    as long as they're registered in a consistent order.
    """
    with tf.Graph().as_default():
      w_1 = tf.get_variable('w_1', shape=[10, 10])
      b_1 = tf.get_variable('b_1', shape=[10])
      w_2 = tf.get_variable('w_2', shape=[10, 10])
      b_2 = tf.get_variable('b_2', shape=[10])
      layer_collection = lc.LayerCollection()
      layer_collection_manual = lc.LayerCollection()
      for tower_num in range(5):
        x = tf.placeholder(tf.float32, shape=(32, 10))
        logits_1 = tf.matmul(x, w_1) + b_1
        logits_2 = tf.matmul(x, w_2) + b_2
        if tower_num == 0:
          reuse = False
        else:
          reuse = True
        with tf.variable_scope('tower%d' % tower_num, reuse=reuse):
          for l in [layer_collection, layer_collection_manual]:
            l.register_categorical_predictive_distribution(
                logits_1, name='loss_1')
            l.register_categorical_predictive_distribution(
                logits_2, name='loss_2')
          layer_collection_manual.register_fully_connected((w_1, b_1), x,
                                                           logits_1)
          layer_collection_manual.register_fully_connected((w_2, b_2), x,
                                                           logits_2)

      gs.register_layers(layer_collection,
                         tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  def test_filter_user_registered_records(self):
    """Matches containing already registered variables should be removed."""
    with tf.Graph().as_default():
      tensor_dict, record_list_dict = _build_mock_records()

      layer_collection = lc.LayerCollection()
      layer_collection.register_fully_connected(
          params=(tensor_dict['w'], tensor_dict['b_1']),
          inputs=tensor_dict['x'],
          outputs=tensor_dict['pre_bias_0'])
      user_registered_variables = set()
      for params in layer_collection.fisher_blocks.keys():
        for variable in gs.ensure_sequence(params):
          user_registered_variables.add(variable)
      filtered_record_list_dict = gs.filter_user_registered_records(
          record_list_dict, user_registered_variables)
      expected_keys = [tensor_dict['b_0']]
      self.assertDictEqual(filtered_record_list_dict,
                           {k: record_list_dict[k]
                            for k in expected_keys})

  def test_filter_grouped_variable_records(self):
    """Matches violating specified parameter groupings should be removed."""
    with tf.Graph().as_default():
      tensor_dict, record_list_dict = _build_mock_records()

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(params=tensor_dict['w'])
      filtered_record_list_dict = gs.filter_grouped_variable_records(
          layer_collection, record_list_dict)
      expected_keys = [tensor_dict['w'], tensor_dict['b_0']]
      self.assertDictEqual(filtered_record_list_dict,
                           {k: record_list_dict[k]
                            for k in expected_keys})

    with tf.Graph().as_default():
      tensor_dict, record_list_dict = _build_mock_records()

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(
          params=(tensor_dict['w'], tensor_dict['b_0']))
      filtered_record_list_dict = gs.filter_grouped_variable_records(
          layer_collection, record_list_dict)
      expected_keys = [(tensor_dict['w'], tensor_dict['b_0'])]
      self.assertDictEqual(filtered_record_list_dict,
                           {k: record_list_dict[k]
                            for k in expected_keys})

  def test_filter_subgraph_records(self):
    """Matches that are strict subgraphs of other matches should be removed."""
    with tf.Graph().as_default():
      tensor_dict, record_list_dict = _build_mock_records()
      filtered_record_list_dict = gs.filter_subgraph_records(record_list_dict)
      expected_keys = [(tensor_dict['w'], tensor_dict['b_0']),
                       (tensor_dict['w'], tensor_dict['b_1'])]
      self.assertDictEqual(filtered_record_list_dict,
                           {k: record_list_dict[k]
                            for k in expected_keys})

  # TODO(b/69055612): Enable once layer_collection is updated
  def DISABLED_test_rnn_multi(self):
    """Test automatic registration on a static RNN.

    The model tested here is designed for MNIST classification. To classify
    images using a recurrent neural network, we consider every image row as a
    sequence of pixels. Because MNIST image shape is 28*28px, we will then
    handle 28 sequences of 28 steps for every sample.
    """
    with tf.Graph().as_default():
      dtype = tf.float32
      n_input = 28  # MNIST data input (img shape: 28*28)
      n_timesteps = 28  # timesteps
      n_hidden = 128  # hidden layer num of features
      n_classes = 10  # MNIST total classes (0-9 digits)

      x = tf.placeholder(dtype, [None, n_timesteps, n_input])
      y = tf.placeholder(tf.int32, [None])
      x_unstack = tf.unstack(x, n_timesteps, 1)

      w_input = tf.get_variable(
          'w_input', shape=[n_input, n_hidden], dtype=dtype)
      b_input = tf.get_variable('b_input', shape=[n_hidden], dtype=dtype)

      w_recurrent = tf.get_variable(
          'w_recurrent', shape=[n_hidden, n_hidden], dtype=dtype)
      b_recurrent = tf.get_variable(
          'b_recurrent', shape=[n_hidden], dtype=dtype)

      w_output = tf.get_variable(
          'w_output', shape=[n_hidden, n_classes], dtype=dtype)
      b_output = tf.get_variable('b_output', shape=[n_classes], dtype=dtype)

      layer_collection_manual = lc.LayerCollection()
      layer_collection_auto = lc.LayerCollection()

      a = tf.zeros([tf.shape(x_unstack[0])[0], n_hidden], dtype=dtype)

      # Here 'a' are the activations, 's' the pre-activations.
      a_list = [a]
      s_input_list = []
      s_recurrent_list = []
      s_list = []
      s_out_list = []
      cost = 0.0

      for i in range(len(x_unstack)):
        input_ = x_unstack[i]

        s_in = tf.matmul(input_, w_input) + b_input
        s_rec = tf.matmul(a, w_recurrent) + b_recurrent
        s = s_in + s_rec

        s_input_list.append(s_in)
        s_recurrent_list.append(s_rec)
        s_list.append(s)

        a = tf.tanh(s)
        a_list.append(a)

        s_out = tf.matmul(a, w_output) + b_output
        s_out_list.append(s_out)

        if i == len(x_unstack) - 1:
          labels = y
        else:
          labels = tf.zeros([tf.shape(y)[0]], dtype=tf.int32)

        cost += tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=s_out, labels=labels))

        layer_collection_manual.register_categorical_predictive_distribution(
            s_out)
        layer_collection_auto.register_categorical_predictive_distribution(
            s_out)

      layer_collection_manual.register_fully_connected_multi(
          (w_input, b_input), x_unstack, s_input_list)
      layer_collection_manual.register_fully_connected_multi(
          (w_recurrent, b_recurrent), a_list[:-1], s_recurrent_list)
      layer_collection_manual.register_fully_connected_multi(
          (w_output, b_output), a_list[1:], s_out_list)

      # Constructing the optimizer performs automatic layer registration.
      auto_optimizer = optimizer.KfacOptimizer(  # pylint: disable=unused-variable
          learning_rate=1,
          cov_ema_decay=1,
          damping=1,
          layer_collection=layer_collection_auto,
          momentum=1)

      assert_fisher_blocks_match(self, layer_collection_manual,
                                 layer_collection_auto)

  def test_graph_search_match_fail(self):
    """Tests graph search with linked bias tensors.

    In this code snippet two non adjacent bias tensors are linked together.
    There is no fisher block in kfac that matches this configuration, so the
    biases should not be registered.
    """
    with tf.Graph().as_default():
      tensor_dict = _build_model()

      layer_collection = lc.LayerCollection()
      # TODO(b/69055612): remove this manual registration once layer_collection
      # implements register_fully_connected_multi.
      layer_collection.register_fully_connected(
          tensor_dict['w'], tensor_dict['x'], tensor_dict['pre_bias_0'])
      layer_collection.define_linked_parameters((tensor_dict['b_0'],
                                                 tensor_dict['b_1']))

      with self.assertRaises(ValueError) as cm:
        gs.register_layers(layer_collection,
                           tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

      self.assertIn('in linked group', str(cm.exception))
      self.assertIn('was not matched', str(cm.exception))
      self.assertIn(
          str(frozenset([tensor_dict['b_0'], tensor_dict['b_1']])),
          str(cm.exception))

  def test_specify_approximation(self):
    """Test specifying approximations.

    If linked parameters are identified along with an approximation, then
    that approximation should be used when registering those parameters.
    """
    with tf.Graph().as_default():
      w_0 = tf.get_variable('w_0', [10, 10])
      w_1 = tf.get_variable('w_1', [10, 10])

      b_0 = tf.get_variable('b_0', [10])
      b_1 = tf.get_variable('b_1', [10])

      x_0 = tf.placeholder(tf.float32, shape=(32, 10))
      x_1 = tf.placeholder(tf.float32, shape=(32, 10))

      pre_bias_0 = tf.matmul(x_0, w_0)
      pre_bias_1 = tf.matmul(x_1, w_1)

      out_0 = pre_bias_0 + b_0  # pylint: disable=unused-variable
      out_1 = pre_bias_1 + b_1  # pylint: disable=unused-variable

      # Group variables as affine layers.
      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(
          (w_0, b_0), approximation=lc.APPROX_KRONECKER_NAME)
      layer_collection.define_linked_parameters(
          (w_1, b_1), approximation=lc.APPROX_DIAGONAL_NAME)
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=32)
      self.assertIsInstance(layer_collection.fisher_blocks[(w_0, b_0)],
                            fb.FullyConnectedKFACBasicFB)
      self.assertIsInstance(layer_collection.fisher_blocks[(w_1, b_1)],
                            fb.FullyConnectedDiagonalFB)

      # Group variables as linear layers and generic parameters.
      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(
          w_0, approximation=lc.APPROX_DIAGONAL_NAME)
      layer_collection.define_linked_parameters(
          b_0, approximation=lc.APPROX_DIAGONAL_NAME)
      layer_collection.define_linked_parameters(
          w_1, approximation=lc.APPROX_KRONECKER_NAME)
      layer_collection.define_linked_parameters(
          b_1, approximation=lc.APPROX_FULL_NAME)
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=32)
      self.assertIsInstance(layer_collection.fisher_blocks[w_0],
                            fb.FullyConnectedDiagonalFB)
      self.assertIsInstance(layer_collection.fisher_blocks[b_0],
                            fb.NaiveDiagonalFB)
      self.assertIsInstance(layer_collection.fisher_blocks[w_1],
                            fb.FullyConnectedKFACBasicFB)
      self.assertIsInstance(layer_collection.fisher_blocks[b_1], fb.FullFB)

  # TODO(b/69055612): Enable once layer_collection implements
  # register_fully_connected_multi.
  def DISABLED_test_specify_approximation_shared_parameters(self):
    """Test specifying approximations with layers containing shared parameters.

    If linked parameters are identified along with an approximation, then
    that approximation should be used when registering those parameters.
    """
    with tf.Graph().as_default():
      tensor_dict = _build_model()

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(
          tensor_dict['w'], approximation=lc.APPROX_KRONECKER_INDEP_NAME)
      layer_collection.define_linked_parameters(
          tensor_dict['b_0'], approximation=lc.APPROX_DIAGONAL_NAME)
      layer_collection.define_linked_parameters(
          tensor_dict['b_1'], approximation=lc.APPROX_FULL_NAME)

      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=1)

      self.assertIsInstance(layer_collection.fisher_blocks[tensor_dict['w']],
                            fb.FullyConnectedMultiIndepFB)
      self.assertIsInstance(
          layer_collection.fisher_blocks[(tensor_dict['b_0'],)],
          fb.NaiveDiagonalFB)
      self.assertIsInstance(
          layer_collection.fisher_blocks[(tensor_dict['b_1'],)], fb.FullFB)

  # TODO(b/69055612): Enable once layer_collection implements
  # register_fully_connected_multi.
  def DISABLED_test_tied_weights_untied_bias_registered_weights(self):
    """Tests that graph search produces right solution on toy model."""
    with tf.Graph().as_default():
      tensor_dict = _build_model()

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected_multi(
          tensor_dict['w'], (tensor_dict['x'], tensor_dict['y']),
          (tensor_dict['pre_bias_0'], tensor_dict['pre_bias_1']))
      layer_collection_manual.register_generic(tensor_dict['b_0'], batch_size=1)
      layer_collection_manual.register_generic(tensor_dict['b_1'], batch_size=1)

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters((tensor_dict['w']))
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=1)

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  def test_tied_weights_untied_bias_registered_affine(self):
    """Test registering linked variables.

    Registering (w, b_1) as linked variables should not raise an error, since
    the matches with parameters (w) and (w, b_0) will be filtered out.
    """
    with tf.Graph().as_default():
      tensor_dict = _build_model()

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected(
          params=(tensor_dict['w'], tensor_dict['b_1']),
          inputs=tensor_dict['y'],
          outputs=tensor_dict['out_1'])
      layer_collection_manual.register_generic(
          tensor_dict['b_0'], batch_size=32)

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters((tensor_dict['w'],
                                                 tensor_dict['b_1']))
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=32)

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  def test_tied_weights_untied_bias(self):
    """Tests that ambiguity in graph raises an error.

    Graph search will find several possible registrations containing w including
    (w, b_1) & (w, b_2). Without any instructions in form of linked tensors or
    manual registration it defaults to registering an error and suggesting that
    the user register (w) as a linked tensor.
    """
    with tf.Graph().as_default():
      _build_model()

      layer_collection = lc.LayerCollection()

      with self.assertRaises(gs.AmbiguousRegistrationError):
        gs.register_layers(layer_collection,
                           tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

  def test_tied_weights_untied_bias_registered_bias(self):
    """Tests that ambiguity in graph raises value error.

    Graph search will find several possible registrations for tensors.
    In this registering b_1 as a linked variable will result in an error
    because there will remain an ambiguity on the other branch of the graph.
    """
    with tf.Graph().as_default():
      tensor_dict = _build_model()

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters((tensor_dict['b_1']))

      with self.assertRaises(gs.AmbiguousRegistrationError):
        gs.register_layers(layer_collection,
                           tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

  def test_multi_time_batch_fold(self):
    """Test that graph search provides desired registration on toy model.

      In this toy example we apply the same linear layer to two different
      inputs. This tests whether graph search can correctly group them. Also
      tests whether batch/time folded is correctly registered as fully
      connected multi fisher blocks.
    """
    with tf.Graph().as_default():
      w = tf.get_variable('W', [10, 10])
      b_0 = tf.get_variable('b_0', [
          10,
      ])
      x = tf.placeholder(tf.float32, shape=(32, 10))
      y = tf.placeholder(tf.float32, shape=(32, 10))

      out_0 = tf.matmul(x, w) + b_0
      out_1 = tf.matmul(y, w) + b_0

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected_multi(
          (w, b_0), (x, y), (out_0, out_1), num_uses=2)

      layer_collection = lc.LayerCollection()
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=16)

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  # TODO(b/69055612): Enable once layer_collection implements
  # register_fully_connected_multi.
  def DISABLED_test_multiple_weights(self):
    """Test that graph search provides desired registration on toy model.

    In this toy example we apply the same linear layer to two different inputs.
    This tests whether graph search can correctly group them.
    """
    with tf.Graph().as_default():
      w = tf.get_variable('W', [10, 10])
      b_0 = tf.get_variable('b_0', [
          10,
      ])
      x = tf.placeholder(tf.float32, shape=(32, 10))
      y = tf.placeholder(tf.float32, shape=(32, 10))

      out_0 = tf.matmul(x, w) + b_0
      out_1 = tf.matmul(y, w) + b_0

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected_multi((w, b_0), (x, y),
                                                             (out_0, out_1))

      layer_collection = lc.LayerCollection()
      gs.register_layers(layer_collection,
                         tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  # TODO(b/69055612): Enable once layer_collection implements
  # register_fully_connected_multi.
  def DISABLED_test_subset_weights_manual_registration(self):
    """Test that graph search provides desired registration on toy model.

    In this toy example we apply the same matmul op to two different inputs
    followed by adding a bias to one of the inputs. This tests whether graph
    search can correctly group them.
    """
    with tf.Graph().as_default():
      w = tf.get_variable('W', [10, 10])
      b_0 = tf.get_variable('b_0', [
          10,
      ])
      x = tf.placeholder(tf.float32, shape=(32, 10))
      y = tf.placeholder(tf.float32, shape=(32, 10))

      out_0 = tf.matmul(x, w) + b_0
      out_1 = tf.matmul(y, w)

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected_multi(
          w, (x, y), (out_0, out_1))
      layer_collection_manual.register_generic(b_0, batch_size=1)

      layer_collection = lc.LayerCollection()
      layer_collection.define_linked_parameters(w)
      gs.register_layers(
          layer_collection,
          tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES),
          batch_size=1)

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)

  def mixed_usage_test(self):
    """Tests that graph search raises error on mixed types usage for tensors.

    Tensors can be reused in various locations in the tensorflow graph. This
    occurs regularly in the case of recurrent models or models with parallel
    graphs. However the tensors must be used for the same operation in each
    location or graph search should raise an error.
    """
    with tf.Graph().as_default():
      w = tf.get_variable('W', [10, 10])
      x = tf.placeholder(tf.float32, shape=(32, 10))
      y = tf.placeholder(tf.float32, shape=(32, 10, 10))

      out_0 = tf.matmul(x, w)  # pylint: disable=unused-variable
      out_1 = y + w  # pylint: disable=unused-variable

      layer_collection = lc.LayerCollection()

      with self.assertRaises(ValueError) as cm:
        gs.register_layers(layer_collection,
                           tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))

      self.assertIn('mixed record types', str(cm.exception))

  def test_resource_variable(self):
    """Ensures that ResourceVariables can be matched."""
    with tf.Graph().as_default():
      w = tf.get_variable('w', [10, 10], use_resource=True)
      b = tf.get_variable('b', [10], use_resource=True)
      x = tf.placeholder(tf.float32, shape=(32, 10))
      out_0 = tf.matmul(x, w) + b

      layer_collection = lc.LayerCollection()
      gs.register_layers(layer_collection, [w, b])

      layer_collection_manual = lc.LayerCollection()
      layer_collection_manual.register_fully_connected((w, b), x, out_0)

      assert_fisher_blocks_match(self, layer_collection,
                                 layer_collection_manual)
      self.assertEqual(1, len(layer_collection.get_blocks()))


if __name__ == '__main__':
  tf.test.main()
