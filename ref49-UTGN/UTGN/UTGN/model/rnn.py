"""Module for constructing a variant of RNN.
"""

from copy import deepcopy
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from geom_ops import *
from net_ops import *
from utils import *


def _higher_recurrence(mode, config, inputs, num_stepss, alphabet=None):
    """Higher-order recurrence

    Creates multiple layers, possibly with interleaving dihedrals.
    Transforms inputs (like primary sequences) into an internal representation.

    Args:
        mode: train or predict
        config: dict of params
        inputs: input to feed the layers [num_steps, batch_size, FEATURES]
        num_stepss: max length of batch
        alphabet: trainable alphabet variable

    Returns:
        outputs: [BATCH_SIZE, RECURRENT_LAYER_SIZE]
        states: [NUM_STEPS, BATCH_SIZE, RECURRENT_LAYER_SIZE]
    """

    is_training = (mode == 'training')
    initial_inputs = inputs

    if config['higher_order_layers']:
        # higher-order recurrence that concatenates both directions
        # and possibly additional outputs before sending to the next layer.
        # allow additional information to be incorporated in the passed activations
        # like dihedrals

        layer_inputs = initial_inputs
        layers_recurrent_outputs = []
        layers_recurrent_states = []
        num_layers = len(config['recurrent_layer_size'])
        residual_n = config['residual_connections_every_n_layers']
        residual_shift = config['first_residual_connection_from_nth_layer'] - 1

        # iteratively construct each layer
        for layer_idx in range(num_layers):
            with tf.variable_scope('layer' + str(layer_idx)):
                # prepare layer-specific config
                layer_config = deepcopy(config)
                layer_config.update({
                    k: [config[k][layer_idx]]
                    for k in [
                        'recurrent_layer_size',
                        'recurrent_input_keep_probability',
                        'recurrent_output_keep_probability',
                        'recurrent_keep_probability',
                        'recurrent_state_zonein_probability',
                        'recurrent_memory_zonein_probability'
                    ]
                })
                layer_config.update({
                    k: config[k][layer_idx]
                    for k in [
                        'alphabet_keep_probability', 'alphabet_normalization',
                        'recurrent_init'
                    ]
                })
                layer_config.update(
                    {k: (config[k][layer_idx]
                    if not config['single_or_no_alphabet']
                    else config[k]) for k in ['alphabet_size']})

                # core lower-level recurrence
                layer_recurrent_outputs, layer_recurrent_states = _recurrence(
                    mode, layer_config, layer_inputs, num_stepss)

                # residual connections (only for recurrent outputs;
                # other outputs are maintained but not wired in a residual manner)
                # all recurrent layer sizes must be the same
                if (residual_n >= 1) \
                and ((layer_idx - residual_shift) % residual_n == 0) \
                and (layer_idx >= residual_n + residual_shift):
                    layer_recurrent_outputs = layer_recurrent_outputs \
                    + layers_recurrent_outputs[-residual_n]
                    print(('residually wired layer '
                          + str(layer_idx - residual_n + 1)
                          + ' to layer ' + str(layer_idx + 1)))

                # add to list of recurrent layers' outputs
                # (needed for residual connection and some skip connections)
                layers_recurrent_outputs.append(layer_recurrent_outputs)
                layers_recurrent_states.append(layer_recurrent_states)

                # intermediate recurrences,
                # only created if there's at least one layer on top of the current one
                if layer_idx != num_layers - 1:  # not last layer
                    layer_outputs = []

                    # skip connections from all previous layers
                    # (these will not be connected to the final linear output layer)
                    if config['all_to_recurrent_skip_connections']:
                        layer_outputs.append(layer_inputs)

                    # skip connections from initial inputs only
                    # (these will not be connected to the final linear output layer)
                    if config['input_to_recurrent_skip_connections'] \
                    and not config['all_to_recurrent_skip_connections']:
                        layer_outputs.append(initial_inputs)

                    # recurrent state
                    if config['include_recurrent_outputs_between_layers']:
                        layer_outputs.append(layer_recurrent_outputs)

                    # feed outputs as inputs to the next layer up
                    layer_inputs = tf.concat(layer_outputs, 2)

        # if recurrent to output skip connections are enabled,
        # return all recurrent layer outputs, otherwise return only last one.
        # always return all states.
        if config['recurrent_to_output_skip_connections']:
            return tf.concat(layers_recurrent_outputs, 2), \
                   tf.concat(layers_recurrent_states, 1)
        else:
            return layer_recurrent_outputs, \
                   tf.concat(layers_recurrent_states, 1)
    else:
        # simple recurrence, including multiple layers
        # that use TF's builtin functionality,
        # call lower-level recurrence function
        return _recurrence(mode, config, initial_inputs, num_stepss)


def _recurrence(mode, config, inputs, num_stepss):
    """Recurrent layer

    Transforms inputs (like primary sequences) into an internal representation.

    Args:
        mode: train or predict
        config: dict of params
        inputs: input to feed the layers [num_steps, batch_size, FEATURES]
        num_stepss: max length in batch

    Returns:
        outputs: [BATCH_SIZE, RECURRENT_LAYER_SIZE]
        states: [NUM_STEPS, BATCH_SIZE, RECURRENT_LAYER_SIZE]
     """

    is_training = (mode == 'training')
    # reverse seq
    reverse = lambda seqs: tf.reverse_sequence(
        seqs, num_stepss, seq_axis=0, batch_axis=1)

    # create recurrent initialization dict
    if config['recurrent_init'] != None:
        recurrent_init = dict_to_inits(config['recurrent_init'],
                                       config['recurrent_seed'])
    else:
        for case in switch(config['recurrent_unit']):
            if case('LNLSTM'):
                recurrent_init = {'base': None, 'bias': None}
            elif case('CudnnLSTM') or case('CudnnGRU'):
                recurrent_init = {'base': dict_to_init({}), 'bias': None}
            else:
                recurrent_init = {'base': None, 'bias': tf.zeros_initializer()}

    # fused mode vs. explicit dynamic rollout mode
    if 'Cudnn' in config['recurrent_unit']:
        # cuDNN-based fusion
        # assumes all (lower-order) layers are of the same size
        # (first layer size) and all input dropouts are the same
        # (first layer one). Does not support peephole connections,
        # and only supports input dropout as a form of regularization.
        layer_size = config['recurrent_layer_size'][0]
        num_layers = len(config['recurrent_layer_size'])
        input_keep_prob = config['recurrent_input_keep_probability'][0]

        for case in switch(config['recurrent_unit']):
            if case('CudnnLSTM'):
                cell = cudnn_rnn.CudnnLSTM
            elif case('CudnnGRU'):
                cell = cudnn_rnn.CudnnGRU

        # need this layer because cuDNN dropout only applies to
        # inputs between layers, not the first inputs
        if is_training and input_keep_prob < 1:
            inputs = tf.nn.dropout(inputs,
                                   input_keep_prob,
                                   seed=config['dropout_seed'])

        # this isn't needed, but it allows multiple cuDNN-based models
        # to run on the same GPU when num_layers = 1
        if num_layers > 1:
            dropout_kwargs = {
                'dropout': 1 - input_keep_prob,
                'seed': config['dropout_seed']
            }
        else:
            dropout_kwargs = {}

        outputs = []
        states = []
        scopes = ['fw', 'bw'] if config['bidirectional'] else ['fw']
        for scope in scopes:
            with tf.variable_scope(scope):
                rnn = cell(num_layers=num_layers,
                           num_units=layer_size,
                           direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
                           kernel_initializer=recurrent_init['base'],
                           bias_initializer=recurrent_init['bias'],
                           **dropout_kwargs)
                inputs_directed = inputs if scope == 'fw' else reverse(inputs)
                outputs_directed, (_,
                                   states_directed) = rnn(inputs_directed,
                                                          training=is_training)
                outputs_directed = outputs_directed if scope == 'fw' \
                else reverse(outputs_directed)
                outputs.append(outputs_directed)
                states.append(states_directed)
        outputs = tf.concat(outputs, 2)
        states = tf.concat(states, 2)[0]

    else:
        # TF-based dynamic rollout
        if config['bidirectional']:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=_recurrent_cell(mode, config, recurrent_init, 'fw'),
                cell_bw=_recurrent_cell(mode, config, recurrent_init, 'bw'),
                inputs=inputs,
                time_major=True,
                sequence_length=tf.to_int64(num_stepss),
                dtype=tf.float32,
                swap_memory=True,
                parallel_iterations=config['num_recurrent_parallel_iters'])
            outputs = tf.concat(outputs, 2)
            states = tf.concat(states, 2)
            # [NUM_STEPS, BATCH_SIZE, 2 x RECURRENT_LAYER_SIZE]
            # outputs of recurrent layer over all time steps.
        else:
            outputs, states = tf.nn.dynamic_rnn(
                cell=_recurrent_cell(mode, config, recurrent_init),
                inputs=inputs,
                time_major=True,
                sequence_length=num_stepss,
                dtype=tf.float32,
                swap_memory=True,
                parallel_iterations=config['num_recurrent_parallel_iters'])
            # [NUM_STEPS, BATCH_SIZE, RECURRENT_LAYER_SIZE]
            # outputs of recurrent layer over all time steps.

        # add newly created variables to respective collections
        if is_training:
            for v in tf.trainable_variables():
                if 'rnn' in v.name and ('cell/kernel' in v.name):
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, v)
                if 'rnn' in v.name and ('cell/bias' in v.name):
                    tf.add_to_collection(tf.GraphKeys.BIASES, v)

    return outputs, states


def _recurrent_cell(mode, config, recurrent_init, name=''):
    """Create recurrent cell(s) used in RNN

    Args:
        mode: train or predict
        config: dict of params
        recurrent_init: dictionary of tf initializations
        name: name of cell

    Returns:
        RNN cell
    """

    is_training = (mode == 'training')

    # lower-order multilayer
    cells = []
    for layer_idx, (layer_size, input_keep_prob, output_keep_prob, \
        keep_prob, hidden_state_keep_prob, memory_cell_keep_prob) \
        in enumerate(zip(
            config['recurrent_layer_size'],
            config['recurrent_input_keep_probability'],
            config['recurrent_output_keep_probability'],
            config['recurrent_keep_probability'],
            config['recurrent_state_zonein_probability'],
            config['recurrent_memory_zonein_probability'])):

        # set context
        with tf.variable_scope('sublayer' + str(layer_idx) +
                               (name if name is '' else '_' + name),
                               initializer=recurrent_init['base']):

            # create core cell
            for case in switch(config['recurrent_unit']):
                if case('Basic'):
                    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=layer_size,
                                                       reuse=(not is_training))
                elif case('GRU'):
                    cell = tf.nn.rnn_cell.GRUCell(num_units=layer_size,
                                                  reuse=(not is_training))
                elif case('LSTM'):
                    cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=layer_size,
                        use_peepholes=config['recurrent_peepholes'],
                        forget_bias=config['recurrent_forget_bias'],
                        cell_clip=config['recurrent_threshold'],
                        initializer=recurrent_init['base'],
                        reuse=(not is_training))
                elif case('LNLSTM'):
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                        num_units=layer_size,
                        forget_bias=config['recurrent_forget_bias'],
                        layer_norm=config['recurrent_layer_normalization'],
                        dropout_keep_prob=keep_prob,
                        reuse=(not is_training))
                elif case('LSTMBlock'):
                    cell = tf.contrib.rnn.LSTMBlockCell(
                        num_units=layer_size,
                        forget_bias=config['recurrent_forget_bias'],
                        use_peephole=config['recurrent_peepholes'])

            # wrap cell with zoneout
            if hidden_state_keep_prob < 1 or memory_cell_keep_prob < 1:
                cell = ZoneoutWrapper(
                    cell=cell,
                    is_training=is_training,
                    seed=config['zoneout_seed'],
                    hidden_state_keep_prob=hidden_state_keep_prob,
                    memory_cell_keep_prob=memory_cell_keep_prob)

            # if not just evaluation, then wrap cell in dropout
            if is_training and (input_keep_prob < 1
            or output_keep_prob < 1 or keep_prob < 1):
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=cell,
                    input_keep_prob=input_keep_prob,
                    output_keep_prob=output_keep_prob,
                    state_keep_prob=keep_prob,
                    variational_recurrent=config[
                        'recurrent_variational_dropout'],
                    seed=config['dropout_seed'])

            # add to collection
            cells.append(cell)

    # stack multiple cells if needed
    if len(cells) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = cells[0]

    return cell


class ZoneoutWrapper(RNNCell):
    """Operator adding zoneout to hidden state and memory of the given cell."""

    def __init__(self,
                 cell,
                 memory_cell_keep_prob=1.0,
                 hidden_state_keep_prob=1.0,
                 seed=None,
                 is_training=True):
        """Create a cell with hidden state and memory zoneout.

        If this class is used to wrap a Dropout cell, then it will override the output 
        Dropout but maintain input Dropout. If a Dropout cell wraps a Zoneout cell,
        then both Dropout and Zoneout will be applied to the outputs.

        This function assumes that LSTM Cells are using the new tuple-based state.

        Args:
            cell: an BasicLSTMCell or LSTMCell
            memory_cell_keep_prob: unit Tensor or float between 0 and 1, memory cell
                keep probability; if it is float and 1, no zoneout will be added.
            hidden_state_keep_prob: unit Tensor or float between 0 and 1, hidden state
                keep probability; if it is float and 1, no zoneout will be added.
            seed: (optional) integer, the randomness seed.
            is_training: boolean, determines which mode of the zoneout is used.

        Raises:
          TypeError: if cell is not a BasicLSTMCell or LSTMCell.
          ValueError: if memory_cell_keep_prob or hidden_state_keep_prob is not between 0 and 1.
        """
        # if not (isinstance(cell, BasicLSTMCell) or isinstance(cell, LSTMCell)):
        #   raise TypeError("The parameter cell is not a BasicLSTMCell or LSTMCell.")
        if (isinstance(memory_cell_keep_prob, float)
                and not (memory_cell_keep_prob >= 0.0
                         and memory_cell_keep_prob <= 1.0)):
            raise ValueError("Parameter memory_cell_keep_prob must be \
                              between 0 and 1: %d" % memory_cell_keep_prob)
        if (isinstance(hidden_state_keep_prob, float)
                and not (hidden_state_keep_prob >= 0.0
                         and hidden_state_keep_prob <= 1.0)):
            raise ValueError("Parameter hidden_state_keep_prob must be \
                              between 0 and 1: %d" % hidden_state_keep_prob)
        self._cell = cell
        self._memory_cell_keep_prob = memory_cell_keep_prob
        self._hidden_state_keep_prob = hidden_state_keep_prob
        self._seed = seed
        self._is_training = is_training

        self._has_memory_cell_zoneout = (
            not isinstance(self._memory_cell_keep_prob, float)
            or self._memory_cell_keep_prob < 1)
        self._has_hidden_state_zoneout = (
            not isinstance(self._hidden_state_keep_prob, float)
            or self._hidden_state_keep_prob < 1)

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared zoneouts."""

        # compute output and new state as before
        output, new_state = self._cell(inputs, state, scope)

        # if either hidden state or memory cell zoneout is applied,
        # then split state and process
        if self._has_hidden_state_zoneout or self._has_memory_cell_zoneout:
            # split state
            c_old, m_old = state
            c_new, m_new = new_state

            # apply zoneout to memory cell and hidden state
            c_and_m = []
            for s_old, s_new, p, has_zoneout in [
                (c_old, c_new, self._memory_cell_keep_prob,
                 self._has_memory_cell_zoneout),
                (m_old, m_new, self._hidden_state_keep_prob,
                 self._has_hidden_state_zoneout)
            ]:
                if has_zoneout:
                    if self._is_training:
                        mask = nn_ops.dropout(
                            array_ops.ones_like(s_new), p, seed=self._seed
                        ) * p  # this should just random ops instead. See dropout code for how.
                        s = ((1. - mask) * s_old) + (mask * s_new)
                    else:
                        s = ((1. - p) * s_old) + (p * s_new)
                else:
                    s = s_new

                c_and_m.append(s)

            # package final results
            new_state = LSTMStateTuple(*c_and_m)
            output = new_state.h

        return output, new_state
