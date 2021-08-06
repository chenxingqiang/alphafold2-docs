"""Encoder portion of the transformer.

Input tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES]
Output tensor of the same shape.

Code adapted from tensor2tensor.
https://github.com/tensorflow/tensor2tensor/

TODO: include positional encodings at every time step for UT
"""

import numpy as np
import tensorflow as tf
from utils import *

cast32 = lambda x: tf.dtypes.cast(x, tf.float32)
none_to1 = lambda x: -1 if x == None else x


def _get_mean_std(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    squared = tf.square(x - mean)
    variance = tf.reduce_mean(squared, axis=-1, keepdims=True)
    std = tf.sqrt(variance)
    return mean, std


def _layer_norm(layer):
    """Perform layer normalization.

    Not the same as batch normalization.

    Args:
        layer: Tensor

    Returns:
        Tensor
    """

    with tf.variable_scope("norm"):
        scale = tf.get_variable("scale",
                                shape=layer.shape[-1],
                                dtype=tf.float32)
        base = tf.get_variable("base", shape=layer.shape[-1], dtype=tf.float32)
        mean, std = _get_mean_std(layer)
        norm = (layer - mean) / (std + 1e-6)
        return norm * scale + base


def _feed_forward(x, d_model, d_ff, keep_prob, train=True):
    """Feed forward layer along with of relu and dropout.

    FFN(x) = max(0,xW1+b1)W2+b2

    Args:
        x: Input tensor.
        d_model: dimension of W2.
        d_ff: dimension of W1.
        keep_prob: The drop out probability.
        train: train or predict

    Returns:
        Tensor
    """

    with tf.variable_scope("feed_forward"):
        hidden = tf.layers.dense(x, units=d_ff, name="hidden")
        hidden = tf.nn.relu(hidden)
        if train:
            hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.layers.dense(hidden, units=d_model, name="out")


def _generate_positional_encodings(d_model, seq_len=5000, time_step=0):
    """Create positional encoding.

    Args:
        d_model: dimension of input embeddings
        seq_len: maximum sequence length of batch

    Returns:
        Constant tensor of shape [1, seq_len, d_model]
    """

    encodings = np.zeros((seq_len, d_model), dtype=float)
    position = np.arange(0, seq_len).reshape((seq_len, 1))
    two_i = np.arange(0, d_model, 2)
    div_term = np.exp(-np.log(10000.0) * two_i / d_model)
    encodings[:, 0::2] = np.sin(position * div_term) \
                         + np.sin(time_step * div_term)
    encodings[:, 1::2] = np.cos(position * div_term) \
                         + np.cos(time_step * div_term)

    pos_encodings = tf.constant(encodings.reshape((1, seq_len, d_model)),
                                dtype=tf.float32,
                                name="positional_encodings")

    return pos_encodings


def _prepare_embeddings(x,
                        positional_encodings,
                        keep_prob,
                        train=True,
                        include_pos_encodings=False):
    """Add positional encoding and normalize embeddings.

    Args:
        x: input embeddings of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        positional_encodings: encoding tensor of shape [1, SEQ_LEN, FEATURES].
        keep_prob: The drop out probability.
        train: train or predict

    Returns:
        Tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    with tf.variable_scope("prepare_input"):
        if include_pos_encodings:
            seq_len = tf.shape(x)[1]
            x = x + positional_encodings[:, :seq_len, :]

        if train:
            x = tf.nn.dropout(x, keep_prob)
        return _layer_norm(x)


def _attention(query, key, value, mask, keep_prob, train=True):
    """Calculates scaled dot-product attention.

    softmax(Q K^{T} / sqrt(d_{k}))V

    Args:
        query: A query tensor of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES].
        key:  The key tensor.
        value: The value tensor.
        mask: Mask of shape [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
        keep_prob: The drop out probability.
        train: train or predict

    Returns:
        The scaled dot-product attention.
        Shape: [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]
    """

    d_k = query.shape[-1].value
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2]))
    scores = scores / tf.constant(np.sqrt(d_k), dtype=tf.float32)
    mask_add = ((scores * 0) - cast32(1e9)) * (tf.constant(1.) - cast32(mask))
    scores = scores * cast32(mask) + mask_add
    attn = tf.nn.softmax(scores, axis=-1)
    if train:
        attn = tf.nn.dropout(attn, keep_prob)
    return tf.matmul(attn, value)


def _prepare_multi_head_attention(x, heads, name):
    """Prepares for multihead attention.

    Prepares query, key, value that have form [BATCH_SIZE, SEQ_LEN, FEATURES].

    Args:
        x: Tensor input.
        heads: Number of heads.
        name: Either query, key, or value.

    Returns:
        A prepared Q, K, or V of form [BATCH_SIZE, HEADS, SEQ_LEN, FEATURES]

    Raises:
        AssertionError: Dimension of features must be divisible
                        by the number of heads.
    """

    n_batches, seq_len, d_model = x.get_shape().as_list()
    seq_len = none_to1(seq_len)
    assert d_model % heads == 0, "Dimension of features needs to be divisible by the number of heads."
    d_k = d_model // heads
    x = tf.layers.dense(x, units=d_model, name=name)
    x = tf.reshape(x, shape=(n_batches, seq_len, heads, d_k))
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    return x


def _multi_head_attention(query,
                          key,
                          value,
                          mask,
                          heads,
                          keep_prob,
                          train=True):
    """Calculates the multihead attention.

    Args:
        query: query tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        key: key tensor.
        value: value tensor.
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        heads: number of heads.
        keep_prob: The drop out probability.
        train: train or predict

    Returns:
        Tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES]
    """

    with tf.variable_scope("multi_head"):
        n_batches, seq_len, d_model = query.get_shape().as_list()
        query = _prepare_multi_head_attention(query, heads, "query")
        key = _prepare_multi_head_attention(key, heads, "key")
        value = _prepare_multi_head_attention(value, heads, "value")
        mask = tf.expand_dims(mask, axis=1)
        out = _attention(query,
                         key,
                         value,
                         mask=mask,
                         keep_prob=keep_prob,
                         train=train)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        seq_len = none_to1(seq_len)
        out = tf.reshape(out, shape=[n_batches, seq_len, d_model])
        return tf.layers.dense(out, units=d_model, name="attention")


def _encoder_layer(x,
                   mask,
                   layer_num,
                   heads,
                   keep_prob,
                   d_ff,
                   config,
                   train=True,
                   transition_function="ff"):
    """Create a single encoder layer.

    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        layer_num: The number label of an encoder layer.
        heads: Number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        train: train or predict

    Returns:
        Tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    d_model = x.shape[-1]
    with tf.variable_scope("attention_" + str(layer_num)):
        attention_out = _multi_head_attention(x,
                                              x,
                                              x,
                                              mask=mask,
                                              heads=heads,
                                              keep_prob=keep_prob,
                                              train=train)

        if train:
            attention_out = tf.nn.dropout(attention_out, keep_prob)
        added = x + attention_out
        x = _layer_norm(added)

    transition_func = config['transition_function']
    with tf.variable_scope(transition_func + "_" + str(layer_num)):

        for case in switch(transition_func):
            if case('feed_forward'):
                trans_out = _feed_forward(x,
                                          d_model,
                                          d_ff,
                                          keep_prob,
                                          train=train)
            elif case('1d_seperable_conv'):
                trans_out = tf.layers.separable_conv1d(
                    x,
                    d_model,
                    kernel_size=config['seperable_kernel_size'],
                    padding="same",
                    name="seperable_conv1d",
                    activation=tf.nn.relu)
            else:
                raise ValueError("Not an available transition function.")

        if train:
            trans_out = tf.nn.dropout(trans_out, keep_prob)
        added = x + trans_out
        return _layer_norm(added)


def _encoder(x, mask, n_layers, heads, keep_prob, d_ff, config, train=True):
    """Create the encoder architecture

    Args:
        x: input tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
        mask: mask tensor of shape [BATCH_SIZE, SEQ_LEN, FEATURES].
        n_layers: number of layers of the encoder model.
        heads: number of heads.
        keep_prob: The drop out probability.
        d_ff: dimension of W1.
        train: train or predict

    Returns:
        Tensor of shape: [BATCH_SIZE, SEQ_LEN, FEATURES].
    """

    with tf.variable_scope("encoder"):
        for i in range(n_layers):
            x = _encoder_layer(x,
                               mask=mask,
                               layer_num=i,
                               heads=heads,
                               keep_prob=keep_prob,
                               d_ff=d_ff,
                               config=config,
                               train=train)
        return x


def _ut_function(state, step, halting_probability, remainders, n_updates,
                 previous_state, encoder_layer_func, config):
    """Implements ACT (position-wise halting).

    Args:
        state: Tensor of shape [batch_size, length, input_dim]
        step: indicates number of steps taken so far
        halting_probability: halting probability
        remainders: ACT remainders
        n_updates: ACT n_updates
        previous_state: previous state
        encoder_layer_func: encoder layer function
        config: configuration dict

    Returns:
        transformed_state: transformed state
        step: step + 1
        halting_probability: halting probability
        remainders: act remainders
        n_updates: act n_updates
        new_state: new state
    """

    num_layers = config['transformer_layers']
    threshold = config['act_threshold']

    # if config['include_pos_encodings']:
    #     seq_len = tf.shape(state)[1]
    #     d_model = int(state.shape[2])
    #     state = state + _generate_positional_encodings(
    #         d_model,
    #         time_step=step+1
    #     )[:, :seq_len, :]

    with tf.variable_scope("sigmoid_activation_for_pondering"):
        p = tf.layers.dense(state, 1, activation=tf.nn.sigmoid, use_bias=True)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
        tf.greater(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
        tf.less_equal(halting_probability + p * still_running, threshold),
        tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = p * still_running + new_halted * remainders

    transformed_state = state

    for i in range(num_layers):
        with tf.variable_scope("rec_layer_%d" % i):
            transformed_state = encoder_layer_func(state, i)

    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) + (previous_state *
                                                         (1 - update_weights)))

    step += 1
    return (transformed_state, step, halting_probability, remainders,
            n_updates, new_state)


def _should_continue(u0, u1, halting_probability, u2, n_updates, u3, config):
    """While loop stops when this predicate is FALSE.

    I.e. all (probability < 1-eps AND counter < N) are false.

    Args:
        u0: Not used
        u1: Not used
        halting_probability: halting probability
        u2: Not used
        n_updates: ACT n_updates
        u3: Not used
        config: parameter configurations

    Returns:
        bool
    """

    threshold = config['act_threshold']
    act_max_steps = config['act_max_steps']

    del u0, u1, u2, u3
    return tf.reduce_any(
        tf.logical_and(tf.less(halting_probability, threshold),
                       tf.less(n_updates, act_max_steps)))


def _ut_encoder(state, inputs_mask, config, train):
    """Universal transformer (encoder portion).

    Args:
        state: Tensor of shape [batch_size, length, input_dim]

    Returns:
        Tensor of shape [batch_size, length, input_dim]
    """

    seq_length = tf.shape(state)[1]
    batch_size = state.shape[0]
    input_dim = state.shape[2]
    step = 0
    halting_probability = tf.zeros([batch_size, seq_length, 1])
    remainders = tf.zeros([batch_size, seq_length, 1])
    n_updates = tf.zeros([batch_size, seq_length, 1])
    previous_state = tf.zeros([batch_size, seq_length, input_dim])
    act_max_steps = config['act_max_steps']
    keep_prob = config['transformer_keep_prob']
    heads = config['transformer_heads']
    d_ff = config['transformer_ff_dims']

    def _should_continue_(u0, u1, halting_probability, u2, n_updates, u3):
        return _should_continue(u0, u1, halting_probability, u2, n_updates, u3,
                                config)

    def _encoder_layer_(x, layer_num):
        return _encoder_layer(x,
                              inputs_mask,
                              layer_num,
                              heads,
                              keep_prob,
                              d_ff,
                              config,
                              train=train)

    def _ut_function_(state, step, halting_probability, remainders, n_updates,
                      previous_state):
        return _ut_function(state, step, halting_probability, remainders,
                            n_updates, previous_state, _encoder_layer_, config)

    with tf.variable_scope("ut_encoder"):
        (_, _, _, remainder, n_updates,
         encoding) = tf.while_loop(_should_continue_,
                                   _ut_function_,
                                   (state, step, halting_probability,
                                    remainders, n_updates, previous_state),
                                   maximum_iterations=act_max_steps + 1)

    return encoding


def _encoder_model(state, config, mode):
    """
    Create the tranformer encoder model.

    Possible architectures include:
        Vanilla Transformer
        Universal Transformer

    Args:
        state: Tensor of shape [batch_size, length, input_dim]
        config: configuration params

    Returns:
        Tensor of shape [batch_size, length, input_dim]
    """

    train = (mode == 'training')
    dense_input_dim = config['transformer_dense_input_dim']
    max_length = config['num_steps']
    keep_prob = config['transformer_keep_prob']
    n_layers = config['transformer_layers']
    heads = config['transformer_heads']
    d_ff = config['transformer_ff_dims']

    state = tf.layers.dense(state, dense_input_dim, activation=tf.nn.relu)

    state_shape = tf.shape(state)
    embed_dim = state.get_shape()[2].value
    step_dim = state_shape[1]

    positional_encodings = _generate_positional_encodings(embed_dim,
                                                          seq_len=max_length)

    inputs_mask = tf.ones((1, 1, step_dim), dtype=float)

    input_embeddings = _prepare_embeddings(
        state,
        positional_encodings=positional_encodings,
        keep_prob=keep_prob,
        include_pos_encodings=config['include_pos_encodings'])

    for case in switch(config['transformer_type']):
        if case('vanilla'):
            out = _encoder(input_embeddings,
                           mask=inputs_mask,
                           n_layers=n_layers,
                           heads=heads,
                           keep_prob=keep_prob,
                           d_ff=d_ff,
                           config=config,
                           train=train)
        elif case('universal'):
            out = _ut_encoder(input_embeddings,
                              inputs_mask,
                              config,
                              train=train)
        else:
            raise ValueError('Not an available transformer type.')

    return out
