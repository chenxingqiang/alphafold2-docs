"""Neural Network tf operations for protein structure prediction.

In general this module contains functions for constructing different parts
of GeomNetModel networks, excepting ones related to geometric operations.

Conventions in this module:
    functions take both TF tensors and python objects
    python objects are fixed parameters used once for TF graph construction
    TF tensors are variables that change from data point to data point or
    iteration to iteration.

However, some funcs are somewhat loose, and would work with dynamic values
for the supposedly fixed arguments.

All these functions are strictly stateless, with no internal TF variables.
"""

import numpy as np
import tensorflow as tf

NUM_AAS = 20
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3


def masking_matrix(mask, name=None):
    """Constructs a masking matrix.

    It zeros out pairwise distances due to missing residues or padding.
    This is called for each individual sequence, and so it's folded in
    the reading/queuing pipeline for performance reasons.

    Args:
        mask: 0/1 vector

    Returns:
        A square matrix with 1s except for rows and cols
        whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope(name, 'masking_matrix', [mask]) as scope:
        mask = tf.convert_to_tensor(mask, name='mask')

        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)
        return matrix_mask


def effective_steps(masks, num_edge_residues, name=None):
    """Find the effective number of steps.

    (Number of residues that are non-missing and are not just padding.)

    Args:
        masks: A batch of square masking matrices
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH, BATCH_SIZE]

    Returns:
        A vector with the effective number of steps
        [BATCH_SIZE]
    """

    with tf.name_scope(name, 'effective_steps', [masks]) as scope:
        masks = tf.convert_to_tensor(masks, name='masks')

        traces = tf.matrix_diag_part(tf.transpose(masks, [2, 0, 1]))
        eff_stepss = tf.add(tf.reduce_sum(traces, [1]),
                            num_edge_residues,
                            name=scope)
        # NUM_EDGE_RESIDUES shouldn't be here, but I'm keeping it for
        # legacy reasons. Just be clear that it's _always_ wrong to have
        # it here, even when it's not equal to 0.

        return eff_stepss


def read_protein(filename_queue,
                 max_length,
                 num_edge_residues,
                 num_evo_entries,
                 name=None):
    """Reads and parses a protein TF Record.

    Map primary sequences to 20-dim 1-hot vectors.
    Map evolutionary sequences to num_evo_entries-dim vectors.
    Map secondary structures to ints indicating one of 8 class labels.
    Flatten tertiary coordinates (there are 3x many coordinates as residues.)

    Evolutionary, secondary, and tertiary entries are optional.

    Args:
        filename_queue: TF queue for reading files
        max_length: Fixed maximum length of the sequences.

    Returns:
        id_: string identifier of record
        one_hot_primary: AA sequence as one-hot vectors [SEQ_LEN, SEQ_LEN]
        evolutionary: PSSM sequence as vectors
        secondary: DSSP sequence as int class labels
        tertiary: 3D structure coordinates
        matrix_mask: zeros out pairwise distances in the masked regions
        pri_length: Length of amino acid sequence
        keep: True if primary length is less than or equal to max_length
    """

    with tf.name_scope(name, 'read_protein', []) as scope:
        # Set up reader and read
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Parse TF Record
        seq_feature = tf.FixedLenSequenceFeature
        context, features = tf.parse_single_sequence_example(
            serialized_example,
            context_features={'id': tf.FixedLenFeature((1, ), tf.string)},
            sequence_features={
                'primary':
                seq_feature((1, ), tf.int64),
                'evolutionary':
                seq_feature((num_evo_entries, ),
                            tf.float32,
                            allow_missing=True),
                'secondary':
                seq_feature((1, ), tf.int64, allow_missing=True),
                'tertiary':
                seq_feature((NUM_DIMENSIONS, ), tf.float32,
                            allow_missing=True),
                'mask':
                seq_feature((1, ), tf.float32, allow_missing=True)
            })
        id_ = context['id'][0]
        primary = tf.to_int32(features['primary'][:, 0])
        evolutionary = features['evolutionary']
        secondary = tf.to_int32(features['secondary'][:, 0])
        tertiary = features['tertiary']
        mask = features['mask'][:, 0]

        # Predicate for when to retain protein
        pri_length = tf.size(primary)
        keep = pri_length <= max_length

        # Convert primary to one-hot
        one_hot_primary = tf.one_hot(primary, NUM_AAS)

        # Generate tertiary masking matrix.
        # If mask is missing then assume all residues are present
        mask = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf
                       .ones([pri_length - num_edge_residues]))
        ter_mask = masking_matrix(mask, name='ter_mask')
        return (id_, one_hot_primary, evolutionary, secondary,
                tertiary, ter_mask, pri_length, keep)


def curriculum_weights(base, slope, max_seq_length, name=None):
    """Creates a tensor of weights based on the current curriculum.

    Parametrized by base and slope.

    Args:
        base: scalar TF tensor that changes as training progresses.
        slope: Fixed value (not a TF tensor).
        max_seq_length: Fixed value (not a TF tensor).

    Returns:
        [MAX_SEQ_LENGTH - 1]
        The minus one factor is because we ignore self-distances.
    """

    with tf.name_scope(name, 'curriculum_weights', [base]) as scope:
        base = tf.convert_to_tensor(base, name='base')
        steps = tf.to_float(tf.range(max_seq_length - 1))
        weights = tf.sigmoid(-(slope * (steps - base)), name=scope)
        return weights


def weighting_matrix(weights, name=None):
    """ Creates a weighting matrix.

    The ith weight is in the ith upper diagonal of the matrix.
    All other entries are 0.
    This functions is called once per curriculum update / iteration,
    but then used for the entire batch.

    Args:
        weights: Curriculum weights. Changes as curriculum progresses.
                 [MAX_SEQ_LENGTH - 1]
        name: name for creating the weighting matrix

    Returns:
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope(name, 'weighting_matrix', [weights]) as scope:
        weights = tf.convert_to_tensor(weights, name='weights')

        max_seq_length = weights.get_shape().as_list()[0] + 1
        split_indices = np.diag_indices(max_seq_length)

        flat_indices = []
        flat_weights = []
        for i in range(max_seq_length - 1):
            indices_subset = np.concatenate(
                (split_indices[0][:-(i + 1), np.newaxis],
                 split_indices[1][i + 1:, np.newaxis]), 1)
            weights_subset = tf.fill([len(indices_subset)], weights[i])
            flat_indices.append(indices_subset)
            flat_weights.append(weights_subset)
        flat_indices = np.concatenate(flat_indices)
        flat_weights = tf.concat(flat_weights, 0)

        mat = tf.sparse_to_dense(flat_indices,
                                 [max_seq_length, max_seq_length],
                                 flat_weights,
                                 validate_indices=False,
                                 name=scope)

        return mat


def id_filter(ids, filter_string, delimiter='#', name=None):
    """Create a filter for id.

    Args:
        ids: list of ids
        filter_string: string to seek
        delimiter: seperation
        name: name of scope

    Returns:
        a boolean mask corresponding to the chosen id filter
    """

    with tf.name_scope(name, 'id_filter', [ids, filter_string]) as scope:
        ids = tf.convert_to_tensor(ids, name='ids')
        filter_string = tf.convert_to_tensor(filter_string,
                                             name='filter_string')

        bool_mask = tf.equal(tf.string_split(ids,
                                             delimiter=delimiter).values[0::2],
                             filter_string,
                             name=scope)
        return bool_mask
