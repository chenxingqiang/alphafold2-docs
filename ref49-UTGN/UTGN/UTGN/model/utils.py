"""Utility functions.

Contains:
    switch iterator
    dictionary merger

"""

import numpy as np
import tensorflow as tf
from ast import literal_eval


class switch(object):
    """Switch iterator.

    Attributes:
        value: input value to test.
        fall: whether to stop
    """

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False


def merge_dicts(*dict_args):
    """Merges arbitrary number of dicts.

    Gives precedence to latter dicts.

    Args:
        *dict_arg: arbitrary number of dicts

    Returns:
        a single merged dict
    """

    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def ops_to_dict(session, ops):
    """Converts canonical dict of TF ops to an actual dict.

    Runs ops first.

    Args:
        session: tf session
        ops: dict mapping name to tf operation

    Returns:
        dict
    """

    dict_ = dict(list(zip(list(ops.keys()), session.run(list(ops.values())))))

    return dict_


def cum_quantile_positions(weights, quantiles=np.linspace(0.25, 0.99, 4)):
    """ Computes cumulative quantiles from curriculum weights. """
    if len(weights) != 0:
        return [next(x[0] + 1 for x
                in enumerate(np.cumsum(weights / sum(weights))) if x[1] > p)
                for p in quantiles]
    else:
        return []


def dict_to_init(dict_, seed=None, dtype=tf.float32):
    """Decide the appropriate initializer.

    Args:
        dict_: config dict
        seed: random seed
        dtype: datatype

    Returns:
        TF initialization
    """

    init_center = dict_.get('center', 0.0)
    init_range = dict_.get('range', 0.01)
    init_dist = dict_.get('dist', 'gaussian')
    init_scale = dict_.get('scale', 1.0)
    init_mode = dict_.get('mode', 'fan_in')  # also accepts fan_out, fan_avg

    for case in switch(init_dist):
        if case('gaussian'):
            init = tf.initializers.random_normal(init_center,
                                                 init_range,
                                                 seed=seed,
                                                 dtype=dtype)
        elif case('uniform'):
            init = tf.initializers.random_uniform(init_center - init_range,
                                                  init_center + init_range,
                                                  seed=seed,
                                                  dtype=dtype)
        elif case('orthogonal'):
            init = tf.initializers.orthogonal(init_scale,
                                              seed=seed,
                                              dtype=dtype)
        elif case('gaussian_variance_scaling'):
            init = tf.initializers.variance_scaling(init_scale,
                                                    init_mode,
                                                    'normal',
                                                    seed=seed,
                                                    dtype=dtype)
        elif case('uniform_variance_scaling'):
            init = tf.initializers.variance_scaling(init_scale,
                                                    init_mode,
                                                    'uniform',
                                                    seed=seed,
                                                    dtype=dtype)
    return init


def dict_to_inits(dict_, seed=None, dtype=tf.float32):
    """Decide appropriate initializer.

    Args:
        dict_: dict of config dicts
        seed: random seed
        dtype: datatype

    Returns:
        dict of TF initialization
    """

    inits = {k: dict_to_init(v, seed, dtype) for k, v in dict_.items()}

    return inits


def count_trainable_params():
    """Count the total trainable parameters. """

    total_params = [np.prod(v.get_shape().as_list())
                    for v in tf.trainable_variables()]

    return np.sum(total_params)
