# -*- coding: utf-8 -*-
# File: batch_norm.py


import re

import six
import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
import logging
logging.getLogger().level = logging.INFO
from tensorflow.contrib.framework import add_arg_scope
from tensorpack.models.registry import layer_register
__all__ = ['c_batch_norm']


def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [n_out], initializer=beta_init)
    else:
        beta = tf.zeros([n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    with tf.device('/cpu:0'):
        update_op1 = moving_averages.assign_moving_average(
            moving_mean, batch_mean, decay, zero_debias=False,
            name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
            moving_var, batch_var, decay, zero_debias=False,
            name='var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return tf.identity(xn, name='output')


@layer_register(log_shape=True)
def c_batch_norm(inputs, scope, training=None, is_main_training_tower=True, axis=None,
              momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              beta_initializer=tf.zeros_initializer(),
              gamma_initializer=tf.ones_initializer(),
              virtual_batch_size=None,
              data_format='NCHW',
              internal_update=False,
              sync_statistics='nccl'):
    """
    Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
    in the following:

    1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
    2. Default value for `momentum` and `epsilon` is different.
    3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
    4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
    5. Support the `sync_statistics` option, which is very useful in small-batch models.

    Args:
        internal_update (bool): if False, add EMA update ops to
            `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer
            by control dependencies.
            They are very similar in speed, but `internal_update=True` can be used
            when you have conditionals in your model, or when you have multiple networks to train.
        sync_statistics: either None or "nccl". By default (None), it uses statistics of the input tensor to normalize.
            When set to "nccl", this layer must be used under tensorpack multi-gpu trainers,
            and it then uses per-machine (multiple GPU) statistics to normalize.

            This option has no effect when not training.
            The option is also known as "Cross-GPU BatchNorm" as mentioned in https://arxiv.org/abs/1711.07240.

    Variable Names:

    * ``beta``: the bias term. Will be zero-inited by default.
    * ``gamma``: the scale term. Will be one-inited by default. Input will be transformed by ``x * gamma + beta``.
    * ``mean/EMA``: the moving average of mean.
    * ``variance/EMA``: the moving average of variance.

    Note:
        1. Combinations of ``training`` and ``ctx.is_training``:
            * ``training == ctx.is_training``: standard BN, EMA are
                maintained during training and used during inference. This is
                the default.
            * ``training and not ctx.is_training``: still use batch statistics in inference.
            * ``not training and ctx.is_training``: use EMA to normalize in
                training. This is useful when you load a pre-trained BN and
                don't want to fine tune the EMA. EMA will not be updated in
                this case.
    """
    # parse shapes

    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4], ndims
    if sync_statistics is not None:
        sync_statistics = sync_statistics.lower()
    assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

    if axis is None:
        if ndims == 2:
            data_format = 'NHWC'
            axis = 1
        else:
            axis = 1 if data_format == 'NCHW' else 3
    else:
        data_format = 'NCHW' if axis == 1 else 'NHWC'
    num_chan = shape[axis]


    if sync_statistics is None:

        raise ValueError
    else:
        red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])

        new_shape = None
        if ndims == 4 and axis == 1:
            new_shape = [1, num_chan, 1, 1]

        batch_mean = tf.reduce_mean(inputs, axis=red_axis)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)
        # for debuging cgbn
        # tower_number = is_main_training_tower
        #is_main_training_tower = (is_main_training_tower == 0)
        # batch_mean =tf.Print(batch_mean, [batch_mean], 'batch_norm_mean %s' %tower_number)
        # batch_mean_square =tf.Print(batch_mean_square, [batch_mean_square], 'batch_norm_var %s' %tower_number)

        if sync_statistics == 'nccl':
            if six.PY3 and is_main_training_tower:
                logging.warn("A TensorFlow bug will cause cross-GPU BatchNorm to fail. "
                            "Apply this patch: https://github.com/tensorflow/tensorflow/pull/20360")

            from tensorflow.contrib.nccl.ops import gen_nccl_ops
            with tf.variable_scope(scope):
                shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
            num_dev = 4
            batch_mean = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
            batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                input=batch_mean_square,
                reduction='sum',
                num_devices=num_dev,
                shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
            # if is_main_training_tower:
            #     batch_mean=tf.Print(batch_mean, [batch_mean], 'batch_norm_mean' )
            #     batch_mean_square =tf.Print(batch_mean_square, [batch_mean_square], 'batch_norm_var')

        elif sync_statistics == 'horovod':
            # Require https://github.com/uber/horovod/pull/331
            # Proof-of-concept, not ready yet.
            import horovod.tensorflow as hvd
            batch_mean = hvd.allreduce(batch_mean, average=True)
            batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
        batch_var = batch_mean_square - tf.square(batch_mean)
        batch_mean_vec = batch_mean
        batch_var_vec = batch_var

        beta, gamma, moving_mean, moving_var = get_bn_variables(
            num_chan, scale, center, beta_initializer, gamma_initializer)
        if new_shape is not None:
            batch_mean = tf.reshape(batch_mean, new_shape)
            batch_var = tf.reshape(batch_var, new_shape)
            r_gamma = tf.reshape(gamma, new_shape)
            r_beta = tf.reshape(beta, new_shape)
        else:
            r_gamma, r_beta = gamma, beta
        xn = tf.nn.batch_normalization(
            inputs, batch_mean, batch_var, r_beta, r_gamma, epsilon)
        if is_main_training_tower:
            ret = update_bn_ema(
                xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var,
                momentum, internal_update)
        else:
            ret = tf.identity(xn, name='output')
    return ret



