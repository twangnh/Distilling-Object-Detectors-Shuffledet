from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.models.nn_skeleton import ModelSkeleton

# import tensorpack.models
from tensorpack.models.batch_norm import BatchNorm
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling, GlobalAvgPooling
from tensorpack.models.registry import layer_register
from tensorpack.tfutils import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models.pool import AvgPooling
from tensorpack.models.nonlin import BNReLU
from tensorpack.tfutils.tower import TowerContext

from lib.models.cross_gpu_bn import c_batch_norm


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, nl=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return nl(conv, name='output')


@under_name_scope()
def channel_shuffle(l, group):
    in_shape = l.get_shape().as_list()
    in_channel = in_shape[1]
    l = tf.reshape(l, [-1, group, in_channel // group] + in_shape[-2:])
    l = tf.transpose(l, [0, 2, 1, 3, 4])
    l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
    return l


def BN(x, name):
    return BatchNorm('bn', x)


def RELU(x, name):
    return tf.nn.relu(x)


def c_BN(x, name):
    return c_batch_norm('bn', x, '')


def c_BNReLU(x, name):
    l = c_batch_norm('bn', x, '')
    return tf.nn.relu(l)


class ShuffleDet_conv1_stride1(ModelSkeleton):
    def __init__(self, mc, gpu_id=0, without_imitation=False):
        with tf.device('/gpu:{}'.format(gpu_id)):
            ModelSkeleton.__init__(self, mc)
        self.without_imitation = without_imitation

    def model_fn(self, student=0.5):
        self._add_forward_graph(student)
        self._add_interpretation_graph()
        self._add_loss_graph()

    def _add_forward_graph(self, student=0.5):
        """NN architecture."""

        self.image_input, self.input_mask, self.box_delta_input, \
        self.box_input, self.labels, self.mimic_mask, self.mimic_mask2 = self.batch_data_queue.dequeue()

        def shufflenet_unit_supervisor(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 16 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel if stride == 1 else out_channel - in_channel,
                       kernel_shape=1, split=first_split, nl=BN)
            if stride == 1:  # unit (b)
                output = tf.nn.relu(shortcut + l)
            else:  # unit (c)
                shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
                output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
            return output

        def shufflenet_unit_add_supervisor(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 24 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel,
                       kernel_shape=1, split=first_split, nl=BN)

            output = tf.nn.relu(shortcut + l)
            return output

        def shufflenet_unit_no_shortcut_supervisor(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 24 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel,
                       kernel_shape=1, split=first_split, nl=BN)

            output = tf.nn.relu(l)
            return output

        def shufflenet_unit(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 24 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=c_BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=c_BN, stride=stride)
            l = Conv2D('conv2', l,
                       out_channel if stride == 1 else out_channel - in_channel,
                       kernel_shape=1, split=group, nl=c_BN)
            if stride == 1:  # unit (b)
                output = tf.nn.relu(shortcut + l)

            else:  # unit (c)
                shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
                output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
            return output

        def shufflenet_unit_add(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]
            shortcut = l

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 24 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=c_BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=c_BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel,
                       kernel_shape=1, split=first_split, nl=c_BN)

            output = tf.nn.relu(shortcut + l)
            return output

        def shufflenet_unit_no_shortcut(l, out_channel, group, stride):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[1]

            # We do not apply group convolution on the first pointwise layer
            # because the number of input channels is relatively small.
            first_split = group if in_channel != 24 else 1
            l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=c_BNReLU)
            l = channel_shuffle(l, group)
            l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=c_BN, stride=stride)

            l = Conv2D('conv2', l,
                       out_channel,
                       kernel_shape=1, split=first_split, nl=c_BN)

            output = tf.nn.relu(l)
            return output

        mc = self.mc
        # if mc.LOAD_PRETRAINED_MODEL:
        #   assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
        #       'Cannot find pretrained model at the given path:' \
        #       '  {}'.format(mc.PRETRAINED_MODEL_PATH)

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
             argscope(Conv2D, use_bias=False):
            with TowerContext(tf.get_default_graph().get_name_scope(), is_training=False):
                with tf.variable_scope('shuffleDet_supervisor'):

                    group = 3
                    channels = [240, 480, 960]

                    l = tf.transpose(self.image_input, [0, 3, 1, 2])
                    l = Conv2D('conv1', l, 16, 3, stride=1, nl=BNReLU)
                    l = MaxPooling('pool1', l, 3, 2, padding='SAME')

                    with tf.variable_scope('group1'):
                        for i in range(4):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit_supervisor(l, channels[0], group, 2 if i == 0 else 1)

                    with tf.variable_scope('group2'):
                        for i in range(6):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit_supervisor(l, channels[1], group, 2 if i == 0 else 1)

                    with tf.variable_scope('group3'):
                        for i in range(4):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit_supervisor(l, channels[2], group, 2 if i == 0 else 1)

                    with tf.variable_scope('added3'):
                        with tf.variable_scope('block{}'.format(0)):
                            l = shufflenet_unit_add_supervisor(l, 960, 3, 1)
                        with tf.variable_scope('block{}'.format(1)):
                            l = shufflenet_unit_no_shortcut_supervisor(l, 768, 3, 1)

                    supervisor_last_feature = tf.transpose(l, [0, 2, 3, 1])
                    self.inspect_last_feature = supervisor_last_feature

            with argscope(c_batch_norm, is_main_training_tower=int(tf.get_default_graph().get_name_scope()[-1]) == 0,
                          data_format='NCHW'):
                with TowerContext(tf.get_default_graph().get_name_scope(), is_training=mc.IS_TRAINING, index=
                int(tf.get_default_graph().get_name_scope()[-1])):
                    # with TowerContext(tf.get_default_graph().get_name_scope(), is_training=mc.IS_TRAINING):
                    group = 3
                    # channels = [120, 240, 480]
                    channels = [int(240 * student), int(480 * student), int(960 * student)]
                    l = tf.transpose(self.image_input, [0, 3, 1, 2])
                    l = Conv2D('conv1', l, 24, 3, stride=1, nl=c_BNReLU)
                    l = MaxPooling('pool1', l, 3, 2, padding='SAME')

                    with tf.variable_scope('group1'):
                        for i in range(4):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit(l, channels[0], group, 2 if i == 0 else 1)

                    with tf.variable_scope('group2'):
                        for i in range(6):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit(l, channels[1], group, 2 if i == 0 else 1)

                    with tf.variable_scope('group3'):
                        for i in range(4):
                            with tf.variable_scope('block{}'.format(i)):
                                l = shufflenet_unit(l, channels[2], group, 2 if i == 0 else 1)

                    with tf.variable_scope('added3'):
                        with tf.variable_scope('block{}'.format(0)):
                            l = shufflenet_unit_add(l, int(960 * student), 3, 1)
                        with tf.variable_scope('block{}'.format(1)):
                            l = shufflenet_unit_no_shortcut(l, int(768 * student), 3, 1)  # 768, 384, 192

                    l = tf.transpose(l, [0, 2, 3, 1])

                    with tf.variable_scope('adaptation'):
                        student_adap = self._conv_layer_no_pretrain(
                            'conv', l, filters=768, size=3, stride=1,
                            padding='SAME', xavier=False, relu=True, stddev=0.0001)
                        # student_adap = Conv2D('conv', l, 768, 3, data_format='channels_last',nl=RELU)

        ###add for mimic
        with tf.variable_scope('mimic_loss'):
            mimic_mask = tf.cast(tf.expand_dims(self.mimic_mask, axis=-1), tf.float32)
            # this normalization is maybe too harsh
            # mask mimic
            if student == 0.5:
                normalization = tf.reduce_sum(mimic_mask) * 2.
            else:
                normalization = tf.reduce_sum(mimic_mask) * 4.

            self.mimic_loss = tf.div(tf.reduce_sum(tf.square(
                supervisor_last_feature - student_adap) *
                                                   mimic_mask), normalization)
            if self.without_imitation:
                self.mimic_loss = self.mimic_loss * 0.

            tf.add_to_collection('losses', self.mimic_loss)

        dropout11 = tf.nn.dropout(l, self.keep_prob, name='drop11')

        num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
        self.preds = self._conv_layer_no_pretrain(
            'conv12', dropout11, filters=num_output, size=3, stride=1,
            padding='SAME', xavier=False, relu=False, stddev=0.0001)
        # self.preds = Conv2D('conv12', dropout11, num_output, 3, data_format='channels_last', nl=None)
