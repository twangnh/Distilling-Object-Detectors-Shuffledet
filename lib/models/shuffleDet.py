from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from lib.models.nn_skeleton_eval import ModelSkeleton

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

def c_BN(x, name):
    return c_batch_norm('bn', x, '')

def c_BNReLU(x, name):
    l = c_batch_norm('bn',x, '')
    return tf.nn.relu(l)

class ShuffleDet_conv1_stride1(ModelSkeleton):
  def __init__(self, mc, gpu_id=0, student=0.5):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph(student)
      self._add_interpretation_graph()
      self._add_post_process_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()



  def _add_forward_graph(self, student=0.5):
    """NN architecture."""

    def shufflenet_unit(l, out_channel, group, stride):
        in_shape = l.get_shape().as_list()
        in_channel = in_shape[1]
        shortcut = l

        first_split = group if in_channel != 24 else 1
        l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
        l = channel_shuffle(l, group)
        l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)
        l = Conv2D('conv2', l,
                   out_channel if stride == 1 else out_channel - in_channel,
                   kernel_shape=1, split=group, nl=BN)
        if stride == 1:
            output = tf.nn.relu(shortcut + l)

        else:
            shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
            output = tf.concat([shortcut, tf.nn.relu(l)], axis=1)
        return output
    def shufflenet_unit_add(l, out_channel, group, stride):
        in_shape = l.get_shape().as_list()
        in_channel = in_shape[1]
        shortcut = l

        first_split = group if in_channel != 24 else 1
        l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
        l = channel_shuffle(l, group)
        l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)

        l = Conv2D('conv2', l,
                   out_channel,
                   kernel_shape=1, split=first_split, nl=BN)

        output = tf.nn.relu(shortcut + l)
        return output

    def shufflenet_unit_no_shortcut(l, out_channel, group, stride):
        in_shape = l.get_shape().as_list()
        in_channel = in_shape[1]


        first_split = group if in_channel != 24 else 1
        l = Conv2D('conv1', l, out_channel // 4, kernel_shape=1, split=first_split, nl=BNReLU)
        l = channel_shuffle(l, group)
        l = DepthConv('dconv', l, out_channel // 4, kernel_shape=3, nl=BN, stride=stride)

        l = Conv2D('conv2', l,
                   out_channel,
                   kernel_shape=1, split=first_split, nl=BN)

        output = tf.nn.relu(l)
        return output


    mc = self.mc

    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
         argscope(Conv2D, use_bias=False):
        with TowerContext('', is_training=mc.IS_TRAINING):
            group = 3
            channels = [int(240 * student), int(480 * student), int(960 * student)]
            l = tf.transpose(self.image_input, [0, 3, 1, 2])
            l = Conv2D('conv1', l, 24, 3, stride=1, nl=BNReLU)
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
                    l = shufflenet_unit_add(l, int(960*student), 3, 1)
                with tf.variable_scope('block{}'.format(1)):
                    l = shufflenet_unit_no_shortcut(l, int(768*student), 3, 1)#768, 384, 192

    l = tf.transpose(l,[0, 2, 3, 1])
    dropout11 = tf.nn.dropout(l, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)

    self.preds = self._conv_layer_no_pretrain(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)


class ShuffleDet_conv1_stride1_supervisor(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_interpretation_graph()
      self._add_post_process_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()



  def _add_forward_graph(self):
    """NN architecture."""

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

    with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'), \
         argscope(Conv2D, use_bias=False):
        with TowerContext('', is_training=mc.IS_TRAINING):
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

    l = tf.transpose(l,[0, 2, 3, 1])
    dropout11 = tf.nn.dropout(l, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    # modify for shuffleunit det head
    self.preds = self._conv_layer_no_pretrain(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)
    # self.preds = Conv2D('conv12', dropout11, num_output, 3, data_format='channels_last', nl=None)


