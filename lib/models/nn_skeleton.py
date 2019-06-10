from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from lib.utils import util
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf


def _add_loss_summaries(total_loss):
    """Add summaries for losses
    Generates loss summaries for visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    """
    losses = tf.get_collection('losses')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name, l)


def _variable_on_device(name, shape, initializer, trainable=True):
    """Helper to create a Variable.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


class ModelSkeleton:
    """Base class of NN detection models."""

    def __init__(self, mc):
        self.mc = mc
        # a scalar tensor in range (0, 1]. Usually set to 0.5 in training phase and
        # 1.0 in evaluation phase
        self.keep_prob = 0.5 if mc.IS_TRAINING else 1.0
        # self.add_input_graph()
        ####################################################### add for scale
        self.scale_eval = tf.placeholder(tf.float32, shape=[1, 2], name='scale_eval')

        # model parameters
        self.model_params = []

        # model size counter
        self.model_size_counter = []  # array of tuple of layer name, parameter size
        # flop counter
        self.flop_counter = []  # array of tuple of layer name, flop number
        # activation counter
        self.activation_counter = []  # array of tuple of layer name, output activations
        self.activation_counter.append(('input', mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3))


    def add_input_graph(self,training_data_gen):

        ds = tf.data.Dataset.from_generator(training_data_gen, (tf.float32, tf.float32, tf.float32,
                                                              tf.float32, tf.float32, tf.float32, tf.float32)
                                            )
        # (tf.TensorShape([8, 384, 1248, 3]), tf.TensorShape([8, 16848, 1]),
        #  tf.TensorShape([8, 16848, 4]), tf.TensorShape([8, 16848, 4]),
        #  tf.TensorShape([8, 16848, 3]))
        elem = ds.make_one_shot_iterator().get_next()
        batched_elems = tf.train.batch(
            elem,
            capacity=1000,
            batch_size=8,
            dynamic_pad=True,
            num_threads=12,
            enqueue_many=True,
            shapes=[tf.TensorShape([384, 1248, 3]), tf.TensorShape([16848, 1]),
                    tf.TensorShape([16848, 4]), tf.TensorShape([16848, 4]),
                    tf.TensorShape([16848, 3]),
                    tf.TensorShape([24, 78]), tf.TensorShape([24, 78, 9])
                    ])

        self.batch_data_queue = tf.FIFOQueue(
            capacity=200,
            dtypes=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32, tf.float32],
            shapes=[[8, self.mc.IMAGE_HEIGHT, self.mc.IMAGE_WIDTH, 3],
                    [8, self.mc.ANCHORS, 1],
                    [8, self.mc.ANCHORS, 4],
                    [8, self.mc.ANCHORS, 4],
                    [8, self.mc.ANCHORS, self.mc.CLASSES],
                    [8, 24, 78],
                    [8, 24, 78, 9]]
        )

        enqueue_op = self.batch_data_queue.enqueue(batched_elems)
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            self.batch_data_queue, [enqueue_op]*8))


    def _add_interpretation_graph(self):
        """Interpret NN output."""
        mc = self.mc

        with tf.variable_scope('interpret_output') as scope:
            preds = self.preds

            # probability
            num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES
            self.pred_class_probs = tf.reshape(
                tf.nn.softmax(
                    tf.reshape(
                        preds[:, :, :, :num_class_probs],
                        [-1, mc.CLASSES]
                    )
                ),
                [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
                name='pred_class_probs'
            )

            # confidence
            num_confidence_scores = mc.ANCHOR_PER_GRID + num_class_probs
            self.pred_conf = tf.sigmoid(
                tf.reshape(
                    preds[:, :, :, num_class_probs:num_confidence_scores],
                    [mc.BATCH_SIZE, mc.ANCHORS]
                ),
                name='pred_confidence_score'
            )

            # bbox_delta
            self.pred_box_delta = tf.reshape(
                preds[:, :, :, num_confidence_scores:],
                [mc.BATCH_SIZE, mc.ANCHORS, 4],
                name='bbox_delta'
            )

            # number of object. Used to normalize bbox and classification loss
            self.num_objects = tf.reduce_sum(self.input_mask, name='num_objects')

        with tf.variable_scope('bbox') as scope:
            with tf.variable_scope('stretching'):
                delta_x, delta_y, delta_w, delta_h = tf.unstack(
                    self.pred_box_delta, axis=2)

                anchor_x = mc.ANCHOR_BOX[:, 0]
                anchor_y = mc.ANCHOR_BOX[:, 1]
                anchor_w = mc.ANCHOR_BOX[:, 2]
                anchor_h = mc.ANCHOR_BOX[:, 3]

                box_center_x = tf.identity(
                    anchor_x + delta_x * anchor_w, name='bbox_cx')
                box_center_y = tf.identity(
                    anchor_y + delta_y * anchor_h, name='bbox_cy')
                box_width = tf.identity(
                    anchor_w * util.safe_exp(delta_w, mc.EXP_THRESH),
                    name='bbox_width')
                box_height = tf.identity(
                    anchor_h * util.safe_exp(delta_h, mc.EXP_THRESH),
                    name='bbox_height')

            with tf.variable_scope('trimming'):
                xmins, ymins, xmaxs, ymaxs = util.bbox_transform(
                    [box_center_x, box_center_y, box_width, box_height])

                # The max x position is mc.IMAGE_WIDTH - 1 since we use zero-based
                # pixels. Same for y.
                xmins = tf.minimum(
                    tf.maximum(0.0, xmins), mc.IMAGE_WIDTH - 1.0, name='bbox_xmin')
                # self._activation_summary(xmins, 'box_xmin')

                ymins = tf.minimum(
                    tf.maximum(0.0, ymins), mc.IMAGE_HEIGHT - 1.0, name='bbox_ymin')
                # self._activation_summary(ymins, 'box_ymin')

                xmaxs = tf.maximum(
                    tf.minimum(mc.IMAGE_WIDTH - 1.0, xmaxs), 0.0, name='bbox_xmax')
                # self._activation_summary(xmaxs, 'box_xmax')

                ymaxs = tf.maximum(
                    tf.minimum(mc.IMAGE_HEIGHT - 1.0, ymaxs), 0.0, name='bbox_ymax')
                # self._activation_summary(ymaxs, 'box_ymax')

                self.det_boxes = tf.transpose(
                    tf.stack(util.bbox_transform_inv([xmins, ymins, xmaxs, ymaxs])),
                    (1, 2, 0), name='bbox'
                )

        with tf.variable_scope('IOU'):
            def _tensor_iou(box1, box2):
                with tf.variable_scope('intersection'):
                    xmin = tf.maximum(box1[0], box2[0], name='xmin')
                    ymin = tf.maximum(box1[1], box2[1], name='ymin')
                    xmax = tf.minimum(box1[2], box2[2], name='xmax')
                    ymax = tf.minimum(box1[3], box2[3], name='ymax')

                    w = tf.maximum(0.0, xmax - xmin, name='inter_w')
                    h = tf.maximum(0.0, ymax - ymin, name='inter_h')
                    intersection = tf.multiply(w, h, name='intersection')

                with tf.variable_scope('union'):
                    w1 = tf.subtract(box1[2], box1[0], name='w1')
                    h1 = tf.subtract(box1[3], box1[1], name='h1')
                    w2 = tf.subtract(box2[2], box2[0], name='w2')
                    h2 = tf.subtract(box2[3], box2[1], name='h2')

                    union = w1 * h1 + w2 * h2 - intersection

                return intersection / (union + mc.EPSILON) \
                       * tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

            self.ious = _tensor_iou(
                util.bbox_transform(tf.unstack(tf.stop_gradient(self.det_boxes), axis=2)),
                util.bbox_transform(tf.unstack(self.box_input, axis=2))
            )

        with tf.variable_scope('probability') as scope:

            probs = tf.multiply(
                self.pred_class_probs,
                tf.reshape(self.pred_conf, [mc.BATCH_SIZE, mc.ANCHORS, 1]),
                name='final_class_prob'
            )

            self.det_probs = tf.reduce_max(probs, 2, name='score')
            self.det_class = tf.argmax(probs, 2, name='class_idx')


    def _add_post_process_graph(self):  # single image batch
        det_boxes, det_probs, det_class = self.det_boxes, self.det_probs, self.det_class

        # rescale
        # det_boxes_1[0, :, 0::2] /= self.scale_eval[0]
        # det_boxes_2[0, :, 1::2] /= self.scale_eval[1]
        det_boxes_1 = tf.slice(det_boxes, [0, 0, 0], [-1, -1, 1]) / self.scale_eval[0][0]
        det_boxes_2 = tf.slice(det_boxes, [0, 0, 1], [-1, -1, 1]) / self.scale_eval[0][1]
        det_boxes_3 = tf.slice(det_boxes, [0, 0, 2], [-1, -1, 1]) / self.scale_eval[0][0]
        det_boxes_4 = tf.slice(det_boxes, [0, 0, 3], [-1, -1, 1]) / self.scale_eval[0][1]
        det_boxes = tf.concat([det_boxes_1, det_boxes_2, det_boxes_3, det_boxes_4], 2)
        self.inspect_det_box = det_boxes[0]
        self.inspect_det_probs = det_probs[0]
        self.inspect_det_class = det_class[0]

        self.det_bbox_post, self.score_post, self.det_class_post = self.filter_prediction_tf_op(
            det_boxes[0], det_probs[0], det_class[0])


    def _add_loss_graph(self):
        """Define the loss operation."""
        mc = self.mc

        with tf.variable_scope('class_regression') as scope:
            # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
            # add a small value into log to prevent blowing up
            self.class_loss = tf.truediv(
                tf.reduce_sum(
                    (self.labels * (-tf.log(self.pred_class_probs + mc.EPSILON))
                     + (1 - self.labels) * (-tf.log(1 - self.pred_class_probs + mc.EPSILON)))
                    * self.input_mask * mc.LOSS_COEF_CLASS),
                self.num_objects,
                name='class_loss'
            )
            tf.add_to_collection('losses', self.class_loss)

        with tf.variable_scope('confidence_score_regression') as scope:
            input_mask = tf.reshape(self.input_mask, [mc.BATCH_SIZE, mc.ANCHORS])
            self.conf_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square((self.ious - self.pred_conf))
                    * (input_mask * mc.LOSS_COEF_CONF_POS / self.num_objects
                       + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - self.num_objects)),
                    reduction_indices=[1]
                ),
                name='confidence_loss'
            )
            tf.add_to_collection('losses', self.conf_loss)
            # with tf.device(deploy_config.optimizer_device()):
            #     tf.summary.scalar('mean iou', tf.reduce_sum(self.ious) / self.num_objects)

        with tf.variable_scope('bounding_box_regression') as scope:
            self.bbox_loss = tf.truediv(
                tf.reduce_sum(
                    mc.LOSS_COEF_BBOX * tf.square(
                        self.input_mask * (self.pred_box_delta - self.box_delta_input))),
                self.num_objects,
                name='bbox_loss'
            )
            tf.add_to_collection('losses', self.bbox_loss)

        # add above losses as well as weight decay losses to form the total loss
        self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')


    def _add_train_graph(self):
        """Define the training operation."""
        mc = self.mc

        ##modify decay
        # mc.DECAY_STEPS = 25000
        # mc.LR_DECAY_FACTOR = 0.1
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                        self.global_step,
                                        mc.DECAY_STEPS,
                                        mc.LR_DECAY_FACTOR,
                                        staircase=True)

        tf.summary.scalar('learning_rate', lr)

        _add_loss_summaries(self.loss)

        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
        grads_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
        ##Mar 16 add fixing bn gamma beta in pretrained net

        # all_trainable_exclude_pretrain_bn_gamma_beta = []
        # all_trainable = tf.trainable_variables()
        # for item in all_trainable:
        #     if ('gamma' in item.name or 'beta' in item.name) and ('added' not in item.name):
        #         pass
        #     else:
        #         all_trainable_exclude_pretrain_bn_gamma_beta.append(item)
        # grads_vars = opt.compute_gradients(self.loss, all_trainable_exclude_pretrain_bn_gamma_beta)

        with tf.variable_scope('clip_gradient') as scope:
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

        apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)
        # modify for batch norm mean and variance update during training

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ##Mar 15 remove pretrained layers' bn update, but keep added layers' bn update
        # update_ops_filtered = []
        # for item in update_ops:
        #     if item.name.startswith("added"):
        #         update_ops_filtered.append(item)
        # if update_ops_filtered:
        #     update_ops = tf.group(*update_ops_filtered)

        if update_ops:
            update_ops = tf.group(*update_ops)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        if mc.IS_TRAINING == True:
            ##Mar 15 remove pretrained layer bn update to see if we can get better performance
            with tf.control_dependencies([apply_gradient_op, update_ops]):
                self.train_op = tf.no_op(name='train')
        else:
            with tf.control_dependencies([apply_gradient_op]):
                self.train_op = tf.no_op(name='train')


    def _add_viz_graph(self):
        """Define the visualization operation."""
        mc = self.mc
        self.image_to_show = tf.placeholder(
            tf.float32, [None, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],
            name='image_to_show'
        )
        self.viz_op = tf.summary.image('sample_detection_results',
                                       self.image_to_show, collections='image_summary',
                                       max_outputs=mc.BATCH_SIZE)

    def _conv_layer_no_pretrain(
            self, layer_name, inputs, filters, size, stride, padding='SAME',
            freeze=False, xavier=False, relu=True, stddev=0.001):
        """Convolutional layer operation constructor.

        Args:
          layer_name: layer name.
          inputs: input tensor
          filters: number of output filters.
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          xavier: whether to use xavier weight initializer or not.
          relu: whether to use relu or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """

        mc = self.mc
        use_pretrained_param = False

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]

            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            if use_pretrained_param:
                if mc.DEBUG_MODE:
                    print('Using pretrained model for {}'.format(layer_name))
                kernel_init = tf.constant(kernel_val, dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(
                    stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters],
                wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            biases = _variable_on_device('biases', [filters], bias_init,
                                         trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(
                inputs, kernel, [1, stride, stride, 1], padding=padding,
                name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            self.model_size_counter.append(
                (layer_name, (1 + size * size * int(channels)) * filters)
            )
            out_shape = out.get_shape().as_list()
            num_flops = \
                (1 + 2 * int(channels) * size * size) * filters * out_shape[1] * out_shape[2]
            if relu:
                num_flops += 2 * filters * out_shape[1] * out_shape[2]
            self.flop_counter.append((layer_name, num_flops))

            self.activation_counter.append(
                (layer_name, out_shape[1] * out_shape[2] * out_shape[3])
            )

            return out



    def filter_prediction_tf_op(self, boxes, probs, cls_idx):
        """Filter bounding box predictions with probability threshold and
        non-maximum supression.

        Args:
          boxes: array of [cx, cy, w, h].
          probs: array of probabilities
          cls_idx: array of class indices
        Returns:
          final_boxes: array of filtered bounding boxes.
          final_probs: array of filtered probabilities
          final_cls_idx: array of filtered class indices
        """
        mc = self.mc

        if mc.TOP_N_DETECTION < probs.shape[0] and mc.TOP_N_DETECTION > 0:
            probs, indexes = tf.nn.top_k(probs, mc.TOP_N_DETECTION)
            boxes = tf.gather(boxes, indexes)
            cls_idx = tf.gather(cls_idx, indexes)

            # cx, cy, w, h = bbox[:]

            # y_min = cy - h / 2
            # x_min = cx - w / 2
            # y_max = cy + h / 2
            # x_max = cx + w / 2

            y_min = boxes[:, 1] - boxes[:, 3] / 2
            x_min = boxes[:, 0] - boxes[:, 2] / 2
            y_max = boxes[:, 1] + boxes[:, 3] / 2
            x_max = boxes[:, 0] + boxes[:, 2] / 2
            boxes = tf.stack([y_min, x_min, y_max, x_max], axis=1)
            # add for inspect
            self.inspect_boxes_transformed = boxes
        else:
            raise NotImplementedError

        final_boxes = []
        final_probs = []
        final_cls_idx = []

        for c in range(mc.CLASSES):
            idx_per_class = tf.reshape(tf.where((cls_idx > c - 1) & (cls_idx < c + 1)), [-1])
            probs_temp = tf.gather(probs, idx_per_class)
            boxes_temp = tf.gather(boxes, idx_per_class)

            selection = tf.image.non_max_suppression(boxes_temp, probs_temp, 50, mc.NMS_THRESH)
            final_boxes.append(tf.gather(boxes_temp, selection))
            final_probs.append(tf.gather(probs_temp, selection))
            final_cls_idx.append(tf.gather(cls_idx, tf.gather(idx_per_class, selection)))
        final_boxes = tf.concat(final_boxes, 0)
        final_boxes = tf.stack([final_boxes[:, 1], final_boxes[:, 0], final_boxes[:, 3], final_boxes[:, 2]], axis=1)
        final_probs = tf.concat(final_probs, 0)
        final_cls_idx = tf.concat(final_cls_idx, 0)

        return final_boxes, final_probs, final_cls_idx


    def filter_prediction(self, boxes, probs, cls_idx):
        """Filter bounding box predictions with probability threshold and
        non-maximum supression.

        Args:
          boxes: array of [cx, cy, w, h].
          probs: array of probabilities
          cls_idx: array of class indices
        Returns:
          final_boxes: array of filtered bounding boxes.
          final_probs: array of filtered probabilities
          final_cls_idx: array of filtered class indices
        """
        mc = self.mc

        if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
            order = probs.argsort()[:-mc.TOP_N_DETECTION - 1:-1]
            probs = probs[order]
            boxes = boxes[order]
            cls_idx = cls_idx[order]
        else:
            filtered_idx = np.nonzero(probs > mc.PROB_THRESH)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            cls_idx = cls_idx[filtered_idx]

        final_boxes = []
        final_probs = []
        final_cls_idx = []

        for c in range(mc.CLASSES):
            idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]
            keep = util.nms(boxes[idx_per_class], probs[idx_per_class], mc.NMS_THRESH)
            for i in range(len(keep)):
                if keep[i]:
                    final_boxes.append(boxes[idx_per_class[i]])
                    final_probs.append(probs[idx_per_class[i]])
                    final_cls_idx.append(c)
        return final_boxes, final_probs, final_cls_idx

