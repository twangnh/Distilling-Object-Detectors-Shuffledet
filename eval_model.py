from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

from lib.config import *
from dataset_tool import kitti
from lib.utils.util import bbox_transform, Timer
from lib.models.shuffleDet import ShuffleDet_conv1_stride1, ShuffleDet_conv1_stride1_supervisor

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently support PASCAL_VOC or KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'test',
                           """Only used for VOC data."""
                           """Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                           """VOC challenge year. 2007 or 2012"""
                           """Only used for VOC data""")
tf.app.flags.DEFINE_string('eval_dir', '/home/wangtao/prj/shuffle_sup_shuffle/with_pretrain/eval1',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_string('checkpoint_path', '',
                           """Path to the training checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_string('net', 'ShuffleDet_conv1_stride1',
                           """Neural net architecture.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_float('student', 0.5, """student model, 0.5 or 0.25""")


# maintain the max mAP
MAX_MAP = 0.

def eval_once(
        saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb,
        model):
    global MAX_MAP
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root= '/home/wangtao/tfdbg_dump')

        # Restores from checkpoint
        saver.restore(sess, ckpt_path)
        all_vars = tf.global_variables()
        for var in all_vars:
            if ('mean' in var.name) or ('variance' in var.name):
                if sess.run(tf.reduce_sum(tf.cast(tf.equal(var, 0), tf.float32))) >0:
                    print (var.name)


        # Assuming model_checkpoint_path looks something like:
        #   /ckpt_dir/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt_path.split('/')[-1].split('-')[-1]

        num_images = len(imdb.image_idx)

        all_boxes = [[[] for _ in xrange(num_images)]
                     for _ in xrange(imdb.num_classes)]

        _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}

        num_detection = 0.0
        for i in xrange(num_images):
            _t['im_read'].tic()
            images, scales = imdb.read_image_batch(shuffle=False)
            _t['im_read'].toc()

            _t['im_detect'].tic()
            det_bbox, score, det_class = sess.run(
                [model.det_bbox_post, model.score_post, model.det_class_post],
                feed_dict={model.image_input: images, model.scale_eval: scales})
            _t['im_detect'].toc()

            num_detection += len(det_bbox)
            for c, b, s in zip(det_class, det_bbox, score):
                all_boxes[c][i].append(np.hstack([b,s]))

            print('im_detect: {:d}/{:d} im_read: {:.3f}s '
                  'detect: {:.3f}s '.format(
                i + 1, num_images, _t['im_read'].average_time,
                _t['im_detect'].average_time))

        print('Evaluating detections...')
        aps, ap_names = imdb.evaluate_detections(
            FLAGS.eval_dir, global_step, all_boxes)

        print('Evaluation summary:')
        print('  Average number of detections per image: {}:'.format(
            num_detection / num_images))
        print('  Timing:')
        print('    im_read: {:.3f}s detect: {:.3f}s misc: {:.3f}s'.format(
            _t['im_read'].average_time, _t['im_detect'].average_time,
            _t['misc'].average_time))
        print('  Average precisions:')

        feed_dict = {}
        for cls, ap in zip(ap_names, aps):
            feed_dict[eval_summary_phs['APs/' + cls]] = ap
            print('    {}: {:.3f}'.format(cls, ap))

        print('    Mean average precision: {:.3f}'.format(np.mean(aps)))
        if np.mean(aps) > MAX_MAP:
            MAX_MAP = np.mean(aps)
        feed_dict[eval_summary_phs['highest_APs/mAP']] = MAX_MAP
        feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)
        feed_dict[eval_summary_phs['timing/im_detect']] = \
            _t['im_detect'].average_time
        feed_dict[eval_summary_phs['timing/im_read']] = \
            _t['im_read'].average_time
        feed_dict[eval_summary_phs['timing/post_proc']] = \
            _t['misc'].average_time
        feed_dict[eval_summary_phs['num_det_per_image']] = \
            num_detection / num_images

        print('Analyzing detections...')
        stats, ims = imdb.do_detection_analysis_in_eval(
            FLAGS.eval_dir, global_step)

        eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
        for sum_str in eval_summary_str:
            summary_writer.add_summary(sum_str, global_step)


def evaluate():
    """Evaluate."""
    assert FLAGS.dataset == 'KITTI', \
        'Currently only supports KITTI dataset'

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default() as g:

        mc = kitti_shuffledet_config()
        mc.BATCH_SIZE = 1
        mc.LOAD_PRETRAINED_MODEL = False
        if FLAGS.net =='ShuffleDet_conv1_stride1_supervisor':
            model = ShuffleDet_conv1_stride1_supervisor(mc)
        elif FLAGS.net =='ShuffleDet_conv1_stride1':
            model = ShuffleDet_conv1_stride1(mc, student=FLAGS.student)

        imdb = kitti(FLAGS.image_set, './data/KITTI', mc)

        # add summary ops and placeholders
        ap_names = []
        for cls in imdb.classes:
            ap_names.append(cls + '_easy')
            ap_names.append(cls + '_medium')
            ap_names.append(cls + '_hard')

        eval_summary_ops = []
        eval_summary_phs = {}
        for ap_name in ap_names:
            ph = tf.placeholder(tf.float32)
            eval_summary_phs['APs/' + ap_name] = ph
            eval_summary_ops.append(tf.summary.scalar('APs/' + ap_name, ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['APs/mAP'] = ph
        eval_summary_ops.append(tf.summary.scalar('APs/mAP', ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['highest_APs/mAP'] = ph
        eval_summary_ops.append(tf.summary.scalar('highest_APs/mAP', ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['timing/im_detect'] = ph
        eval_summary_ops.append(tf.summary.scalar('timing/im_detect', ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['timing/im_read'] = ph
        eval_summary_ops.append(tf.summary.scalar('timing/im_read', ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['timing/post_proc'] = ph
        eval_summary_ops.append(tf.summary.scalar('timing/post_proc', ph))

        ph = tf.placeholder(tf.float32)
        eval_summary_phs['num_det_per_image'] = ph
        eval_summary_ops.append(tf.summary.scalar('num_det_per_image', ph))

        if FLAGS.net == 'ShuffleDet_conv1_stride1' or FLAGS.net == 'ShuffleDet_conv1_stride1_supervisor':
            gr = tf.global_variables(scope='g')
            global_step = tf.global_variables(scope='global')
            gr = [item for item in gr if item not in global_step]
            c = tf.global_variables(scope='conv1')
            add = tf.global_variables(scope='add')
            list = gr + c + add

            saver = tf.train.Saver(var_list=list)
        elif FLAGS.net == 'edet':
            full = tf.global_variables()
            iou = tf.global_variables(scope='iou')
            list = [item for item in full if item not in iou]
            saver = tf.train.Saver(var_list=list)
        else:
            pass
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        ckpts = set()
        while True:
            if FLAGS.run_once:
                # When run_once is true, checkpoint_path should point to the exact
                # checkpoint file.
                eval_once(
                    saver, FLAGS.checkpoint_path, summary_writer, eval_summary_ops,
                    eval_summary_phs, imdb, model)
                return
            else:
                # When run_once is false, checkpoint_path should point to the directory
                # that stores checkpoint files.
                ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    if ckpt.model_checkpoint_path in ckpts:
                        # Do not evaluate on the same checkpoint
                        print('Wait {:d}s for new checkpoints to be saved ... '
                              .format(FLAGS.eval_interval_secs))
                        time.sleep(FLAGS.eval_interval_secs)
                    else:
                        ckpts.add(ckpt.model_checkpoint_path)
                        print('Evaluating {}...'.format(ckpt.model_checkpoint_path))
                        eval_once(
                            saver, ckpt.model_checkpoint_path, summary_writer,
                            eval_summary_ops, eval_summary_phs, imdb, model)
                else:
                    print('No checkpoint file found')
                    if not FLAGS.run_once:
                        print('Wait {:d}s for new checkpoints to be saved ... '
                              .format(FLAGS.eval_interval_secs))
                        time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
