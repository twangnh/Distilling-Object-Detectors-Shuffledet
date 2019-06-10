from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os.path

from six.moves import xrange
import tensorflow as tf

from lib.config import *
from dataset_tool import kitti
from lib.utils.util import bbox_transform
from lib.models.shuffleDet_sup import ShuffleDet_conv1_stride1
from lib.utils import model_deploy
import tensorflow.contrib.slim as slim
import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('year', '2007',
                           """VOC challenge year. 2007 or 2012"""
                           """Only used for Pascal VOC dataset""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/logs/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'ShuffleDet_conv1_stride1',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 50,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('print_step', 20,
                            """Number of steps to print.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 500,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_float('student', 0.5, """student model, 0.5 or 0.25""")
tf.app.flags.DEFINE_bool('without_imitation', False, """whether to turn off imitation loss""")


def train():
    assert FLAGS.dataset == 'KITTI', \
        'Currently only support KITTI dataset'


    mc = kitti_shuffledet_config()
    mc.IS_TRAINING = True
    mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
    model = ShuffleDet_conv1_stride1(mc, without_imitation=FLAGS.without_imitation)


    with tf.Graph().as_default():

        config = model_deploy.DeploymentConfig(num_clones=4)
        with tf.device(config.inputs_device()):
            mc.BATCH_SIZE = 1
            imdb = kitti('train', './data/KITTI', mc)
            model.add_input_graph(imdb.next_batch)

        mc.BATCH_SIZE = 8
        with tf.device(config.variables_device()):
            global_step = slim.create_global_step()
        with tf.device(config.optimizer_device()):

            lr = tf.train.cosine_decay(0.02,
                                       global_step,
                                       40000,
                                       0.0000001)
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        tf.summary.scalar('learning_rate', lr)
        def freeze_variable_func():
            freeze_vars = tf.global_variables(scope='shuffleDet_supervisor')
            # exclude_vars = tf.global_variables(scope='sqDet')
            all_trainable_vars = tf.trainable_variables()
            train_vars = [item for item in all_trainable_vars if item not in freeze_vars]
            return train_vars

        model_dp = model_deploy.deploy(config, model.model_fn,
                                       args=[FLAGS.student],
                                       optimizer=optimizer, freeze_variable_func=freeze_variable_func)

        ## code a bit ugly here, will improve future.
        ## to separate the varibles that are initalized from pretrained model and the variables that needs to be randomly intialized
        full = tf.global_variables()
        g = tf.global_variables(scope='g')
        global_step = tf.global_variables(scope='global')
        g = [item for item in g if item not in global_step]
        c = tf.global_variables(scope='conv1')
        conv12 = tf.global_variables(scope='conv12')
        c = [item for item in c if item not in conv12]
        list = g + c

        momentum_list = []
        for item in list:
            if 'Mom' in item.op.name:
                momentum_list.append(item)
        list = [item for item in list if item not in momentum_list]
        list_to_be_initialized = [item for item in full if item not in list]



        init_saver = tf.train.Saver(var_list=list)

        saver = tf.train.Saver()
        def init_fn(sess):

            init_saver.restore(sess, FLAGS.pretrained_model_path)
            init = tf.variables_initializer(var_list=list_to_be_initialized)
            sess.run(init)
            tf.train.start_queue_runners(sess=sess)
        sess = tf.Session()
        init_fn(sess)

        ## restore supervisor
        vars_shuffleDet_supervisor = tf.global_variables(scope='shuffleDet_supervisor')
        vars_shuffleDet_supervisor_replace = {var.op.name.replace('shuffleDet_supervisor/', ''): var for var in vars_shuffleDet_supervisor}

        saver_superviser = tf.train.Saver(var_list=vars_shuffleDet_supervisor_replace)
        saver_superviser.restore(sess, './kitti-1x-supervisor/model.ckpt-725000')

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):

            # Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % FLAGS.summary_step == 0:

                op_list = [
                    model_dp.train_op, summary_op
                ]
                loss, summary_str = sess.run(op_list)

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            else:
                loss = sess.run(model_dp.train_op)

            if step%FLAGS.print_step ==0:
                print('step: {} total_loss: {} time: {}'.format(step, loss, datetime.datetime.now()))


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
