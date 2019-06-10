from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

ckpt_fpath ='./kitti-0.5x-student-distilled/model.ckpt-13500'
# ckpt_fpath = './kitti-1x-supervisor/model.ckpt-725000'
# Open TensorFlow ckpt
reader = tf.train.NewCheckpointReader(ckpt_fpath)

print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
param_map = reader.get_variable_to_shape_map()
total_count = 0
sortednames=sorted(param_map.keys(), key=lambda x:x.lower())

for k in sortednames:
    if 'Momentum' not in k and 'global_step' not in k \
             and 'iou' not in k and 'adaptation' not in k :
        temp = np.prod(param_map[k])
        total_count += temp
        print('%s: %s => %d' % (k, str(param_map[k]), temp))

print('Total Param Count: %d' % total_count)
mem = total_count*4/1024/1024
print('total memory: %4f MB' % mem)