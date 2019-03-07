# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import tensorflow as tf

'''
You can determine the training flags here.
'''

tf.app.flags.DEFINE_string('f', '', 'Trick to fix jupyter notebook bug')

## Code report info
tf.app.flags.DEFINE_string('model', 'VGG19', 'Model to use') # 'VGG19', 'Wide28_10'
tf.app.flags.DEFINE_string('save_dir', tf.app.flags.FLAGS.model + '_logs/', 'Saving directory')
tf.app.flags.DEFINE_string('load_dir', tf.app.flags.FLAGS.save_dir, 'Checkpoint to restore from directory')
tf.app.flags.DEFINE_integer('save_every', 5000, 'How often to save a checkpoint')
tf.app.flags.DEFINE_integer('print_every', 500, 'How often to print training progress & write summaries')

## Dataset info
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of output classes')
tf.app.flags.DEFINE_boolean('use_augmentation', True, 'Use augmentation during training')
tf.app.flags.DEFINE_integer('padding', 7, 'Augmentation padding size')

FLAGS = tf.app.flags.FLAGS
(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH) = (64, 64, 3)