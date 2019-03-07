# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import tensorflow as tf
from GraphBuilder.layers import *

def build_wide2810_graph(X, num_classes, is_training, hparams):
	'''
	Creates graph for WideResNet28-10 architecture. https://arxiv.org/pdf/1605.07146v1.pdf

	Inputs:
	- X: Placeholder for minibatch of training data.
	- num_classes: Number of classes our model needs to predict.
	- is_training: (Boolean) flag which determines if we are training the model.
	- hparams: HParams object of hyperparameters. Not used here yet (ignore for now).

	Returns:
	- scores: (Tensor) Scores for each class label.
	'''
	with tf.variable_scope('conv0'):
		conv0 = conv(X, num_filters=16, kernel_size=3)


	### Block 1
	with tf.variable_scope('wide_residual_1_0'):
		wr1_0 = wide_residual(conv0, num_filters=160, stride=1,
			input_relu=True, is_training=is_training)

	with tf.variable_scope('wide_residual_1_1'):
		wr1_1 = wide_residual(wr1_0, num_filters=160, is_training=is_training)

	with tf.variable_scope('wide_residual_1_2'):
		wr1_2 = wide_residual(wr1_1, num_filters=160, is_training=is_training)

	with tf.variable_scope('wide_residual_1_3'):
		wr1_3 = wide_residual(wr1_2, num_filters=160, is_training=is_training)


	### Block 2
	with tf.variable_scope('wide_residual_2_0'):
		wr2_0 = wide_residual(wr1_3, num_filters=320, stride=2,
			input_relu=False, is_training=is_training)

	with tf.variable_scope('wide_residual_2_1'):
		wr2_1 = wide_residual(wr2_0, num_filters=320, is_training=is_training)

	with tf.variable_scope('wide_residual_2_2'):
		wr2_2 = wide_residual(wr2_1, num_filters=320, is_training=is_training)

	with tf.variable_scope('wide_residual_2_3'):
		wr2_3 = wide_residual(wr2_2, num_filters=320, is_training=is_training)


	### Block 3
	with tf.variable_scope('wide_residual_3_0'):
		wr3_0 = wide_residual(wr2_3, num_filters=640, stride=2,
			input_relu=False, is_training=is_training)

	with tf.variable_scope('wide_residual_3_1'):
		wr3_1 = wide_residual(wr3_0, num_filters=640, is_training=is_training)

	with tf.variable_scope('wide_residual_3_2'):
		wr3_2 = wide_residual(wr3_1, num_filters=640, is_training=is_training)

	with tf.variable_scope('wide_residual_3_3'):
		wr3_3 = wide_residual(wr3_2, num_filters=640, is_training=is_training)


	### Global Pool
	with tf.variable_scope('average_pool'):
		a = bn_relu(wr3_3, is_training)
		gap = global_average_pool(a)


	### Flatten Activation Maps
	gap_flat = flatten(gap)


	## FC
	with tf.variable_scope('scores'):
		scores = fc(gap_flat, num_classes)

	return scores