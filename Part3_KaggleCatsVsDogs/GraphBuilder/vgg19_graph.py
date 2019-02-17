# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import tensorflow as tf
from GraphBuilder.layers import *

def build_vgg19_graph(X, num_classes, is_training, hparams):
	'''
	Creates graph for VGG19 architecture. https://arxiv.org/pdf/1409.1556.pdf

	Inputs:
	- X: Placeholder for minibatch of training data.
	- num_classes: Number of classes our model needs to predict.
	- is_training: (Boolean) flag which determines if we are training the model.
	- hparams: HParams object of hyperparameters. Used for dropout probability.

	Returns:
	- scores: (Tensor) Scores for each class label.
	'''
	
	### Conv Block 1
	with tf.variable_scope('conv1_1'):
		conv1_1 = conv_bn_relu(X, num_filters=64, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)
		
	with tf.variable_scope('conv1_2'):
		conv1_2 = conv_bn_relu(conv1_1, num_filters=64, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)
	
	with tf.name_scope('block1_max_pool'):
		block1_out = tf.layers.max_pooling2d(conv1_2, 2, 2)


	### Conv Block 2
	with tf.variable_scope('conv2_1'):
		conv2_1 = conv_bn_relu(block1_out, num_filters=128, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv2_2'):
		conv2_2 = conv_bn_relu(conv2_1, num_filters=128, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.name_scope('block2_max_pool'):
		block2_out = tf.layers.max_pooling2d(conv2_2, 2, 2)
	

	### Conv Block 3
	with tf.variable_scope('conv3_1'):
		conv3_1 = conv_bn_relu(block2_out, num_filters=256, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv3_2'):
		conv3_2 = conv_bn_relu(conv3_1, num_filters=256, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv3_3'):
		conv3_3 = conv_bn_relu(conv3_2, num_filters=256, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv3_4'):
		conv3_4 = conv_bn_relu(conv3_3, num_filters=256, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.name_scope('block3_max_pool'):
		block3_out = tf.layers.max_pooling2d(conv3_4, 2, 2)
	

	### Conv Block 4
	with tf.variable_scope('conv4_1'):
		conv4_1 = conv_bn_relu(block3_out, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv4_2'):
		conv4_2 = conv_bn_relu(conv4_1, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv4_3'):
		conv4_3 = conv_bn_relu(conv4_2, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv4_4'):
		conv4_4 = conv_bn_relu(conv4_3, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.name_scope('block4_max_pool'):
		block4_out = tf.layers.max_pooling2d(conv4_4, 2, 2)


	### Conv Block 5
	with tf.variable_scope('conv5_1'):
		conv5_1 = conv_bn_relu(block4_out, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv5_2'):
		conv5_2 = conv_bn_relu(conv5_1, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv5_3'):
		conv5_3 = conv_bn_relu(conv5_2, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.variable_scope('conv5_4'):
		conv5_4 = conv_bn_relu(conv5_3, num_filters=512, kernel_size=3, strides=1,
			padding='SAME', is_training=is_training)

	with tf.name_scope('block5_max_pool'):
		block5_out = tf.layers.max_pooling2d(conv5_4, 2, 2)
	

	### Flatten Activation Maps
	block5_flat = flatten(block5_out)

	## FC 1
	with tf.variable_scope('fc1'):
		fc1 = fc_bn(block5_flat, num_outputs=4096, is_training=is_training)

	with tf.name_scope('fc1_dropout'):
		fc1_drop = dropout(fc1, hparams.dropout_probability, is_training=is_training)


	## FC 2
	with tf.variable_scope('fc2'):
		fc2 = fc_bn(fc1_drop, num_outputs=4096, is_training=is_training)

	with tf.name_scope('fc2_dropout'):
		fc2_drop = dropout(fc2, hparams.dropout_probability, is_training=is_training)


	## FC 3
	with tf.variable_scope('scores'):
		scores = fc_bn(fc2_drop, num_outputs=num_classes, is_training=is_training,
			use_relu=False)

	return scores