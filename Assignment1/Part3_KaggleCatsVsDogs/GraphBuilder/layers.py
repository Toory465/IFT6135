# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import tensorflow as tf

def flatten(inputs):
	'''
	Reshapes input tensors of size (N,H,W,C) to (N, H*W*C)

	Inputs:
	- inputs: Inputs of size (N,H,W,C)

	Returns:
	- inputs_flat: Reshaped inputs of size (N, H*W*C)
	'''
	inputs_shape = inputs.get_shape().as_list()
	inputs_flat = tf.reshape(inputs, (-1, np.prod(inputs_shape[1:])), name='flatten')
	return inputs_flat

def fc(inputs, num_outputs):
	'''
	Fully connected layer.
	
	Inputs:
	- inputs: Input tensor of shape (N, Din) that we want to apply the transformation to
	- num_outputs: Number of output dimensions (Dout)
	- use_relu: Boolean of whether to use relu nonlinearity or no nonlinearity at all.

	Returns:
	- output: Affine output.
	'''
	fc_output = tf.layers.dense(inputs, num_outputs, use_bias=True, kernel_initializer=tf.keras.initializers.he_uniform())
	return fc_output

def fc_bn(inputs, num_outputs, is_training, use_relu=True):
	'''
	Fully connected layer with batch normalization and (optional) nonlinearity.
	
	Inputs:
	- inputs: Input tensor of shape (N, Din) that we want to apply the transformation to
	- num_outputs: Number of output dimensions (Dout)
	- is_training: Boolean of whether model is in training mode or not (used for BN)
	- use_relu: Boolean of whether to use relu nonlinearity or no nonlinearity at all.

	Returns:
	- post_activation: Output of shape (N, Dout)
	'''
	fc_output = fc(inputs, num_outputs)
	fc_bn_output = batch_norm(fc_output, is_training)
	post_activation = tf.nn.relu(fc_bn_output) if use_relu else fc_bn_output
	return post_activation

def conv(inputs, num_filters, kernel_size, strides=1, padding='SAME'):
	'''
	Simple convolutional layer without nonlinearity.

	Inputs:
	- inputs: Input tensor of shape (N,H,W,C).
	- num_filters: Number of kernels to be used.
	- kernel_size: Size of kernel. Can be a list of 2 integers or a single integer if the kernel
	has the same height and width.
	- strides: Stride to use when applying the convolution. Can be a list of 2 integers or a
	single integer if we want to use the same stride in both horizontal and vertical directions.
	- padding: What type of padding to use ('SAME' or 'VALID').

	Returns:
	- output: Output of convolving the specified filters with the inputs.
	'''
	return tf.layers.conv2d(inputs, num_filters, kernel_size, strides, padding,
		kernel_initializer=tf.keras.initializers.he_normal())

def conv_bn_relu(inputs, num_filters, kernel_size, strides, padding, is_training):
	'''
	Convolutional layer with batch normalization and relu nonlinearity.

	Inputs:
	- inputs: Input tensor of shape (N,H,W,C).
	- num_filters: Number of kernels to be used.
	- kernel_size: Size of kernel. Can be a list of 2 integers or a single integer if the kernel
	has the same height and width.
	- strides: Stride to use when applying the convolution. Can be a list of 2 integers or a
	single integer if we want to use the same stride in both horizontal and vertical directions.
	- padding: What type of padding to use ('SAME' or 'VALID').
	- is_training: Boolean of whether model is in training mode or not (used for BN).

	Returns:
	- output: Output of convolving the specified filters with the inputs.
	'''
	conv_output = conv(inputs, num_filters, kernel_size, strides, padding)
	output = bn_relu(conv_output, is_training)
	return output

def wide_residual(inputs, num_filters, stride=1, input_relu=False, is_training=True):
	'''
	Residual block graph with 2 3x3 convolutional layers.

	Inputs:
	- inputs: Input tensor of shape (N,H,W,C).
	- num_filters: Number of kernels to be used.
	- strides: Stride to use when applying the convolution. Can be a list of 2 integers or a
	single integer if we want to use the same stride in both horizontal and vertical directions.

	Returns:
	- out: Output of wide residual block, of shape (N,H',W',num_filters).
	'''
	num_channels = inputs.get_shape().as_list()[-1]

	a0 = bn_relu(inputs, is_training)
	identity = a0 if input_relu else inputs

	with tf.variable_scope('conv1'):
		conv1 = conv(a0, num_filters, kernel_size=3, strides=stride)
		a1 = bn_relu(conv1, is_training)

	with tf.variable_scope('conv2'):
		conv2 = conv(a1, num_filters, kernel_size=3)

	with tf.variable_scope('total'):
		if stride != 1: # Reduce identity size if identity is smaller than conv output.
			identity = average_pool(identity, kernel_size=stride, strides=stride)
		if num_channels < num_filters: # Add extra zero channels if num_channels < num_filtes.
			identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0],
				[(num_filters-num_channels)//2, (num_filters-num_channels)//2]])
		out = conv2 + identity
	return out

def batch_normalization(inputs, mean, var, beta, gamma, epsilon=0.001):
	'''
	Applies batch normalization to inputs based on given mean, var, beta and gamma:
	output = gamma * ((inputs - mean) / var) + beta

	Inputs:
	- inputs: Input tensor that we want to apply Batch Normalization to, could be of
	shape (N,H,W,C) or (N,D).
	- mean: Tensor which holds the means of inputs tensor. Either of shape (C) or (D).
	- var: Tensor which holds the variances of inputs tensor. Either of shape (C) or (D).
	- beta: Batch normalization beta variable (tensor). Either of shape (C) or (D).
	- gamma: Batch normalization gamma variable (tensor). Either of shape (C) or (D).
	- epsilon: Epsilon to add to variance for numerical stability.

	Returns:
	- output: Batch normalized inputs
	'''
	output = (gamma * tf.div(inputs - mean, tf.sqrt(var + epsilon))) + beta
	return output

def batch_norm(inputs, is_training, decay=0.99):
	'''
	Applies Batch Normalization to inputs.

	Inputs:
	- inputs: Input tensor that we want to apply Batch Normalization to, could be of
	shape (N,H,W,C) or (N,D).
	- is_training: Boolean flag which determines whether we are in training mode or in
	inference mode. Note that BN performs differently in these 2 cases.
	- decay (Optional): float which determines BN's decay used for computing the moving
	mean and moving standard deviation.

	Returns:
	- Batch normalized inputs
	'''
	inputs_shape = inputs.get_shape().as_list()
	num_nodes = inputs_shape[-1]
	axes = list(range(len(inputs_shape)-1))

	gamma = tf.get_variable('gamma', num_nodes, tf.float32,
		initializer=tf.constant_initializer(1.0, tf.float32))
	beta = tf.get_variable('beta', num_nodes, tf.float32,
						   initializer=tf.constant_initializer(0.0, tf.float32))
	
	pop_mean = tf.get_variable('test_mean', shape=num_nodes, dtype=tf.float32,
		initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
	pop_var = tf.get_variable('test_var', shape=num_nodes, dtype=tf.float32,
		initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
	
	if is_training:
		batch_mean, batch_var = tf.nn.moments(inputs,axes=axes)
		train_mean = tf.assign(pop_mean,
							   (pop_mean * decay) + (batch_mean * (1 - decay)))
		train_var = tf.assign(pop_var,
							  (pop_var * decay) + (batch_var * (1 - decay)))
		with tf.control_dependencies([train_mean, train_var]):
			return batch_normalization(inputs, batch_mean, batch_var, beta, gamma)
	else:
		return batch_normalization(inputs, pop_mean, pop_var, beta, gamma)

def bn_relu(inputs, is_training):
	'''
	Applies batch normalization and then ReLU.

	Inputs:
	- inputs: Input tensor of shape (N,H,W,C) or (N,D).
	- is_training: Boolean flag which determines whether we are in training mode or in
	inference mode. Note that BN performs differently in these 2 cases.

	Returns: ReLU output of the batch normalized inputs 
	'''
	output = batch_norm(inputs, is_training)
	output = tf.nn.relu(output)
	return output

def dropout(inputs, p_drop, is_training):
	'''
	Inverted dropout implemented using basic TF operations.

	Inputs:
	- inputs: Tensor of inputs that we want to apply dropout to.
	- p_drop: (float) Probability of dropping an input.
	- is_training: Boolean flag which determines whether we are in training mode or in
	inference mode. Note that dropout performs differently in these 2 cases.

	Returns:
	- output: (tensor) Result of applying dropout. Has the same shape and dtype as inputs.
	'''
	if is_training:
		p_keep = 1 - p_drop
		mask = tf.constant(p_keep, dtype=tf.float32)
		mask += tf.random_uniform(tf.shape(inputs), minval=0, maxval=1)
		mask = tf.floor(mask)
		output = tf.div((inputs * mask), p_keep)
	else:
		output = inputs
	return output

def average_pool(inputs, kernel_size, strides, padding='VALID'):
	'''
	Computes the average pool output of inputs.

	Inputs:
	- inputs: Tensor of shape (N,H,W,C)
	- kernel_size: (int) spatial size of pooling window, assuming a square window
	- strides: (int) Stride that is being used for pooling. Same stride along height
	and width of image.
	- padding: Type of padding to use

	Returns:
	- Result of pooling operation, of size (N,H',W',C) where H' and W' depend on the
	input settings.
	'''
	kernel_size = [1, kernel_size, kernel_size, 1]
	stride = [1, strides, strides, 1]
	return tf.nn.avg_pool(inputs, kernel_size, stride, padding)

def global_average_pool(inputs):
	'''
	Takes the global average pool of (N,H,W,C) tensor.

	Inputs:
	- inputs: Tensor of shape (N,H,W,C)

	Returns:
	- Spatial global average pool of the tensor, of shape (N, C). 
	'''
	return tf.reduce_mean(inputs, [1, 2])