# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import datetime
import os
from training_flags import *
from DataPreparation.data_augmentation import *
from DataPreparation.dataset_preparation import *
from GraphBuilder.vgg19_graph import *
from GraphBuilder.wideresnet_graph import *
from GraphBuilder.optimizer import momentum_optimizer, adam_optimizer

class Model(object):
	'''
	This Object holds our model which we can then either train or infer from.
	'''
	def __init__(self, hparams):
		'''
		Initiates the training graph and the validation graph.

		Inputs:
		- hparams: HParams object containing model hyperparameters. Expected format:
		
		hparams = tf.contrib.training.HParams(
		data_dir='Dataset/',
	    validation_split=0.2,
	    split_seed = 6135,
	    num_steps = 50000,
	    lr = 1e-2,
	    train_batch_size=50,
	    eval_batch_size=50,
	    dropout_probability=0.5,
	    resume_training = False,
	    optimizer = 'Momentum', # 'Momentum' or Adam'
	    cosine_lr = False,
	    l2_scale = 1e-4,
	    lr_decay_factor = 0.1,
	    decay_steps = [17500, 40100]
    	)

		Returns: Nothing. (Initializes the model)
		'''
		self.hparams = hparams
		self.initial_lr = hparams.lr
		if FLAGS.model == 'VGG19':
			self.build_model_graph = build_vgg19_graph
		elif FLAGS.model == 'Wide28_10':
			self.build_model_graph = build_wide2810_graph
		else:
			raise ValueError('Unsupported model: %s' % FLAGS.model)
		self.build_train_graph()
		self.build_eval_graph(mode='val')

		global_step = tf.Variable(0, trainable=False)
		self.train_op = self.train_op(global_step, self.train_loss, self.train_top1_error)

	def train(self):
		'''
		Trains the model based on the hyperparameters given by FLAGS and hparams. Can restore model
		from checkpoint to continue training if self.hparams.resume_training is True and FLAGS.load_dir
		(checkpoint directory) is given. If model is restored from checkpoint, it will continue
		its training from checkpoint step (ckpt_step) up to self.hparams.num_steps.
		EMA (Exponential Moving Average) on validation loss and validation top 1 error is computed
		once every FLAGS.print_every number of training steps.
		Learning rate is decayed at steps in self.decay_steps.
		Automatic checkpoints are saved every FLAGS.save_every number of training steps, and also
		on the last training step. A CSV of the training process is also saved to disk every 
		FLAGS.save_every number of training steps.

		Inputs: Nothing.
		Returns: Nothing. (Trains model for self.hparams.num_steps and updates model parameters)
		'''
		step_hist = []
		train_error_hist = []
		train_loss_hist = []
		val_loss_hist = []
		val_error_hist = []
		lr_hist = []
		Xd_train, yd_train, Xd_val, yd_val, _, _ = get_catsvsdogs_dataset(self.hparams.data_dir,
			self.hparams.validation_split, self.hparams.split_seed, not FLAGS.use_augmentation)

		# Initialize saver
		saver = tf.train.Saver()
		
		# Initialize session and (if required) load checkpoint
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		ckpt_step = self.restore_checkpoint(sess, saver) if self.hparams.resume_training else 0
		if ckpt_step == 0: sess.run(tf.global_variables_initializer())

		gpu_available = tf.test.gpu_device_name() is not ''
		dev = '/gpu:0' if gpu_available else '/cpu:0'
		self.initial_message(dev)
		with tf.device(dev):
			for step in range(ckpt_step-1, self.hparams.num_steps):
				# Make an update step, compute training loss and error
				X_train_mb, y_train_mb = self.minibatch(Xd_train, yd_train, self.hparams.train_batch_size)
				if FLAGS.use_augmentation: X_train_mb = self.augment(X_train_mb)
				feed_dict = {self.X_train: X_train_mb, self.y_train: y_train_mb, self.lr: self.hparams.lr}
				fetches = [self.train_op, self.train_top1_error, self.train_loss]
				_, train_error_, train_loss_ = sess.run(fetches, feed_dict)

				if step % FLAGS.print_every == 0:
					# Compute validation loss and error
					X_val_mb, y_val_mb = self.minibatch(Xd_val, yd_val, self.hparams.eval_batch_size)
					feed_dict_val = feed_dict.copy()
					feed_dict_val.update({self.X_val: X_val_mb, self.y_val: y_val_mb})
					fetches_val = [self.val_top1_error, self.val_loss]
					val_error_, val_loss_ = sess.run(fetches_val, feed_dict_val)

					# Update history
					val_error_hist.append(val_error_)
					val_loss_hist.append(val_loss_)
					self.report(step, train_error_, val_error_, train_loss_, val_loss_)
					step_hist.append(step)
					train_error_hist.append(train_error_)
					train_loss_hist.append(train_loss_)
					lr_hist.append(self.hparams.lr)

				self.update_lr(step)

				if step % FLAGS.save_every == 0 or (step == self.hparams.num_steps - 1):
					# Save checkpoint
					checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=step+1)
					df = pd.DataFrame(data={'step':step_hist, 'train_error':train_error_hist,
									'validation_error': val_error_hist})
					df.to_csv(FLAGS.save_dir + FLAGS.model + '_error.csv')

		sess.close()
		history = {'train_error_hist':train_error_hist,
		'train_loss_hist': train_loss_hist,
		'val_error_hist':val_error_hist,
		'val_loss_hist':val_loss_hist,
		'lr_hist': lr_hist}
		return history

	def test(self, Xd_test):
		'''
		Restores model from checkpoint and classifies given test dataset.

		Inputs:
		- Xd_test: Given test datest of shape (minibatch_size,H,W,C)

		Returns:
		- scores: Matrix of shape (minibatch_size, #classes) giving model
		predicted scores (logits) for each class.
		'''
		test_size = self.hparams.eval_batch_size
		num_test = Xd_test.shape[0]
		scores = np.zeros((num_test, FLAGS.num_classes))
		num_batches = int(np.ceil(num_test / test_size))
		self.build_eval_graph(mode='test')

		# Initialize new session and restore checkpoint
		saver = tf.train.Saver()
		sess = tf.Session()
		self.restore_checkpoint(sess, saver)

		# Classify test minibatches
		for i in range(num_batches):
			ind = i * test_size
			X_test_mb = Xd_test[ind:ind+test_size]
			scores[ind:ind+test_size] = sess.run(self.test_scores, {self.X_test: X_test_mb})
		sess.close()
		return scores

	def build_train_graph(self):
		'''
		Builds the training graph (placeholders, architecture, loss, etc.)

		Inputs:
		- minibatch_size: (Int) Size of our training minibatches. Used for
		calculating the top 1 error.

		Returns: Nothing. (Builds training graph)
		'''
		minibatch_size = self.hparams.train_batch_size
		self.X_train = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
		self.y_train = tf.placeholder(dtype=tf.int32, shape=[None])
		self.lr = tf.placeholder(dtype=tf.float32)
		with tf.variable_scope('model', use_resource=False):
			train_scores = self.build_model_graph(self.X_train, FLAGS.num_classes,
				is_training=True, hparams=self.hparams)
		self.train_top1_error = self.top_k_error(train_scores, self.y_train, minibatch_size)
		xent_loss = self.loss(train_scores, self.y_train)

		l2_loss = self.compute_weight_decay()
		self.train_loss = xent_loss + l2_loss

	def build_eval_graph(self, mode='test'):
		'''
		Builds evaluation graph, which can either be for validation set or test set. The
		difference is that for validation we can compute loss and top k error, whereas
		for test set we do not have the ground truth labels, so only the scores (logits)
		can be computed.
		
		Inputs:
		- minibatch_size: (Int) Size of our training minibatches. Used for calculating the
		top k error, therefore only used if mode='val', otherwise minibatch_size does not
		get used anywhere.
		- mode: (String) 'test' or 'val'. If using 'val', loss and top k error will be
		computed (requires ground truth labels). If using 'test', only logits are computed.

		Returns: Nothing. (Builds evaluation graph)
		'''
		minibatch_size = self.hparams.eval_batch_size
		if mode == 'val':
			self.X_val = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
			self.y_val = tf.placeholder(dtype=tf.int32, shape=[None])
			with tf.variable_scope('model', reuse=True, use_resource=False):
				val_scores = self.build_model_graph(self.X_val, FLAGS.num_classes,
					is_training=False, hparams=self.hparams)
			self.val_top1_error = self.top_k_error(val_scores, self.y_val, minibatch_size)
			self.val_loss = self.loss(val_scores, self.y_val)
		elif mode == 'test':
			self.X_test = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
			with tf.variable_scope('model', reuse=True, use_resource=False):
				self.test_scores = self.build_model_graph(self.X_test, FLAGS.num_classes,
					is_training=False, hparams=self.hparams)
		else:
			raise ValueError("mode must be 'val' or 'test', but %s was entered" % str(mode))

	def loss(self, scores, y):
		'''
		Computes loss using tensorflow operations.

		Inputs:
		- scores: Model predicted scores (logits) of shape (minibatch_size, #classes)
		- y: Ground truth labels of shape (minibatch_size)

		Returns:
		- mean_loss: (Tensor) Softmax cross entropy loss
		'''
		y = tf.cast(y, tf.int64)
		xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,
			labels=y, name='xent_loss')
		mean_loss = tf.reduce_mean(xent_loss, name='mean_loss')
		return mean_loss

	def compute_weight_decay(self):
		'''
		Computes l2 regularization loss.

		Inputs: Nothing

		Returns:
		- l2_loss: Computed l2 regularization loss on weights.
		'''
		vars = tf.trainable_variables()
		condition = lambda v: ('bias' not in v.name) and ('beta' not in v.name) and ('gamma' not in v.name)
		weights = [v for v in vars if condition(v)]
		l2_norm = lambda W: tf.div(tf.reduce_sum(tf.square(W)), 2)
		l2_losses = []
		for w in weights:
			l2_losses.append(l2_norm(w))
		l2_loss = tf.multiply(self.hparams.l2_scale, tf.add_n(l2_losses))
		return l2_loss

	def top_k_error(self, scores, y, minibatch_size, k=1):
		'''
		Computes the top k error.
		
		Inputs:
		- scores: Model predicted scores (logits) of shape (minibatch_size, #classes)
		- y: Ground truth labels of shape (minibatch_size)
		- minibatch_size: (Int) Size of our training minibatches.
		- k: (Optional) (Int) Gives us the value for k in top k error.

		Returns:
		- top k error
		'''
		top1 = tf.to_float(tf.nn.in_top_k(scores, y, k=1))
		num_correct = tf.reduce_sum(top1)
		return (minibatch_size - num_correct) / float(minibatch_size)

	def minibatch(self, X, y, minibatch_size):
		'''
		Takes an arbitrary minibatch of the data.

		Inputs:
		- X: Entire dataset images.
		- y: Entire dataset image labels.
		- minibatch_size: (Int) Size of the minibatch that we want as output

		Returns:
		- X_mb: A minibatch of the dataset images.
		- y_mb: A minibatch of the dataset image labels.
		'''
		indicies = np.arange(X.shape[0])
		np.random.shuffle(indicies)
		ind = indicies[:minibatch_size]
		X_mb = X[ind]
		y_mb = y[ind]

		return X_mb, y_mb


	def augment(self, X):
		'''
		Augments a minibatch of images.

		Inputs:
		- X: A minibatch of images.

		Returns:
		- X: Augmented minibatch of images.
		'''
		pad_width = ((0, 0), (FLAGS.padding, FLAGS.padding), (FLAGS.padding, FLAGS.padding), (0, 0))
		X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
		X = random_crop_and_flip(X, padding_size=FLAGS.padding)
		X = random_rescale(X)
		# X = random_rotate(X)
		# X = random_enhance(X)
		X -= np.mean(X, axis=0)
		X /= np.std(X, axis=0)

		return X

	def initial_message(self, dev):
		'''
		Initial message that is printed when training the model.

		Inputs:
		- dev: (String) Device that will be used for training

		Returns: Nothing. (Prints initial message)
		'''
		print('~~~ Training with %s ~~~'% dev)
		num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
		print('Model: %s' % FLAGS.model)
		print('Number of parameters: %.2fM' % (num_params/1e6))
		print('Training minibatch size: %d' % self.hparams.train_batch_size)
		print('Validation minibatch size: %d' % self.hparams.eval_batch_size)
		print('Using data augmentation: %s' % str(FLAGS.use_augmentation))
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')


	def report(self, step, train_error_, val_error_, train_loss_, val_loss_):
		'''
		Report that it printed every number of steps during training.

		Inputs:
		- step: The training step that we are currently on.
		- train_error_: (Float) Top 1 training error that was achieved on
		last minibatch.
		- val_error_: (Float) Exponential Moving Average of the top 1 
		validation error during training.
		- train_loss_: (Float) Training loss that was achieved on the last
		minibatch.
		- val_loss_: (Float) Exponential Moving Average of validation loss
		during training.

		Returns: Nothing. (Prints report)
		'''
		time_str = datetime.datetime.now().time().strftime("%I:%M:%S %p")
		train_acc = 100 * (1 - train_error_)
		val_acc = 100 * (1 - val_error_)
		print('(%s) Iteration %d:' % (time_str, step))
		print('Training data: loss= %.4f, accuracy %.2f%%' % (train_loss_, train_acc))
		print('Validation data: loss= %.4f, accuracy %.2f%%' % (val_loss_, val_acc))
		print('(lr=%.6f)\n' % self.hparams.lr)

	def update_lr(self, step):
		'''
		Updates learning rate according to stepwise or cosine learning rate annealing.

		Inputs:
		- step: (int) Current input step that we are on.

		Returns: Nothing. (Updates self.hparams.lr according to learning rate decay rule)
		'''
		if self.hparams.cosine_lr == True:
			self.hparams.lr = 0.5 * self.initial_lr * (1 + np.cos(np.pi * step / self.hparams.num_steps))
		else:
			if step in self.hparams.decay_steps:
				# Decay learning rate
				self.hparams.lr = self.hparams.lr_decay_factor * self.hparams.lr
				print('* Learning rate decayed to %.6f \n' % self.hparams.lr)


	def restore_checkpoint(self, sess, saver):
		'''
		Restores model from checkpoint and returns the number of training steps the
		checkpointed model was trained for.

		Inputs:
		- sess: Current session that we want to restore model on
		- saver: Saver that is being used to restore parameters.

		Returns:
		- ckpt_step: (Int) The number of training steps the checkpointed model
		was trained for.
		'''
		ckpt = tf.train.get_checkpoint_state(FLAGS.load_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			ckpt_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print('* Restored from checkpoint (%s): %d iterations' % (FLAGS.load_dir, ckpt_step))
		else:
			raise Exception('No valid checkpoint provided.')
		return ckpt_step

	def train_op(self, step, loss, top1_error):
		'''
		Creates tensor operations for the training procedure
		'''
		if self.hparams.optimizer == 'Momentum':
			optimizer = momentum_optimizer(learning_rate=self.lr, momentum=0.9)
		elif self.hparams.optimizer == 'Adam':
			optimizer = adam_optimizer(learning_rate=self.lr, global_step=step)
		else:
			raise ValueError('Unsupported optimizer: %s' % self.hparams.optimizer)
		train_op = optimizer.minimize(loss, step)
		return train_op