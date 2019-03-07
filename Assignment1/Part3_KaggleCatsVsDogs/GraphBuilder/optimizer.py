# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import tensorflow as tf

class momentum_optimizer(tf.train.Optimizer):
	'''
	Momentum optimizer built from lower level tensorflow operations.
	'''
	def __init__(self, learning_rate, momentum=0.9, use_locking=False, name="momentum_optimizer"):
		super(momentum_optimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._m = momentum

	def _prepare(self):
		'''
		Convert learning rate and momentum to tensors
		'''
		self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
		self._m_t = tf.convert_to_tensor(self._m, name="momentum_t")

	def _create_slots(self, var_list):
		'''
		Initialize velocity for each variable
		'''
		for var in var_list:
			self._zeros_slot(var, "v", self._name)

	def _apply_dense(self, grad, var):
		'''
		Create update operations for variables and their velocity.
		'''

		# Convert lr and momentum tensor dtypes to have the
		# same dtype as variables.
		lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
		m_t = tf.cast(self._m_t, var.dtype.base_dtype)
		
		# Get velocity of the variable
		v = self.get_slot(var, "v")

		# Velocity update operation.
		v_t = v.assign((m_t * v) - (lr_t * grad))

		# Variable update operation.
		var_update = tf.assign_sub(var, -v_t)

		# Group variable update and velocity update
		return tf.group(*[var_update, v_t])

class adam_optimizer(tf.train.Optimizer):
	'''
	Adam optimizer built from lower level tensorflow operations.
	https://arxiv.org/pdf/1412.6980.pdf
	'''
	def __init__(self, learning_rate, global_step, beta1=0.9, beta2=0.999,
	epsilon=1e-8, use_locking=False, name="adam_optimizer"):
		super(adam_optimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._b1 = beta1
		self._b2 = beta2
		self._step_t = global_step
		self._eps = epsilon

	def _prepare(self):
		'''
		Convert python scalars to TF tensors
		'''
		self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
		self._b1_t = tf.convert_to_tensor(self._b1, name="beta1_t")
		self._b2_t = tf.convert_to_tensor(self._b2, name="beta2_t")
		self._eps_t = tf.convert_to_tensor(self._eps, name="epsilon_t")

	def _create_slots(self, var_list):
		'''
		Initialize first moment and second moment for each variable
		'''
		for var in var_list:
			self._zeros_slot(var, "m", self._name)
			self._zeros_slot(var, "v", self._name)

	def _apply_dense(self, grad, var):
		'''
		Create update operations for variables and Adam's variables.
		'''
		# Convert lr and momentum tensor dtypes to have the
		# same dtype as variables.
		lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
		b1_t = tf.cast(self._b1_t, var.dtype.base_dtype)
		b2_t = tf.cast(self._b2_t, var.dtype.base_dtype)
		step_t = tf.cast(self._step_t, var.dtype.base_dtype)
		eps_t = tf.cast(self._eps_t, var.dtype.base_dtype)
		
		# Get velocity of the variable
		m = self.get_slot(var, "m")
		v = self.get_slot(var, "v")

		# Update operations.
		m_t = m.assign((m * b1_t) + (grad * (1 - b1_t)))
		v_t = v.assign((v * b2_t) + (grad * grad * (1 - b2_t)))

		# Variable update operation.
		m_unbiased = m / (1 - (b1_t ** (step_t+1)))
		v_unbiased = v / (1 - (b2_t ** (step_t+1)))
		var_update = tf.assign_sub(var, -(lr_t * (m_unbiased / (tf.sqrt(v_unbiased) + eps_t))))

		# Group variable update and velocity update
		return tf.group(*[var_update, m_t, v_t])