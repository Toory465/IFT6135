# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import os
from imageio import imread

def catsvsdogs2numpy(data_dir):
	'''
	Converts catsvsdogs dataset from jpg images in "data_dir/trainset/" and
	"data_dir/testset/" directories to three numpy arrays: train_dataset.npy,
	train_labels.npy, X_test.npy. These numpy arrays are stored in data_dir
	directory.

	Inputs:
	- data_dir: Path to directory of catsvsdogs dataset (should contain
	trainset and testset folders)

	Returns: Nothing. (Saves dataset as three numpy array files into data_dir)
	'''
	TRAIN_PATH = data_dir + "trainset/"
	labels = ['Cat', 'Dog']
	train_dataset = {}
	for label in labels:
		file_names = os.listdir(f'{TRAIN_PATH}{label}')
		train_dataset[label] = np.zeros((len(file_names), 64, 64, 3), dtype = np.uint8)
		for i in range(len(file_names)):
			img = imread(f'{TRAIN_PATH}{label}/{file_names[i]}')
			if img.ndim == 3:
				train_dataset[label][i,:,:,:] = img
			elif img.ndim == 2: # In case it is a grayscale image
				train_dataset[label][i,:,:,:] = img[:,:,None]
		print('Loading ' + label + ' images done...')

	# Concatenate entire dataset into one numpy array
	train_labels = np.hstack([np.zeros(train_dataset['Cat'].shape[0]), np.ones(train_dataset['Dog'].shape[0])])
	train_labels = train_labels.astype('uint8')
	train_dataset = np.concatenate((train_dataset['Cat'], train_dataset['Dog']))

	TEST_PATH = data_dir + "testset/test/"
	file_names = os.listdir(f'{TEST_PATH}')
	X_test = np.zeros((len(file_names), 64, 64, 3), dtype = np.uint8)
	for i in range(len(file_names)):
		img = np.array(imread(f'{TEST_PATH}{(i+1)}.jpg'))
		if img.ndim == 3:
			X_test[i,:,:,:] = img
		elif img.ndim == 2: # In case it is a grayscale image
			X_test[i,:,:,:] = img[:,:,None]

	np.save(f'{data_dir}train_dataset', train_dataset)
	np.save(f'{data_dir}train_labels', train_labels)
	np.save(f'{data_dir}X_test', X_test)
	print('Dataset stored as numpy array in: ' + data_dir)

def get_catsvsdogs_dataset(data_dir, validation_split=0.2, seed=None, normalize_train=True):
	'''
	Reads catsvsdogs dataset from data_dir, then randomly splits the training data based on
	the validation_split ratio and also the optional seed. Then it proceeds to normalize the
	data using the mean and standard deviation of the training set (X_train). If we're using
	data augmentation on X_train, we might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Inputs:
	- data_dir: should contain EITHER trainset and testset folders
	OR train_dataset.npy, train_labels.npy and X_test.npy.
	- validation_split: Determines how to split the train_dataset into train and validation
	sets. validation_split is the ratio of X_val size to the entire train_dataset.
	- seed: (Optional) Performs the random split of training and validation set using this
	seed.
	- normalize_tarin: We might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Returns:
	- X_train: Training images.
	- y_train: Training image labels.
	- X_val: Validation images.
	- y_val: Validation image labels.
	- X_test: Test images.
	- X_train_moments: Tuple of mean and standard deviation of X_train. (Used for
	visualizing original samples after classification)
	'''
	assert validation_split < 1
	if seed is not None: np.random.seed(seed)

	try: # Try to load numpy dataset
		train_dataset = np.load(f'{data_dir}train_dataset.npy').astype('float32')
		train_labels = np.load(f'{data_dir}train_labels.npy').astype('float32')
		X_test = np.load(f'{data_dir}X_test.npy').astype('float32')
	except EnvironmentError: # If numpy dataset doesn't exist, convert jpg data to numpy
		print('Could not find dataset as numpy array in: "' + data_dir + '"')
		try:
			print('Generating numpy dataset from existing jpeg images in "' + data_dir +
			'trainset/" and "' + data_dir + 'testset/"...')
			catsvsdogs2numpy(data_dir)
			train_dataset = np.load(f'{data_dir}train_dataset.npy').astype('float32')
			train_labels = np.load(f'{data_dir}train_labels.npy').astype('float32')
			X_test = np.load(f'{data_dir}X_test.npy').astype('float32')
		except EnvironmentError:
			raise ValueError('Invalid data_dir: ' + data_dir +
				'. data_dir should point to the parent directory of "trainset" and "testset" folders.')
	
	# Split train and validation set
	len_train = len(train_labels)
	len_val = int(validation_split * len_train)
	ind = np.arange(len_train)
	np.random.shuffle(ind)

	X_train = train_dataset[ind[len_val:]]
	y_train = train_labels[ind[len_val:]]
	X_val = train_dataset[ind[:len_val]]
	y_val = train_labels[ind[:len_val]]

	# Normalize the data with training set statistics
	X_train_mean = np.mean(X_train, axis=0)
	X_train_std = np.std(X_train, axis=0)

	# We might want to normalize training set after augmentation
	if normalize_train:
		X_train -= X_train_mean
		X_train /= X_train_std

	X_val -= X_train_mean
	X_val /= X_train_std

	X_test -= X_train_mean
	X_test /= X_train_std

	X_train_moments = (X_train_mean, X_train_std)
	return X_train, y_train, X_val, y_val, X_test, X_train_moments