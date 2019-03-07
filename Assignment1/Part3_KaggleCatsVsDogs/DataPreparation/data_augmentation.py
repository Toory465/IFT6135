# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from training_flags import *

def horizontal_flip(image, p=0.5):
	'''
	Randomly flips image horizontally with probability p.

	Inputs:
	- image: Single image of shape (H, W, C).
	- p: Probability of augmentation being applied.

	Returns:
	- image: Either flipped or original image, of shape (H, W, C).
	'''
	if np.random.uniform(0, 1) < p:
		image = cv2.flip(image, 1)
	return image


def random_crop_and_flip(X, padding_size):
	'''
	Takes a random crop from the image and randomly flips the image
	with a probability of 0.5.

	Inputs:
	- X: A minibatch of images, of shape (N, H+2p, W+2p, C).
	- padding_size: How much padding was applied to the images (p).

	Returns:
	- X: Augmented minibatch of images, of shape (N, H, W, C).
	'''
	N = X.shape[0]
	cropped_X = np.zeros((N, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
	for i in range(N):
		x_offset = np.random.randint(low=0, high=2 * padding_size)
		y_offset = np.random.randint(low=0, high=2 * padding_size)
		cropped_X[i] = X[i, x_offset:x_offset+IMG_HEIGHT, y_offset:y_offset+IMG_WIDTH]
		cropped_X[i] = horizontal_flip(image=cropped_X[i])
	return cropped_X


def random_rescale(X, p=0.3):
	'''
	With a probability of p, randomly enlarges the image to either 1.5x to 2x 
	the original size and takes a random crop from the result.

	Inputs:
	- X: A minibatch of images, of shape (N, H, W, C).
	- p: Probability of augmentation being applied for each image.

	Returns:
	- X: Augmented minibatch of images, of shape (N, H, W, C).
	'''
	for i in range(len(X)):
		if np.random.uniform(0,1) < p:
			scale = np.random.choice([1.5, 2.0])
			zoomed_in = cv2.resize(X[i], None, fx=scale, fy=scale)
			x_offset = np.random.randint(low=0, high=(scale-1)*IMG_WIDTH)
			y_offset = np.random.randint(low=0, high=(scale-1)*IMG_HEIGHT)
			X[i] = zoomed_in[x_offset:x_offset+IMG_HEIGHT, y_offset:y_offset+IMG_WIDTH]
	return X

def random_rotate(X, p=0.3):
	'''
	With a probability of p, rotates the image with a random angle between
	-25 and 25 degrees.

	Inputs:
	- X: A minibatch of images, of shape (N, H, W, C).
	- p: Probability of augmentation being applied for each image.

	Returns:
	- X: Augmented minibatch of images, of shape (N, H, W, C).
	'''
	for i in range(len(X)):
		if np.random.uniform(0,1) < p:
			img = X[i].astype('uint8')
			angle = np.random.uniform(-25, 25)
			img = Image.fromarray(img).rotate(angle)
			X[i] = np.asarray(img, dtype=np.float32)
	return X

def random_enhance(X, p=0.3):
	'''
	With a probability of p, randomly changes one of the following properties
	of the image: Color, Brightness, Contrast, Sharpness.

	Inputs:
	- X: A minibatch of images, of shape (N, H, W, C).
	- p: Probability of augmentation being applied for each image.

	Returns:
	- X: Augmented minibatch of images, of shape (N, H, W, C).
	'''
	for i in range(len(X)):
		if np.random.uniform(0,1) < p:
			img = X[i].astype('uint8')
			img = Image.fromarray(img)
			token = np.random.choice(['color', 'brightness', 'contrast', 'sharpness'])

			if token == 'color':
				color = ImageEnhance.Color(img)
				img = color.enhance(np.random.uniform(0.1, 1))

			elif token == 'brightness':
				brightness = ImageEnhance.Brightness(img)
				img = brightness.enhance(np.random.uniform(1, 1.25))

			elif token == 'contrast':
				contrast = ImageEnhance.Contrast(img)
				img = contrast.enhance(np.random.uniform(0.5, 1))
				
			elif token == 'sharpness':
				sharpness = ImageEnhance.Sharpness(img)
				img = sharpness.enhance(np.random.uniform(0.2, 1.2))

			X[i] = np.asarray(img, dtype=np.float32)
	return X