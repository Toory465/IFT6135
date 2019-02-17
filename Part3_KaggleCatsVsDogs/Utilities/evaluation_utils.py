# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import pandas as pd
import csv
from training_flags import *

def softmax(scores):
	'''
	Computes softmax for given scores.

	Inputs:
	- scores: Logit outputs of model, of shape (N, C).

	Returns:
	- out: Softmax function applied to scores, of shape (N, C).
	'''
	exp = np.exp(scores-np.max(scores, axis=1, keepdims=True))
	out = exp / np.sum(exp, axis=1, keepdims=True)
	return out

def top1_error(scores, labels):
	'''
	Calculate the top 1 error.
	
	Inputs:
	- scores: Logit outputs of model, of shape (N, C).
	- labels: Ground truth labels, of shape (N, ).
	
	Returns:
	- Top 1 error (between 0 and 1) 
	'''
	batch_size = scores.shape[0]
	correct_predictions = np.equal(np.argmax(scores, axis=1), labels)
	num_correct = np.sum(correct_predictions)
	return (batch_size - num_correct) / float(batch_size)

def accuracy(scores, labels):
	'''
	Calculates accuracy.
	
	Inputs:
	- scores: Logit outputs of model, of shape (N, C).
	- labels: Ground truth labels, of shape (N, ).
	
	Returns:
	- Accuracy (between 0 and 1) 
	'''
	return (1 - top1_error(scores, labels))

def MA(x):
	'''
	Computes the moving average for vector x.
	'''
	moving_avg = x[0]
	m = 0.9
	ema = []
	for i, value in enumerate(x):
		moving_avg = (moving_avg * (m)) + ((1-m) * value)
		if i == 0:
			moving_avg = value
		ema.append(moving_avg)
	return ema

def save_history(history, num_steps, checkpoint_frequency):
	'''
	Save training history as csv file.
	'''
	num_test = len(history['train_loss_hist'])
	step_dict = {'step': (np.arange(num_test) * checkpoint_frequency)}
	step_dict.update(history)
	df = pd.DataFrame(data=step_dict)
	filename = FLAGS.model + '_convergence_' + str(num_steps)
	df.to_csv(filename + '.csv', index=False)
	print(filename + '.csv saved.')

def read_history(filename):
	columns = []
	with open(filename,'r') as f: 
		reader = csv.reader(f)
		for row in reader:
			if columns:
				for i, value in enumerate(row):
					columns[i].append(float(value))
			else:
				columns = [[value] for value in row]
	history = {c[0] : c[1:] for c in columns}
	return history

def save_predictions(predictions, labels, num_steps, custom_filename=None):
	'''
	Save preditions as csv file.
	'''
	label_list = [labels[int(x)] for x in predictions]
	num_test = predictions.shape[0]
	id_list = np.arange(num_test) + 1
	df = pd.DataFrame(data={'id':id_list, 'label':label_list})
	filename = FLAGS.model + '_preds_' + str(num_steps)
	if custom_filename is not None: filename = custom_filename
	df.to_csv(filename + '.csv', index=False)
	print(filename + '.csv saved.')

def save_probabilities(probabilities, labels, num_steps):
	'''
	Save probabilities as csv file for ensembling results.
	'''
	num_test = probabilities.shape[0]
	id_list = np.arange(num_test) + 1
	data_dict = {'id':id_list}
	for i, label in enumerate(labels):
		data_dict[label] = probabilities[:, i]
	df = pd.DataFrame(data=data_dict)
	filename = FLAGS.model + '_probs_' + str(num_steps)
	df.to_csv(filename + '.csv', index=False)
	print(filename + '.csv saved.')