# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import datetime
from itertools import combinations
from Utilities.evaluation_utils import *

def load_model_logits(PATH, file_names):
	'''
	Loads model logits stored as CSV files.
	'''
	model_logits = {}
	for file_name in file_names:
		df = pd.read_csv(f'{PATH}{file_name}', usecols=[1,2])
		model_logits[file_name] = df.values
	return model_logits

def logits2predictions(model_logits):
	'''
	Converts {model_name: logits} dictionary to {model_name: predictions}.
	'''
	model_predictions = model_logits.copy()
	for k, v in model_predictions.items():
		preds = np.zeros_like(v)
		preds[np.arange(len(v)), v.argmax(1)] = 1
		model_predictions[k] = preds
	return model_predictions

def ensemble_search(model_predictions, labels, num_ensemble_models, num_weight_search=50):
	'''
	Searches among all possible #num_ensemble_models combinations from all
	the models in model_predictions. For each model combination, tries
	#num_weight_search different random weights for making a weighted
	ensemble of the models. Returns dictionary of the best model found.
	'''
	print('Searching all possible %d model combinations.' % num_ensemble_models)
	print('Trying %d random weights for each combination.' % num_weight_search)
	print('-----------')
	models = []
	model_preds = []
	best_acc = 0
	best_ensemble = {'acc':0}
	for key, value in model_predictions.items():
		models.append(key)
		model_preds.append(value)
	num_all_models = len(models)
	combinations_ = list(combinations(np.arange(num_all_models), num_ensemble_models))
	num_combinations = len(combinations_)
	print_list = np.arange(0, num_combinations, num_combinations // 5)
	models = np.asarray(models)
	model_preds = np.asarray(model_preds)
	for i, combination in enumerate(combinations_):
		combination_model_preds = model_preds[list(combination)]
		for _ in range(num_weight_search):
			weights = np.random.uniform(0, 1, (num_ensemble_models, 1, 1))
			ensemble_result = np.sum(weights * combination_model_preds, axis=0)
			ensemble_acc = accuracy(ensemble_result, labels)
			if ensemble_acc > best_ensemble['acc']:
				best_ensemble['weights'] = weights
				best_ensemble['combination'] = models[list(combination)]
				best_ensemble['acc'] = ensemble_acc
		if i in print_list:
			time_str = datetime.datetime.now().time().strftime("%I:%M:%S %p")
			print('(%s) %d/%d combinations searched...' % (time_str, i, num_combinations))
	best_ensemble['weights'] /= np.sum(best_ensemble['weights'])
	best_ensemble['weights'] = np.array(best_ensemble['weights']).squeeze()
	print('Search done! Best accuracy achieved: %.3f%%' % (100*best_ensemble['acc']))
	return best_ensemble

def ensemble_models(model_predictions, combination, weights):
	'''
	Ensembles several models with given combination and weights and returns
	the logits of the ensembled models.
	'''
	combination_model_preds = []
	for model in combination:
		combination_model_preds.append(model_predictions[model])
	if weights.ndim == 1:
		weights = weights[:, None, None]
	combination_model_preds = np.asarray(combination_model_preds)
	ensemble_output = np.sum(weights * combination_model_preds, axis=0)
	return ensemble_output