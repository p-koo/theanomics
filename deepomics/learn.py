#!/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import theano
from .metrics import calculate_metrics

__all__ = [
	"train_minibatch",
	"train_minibatch_all",
	"train_variable_learning_rate",
	"train_variable_learning_rate_momentum",
	"train_anneal_batch_size",
	"test_model_all"
]


def train_minibatch(nntrainer, data, batch_size=128, num_epochs=500, 
			patience=10, verbose=2, shuffle=True):
	"""Train a model with cross-validation data and test data"""

	# train model
	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nntrainer.train_step(data['train'], batch_size, verbose, shuffle)
		nntrainer.add_loss(train_loss, 'train') 

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(data['valid'], "valid", batch_size)
		
		# save model
		nntrainer.save_model()

		# check for early stopping				
		if patience:	
			status = nntrainer.early_stopping(valid_loss, patience)
			if not status:
				break
				
	return nntrainer




def train_minibatch_all(nntrainer, data, batch_size=128, num_epochs=500, 
			patience=10, verbose=2, shuffle=True, objective='binary'):

	# variables to store training and test metrics 
	train_metrics = []
	test_metrics = []
	valid_metrics = []

	# calculate metrics for training set
	train_loss, train_prediction, train_label = nntrainer.test_step(data['train'], batch_size=batch_size)
	scores = calculate_metrics(train_label, train_prediction, objective=objective)
	train_metrics.append(np.hstack([train_loss, scores[0]]))

	# calculate metrics for cross-validation set
	valid_loss, valid_prediction, valid_label = nntrainer.test_step(data['valid'], batch_size=batch_size)
	scores = calculate_metrics(valid_label, valid_prediction, objective=objective)
	valid_metrics.append(np.hstack([valid_loss, scores[0]]))

	# calculate metrics for test set
	test_loss, test_prediction, test_label = nntrainer.test_step(data['test'], batch_size=batch_size)
	scores = calculate_metrics(test_label, test_prediction, objective=objective)
	test_metrics.append(np.hstack([test_loss, scores[0]]))

	# train model and keep track of metrics 
	min_loss = 1e6
	for epoch in range(num_epochs):
		sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training epoch and calculate metrics for training set
		train_loss = nntrainer.train_step(data['train'], batch_size=batch_size, verbose=verbose, shuffle=shuffle)
		train_loss2, train_prediction, train_label = nntrainer.test_step(data['train'], batch_size=batch_size)
		scores = calculate_metrics(train_label, train_prediction, objective=objective)
		train_metrics.append(np.hstack([train_loss, scores[0]]))
		print("  train loss:\t\t{:.5f}".format(train_loss))

		# calculate metrics for cross-validation set
		valid_loss, valid_prediction, valid_label = nntrainer.test_step(data['valid'], batch_size=batch_size)
		scores = calculate_metrics(valid_label, valid_prediction, objective=objective)
		valid_metrics.append(np.hstack([valid_loss, scores[0]]))
		print("  valid loss:\t\t{:.5f}".format(valid_loss))

		# calculate metrics for test set
		test_loss, test_prediction, test_label = nntrainer.test_step(data['test'], batch_size=batch_size)
		scores = calculate_metrics(test_label, test_prediction, objective=objective)
		test_metrics.append(np.hstack([test_loss, scores[0]]))
		print("  test loss:\t\t{:.5f}".format(test_loss))

		if valid_loss < min_loss:
			min_loss = valid_loss
			# save best model
			nntrainer.nnmodel.save_model_parameters(nntrainer.file_path+'_best.pickle')        
			nntrainer.test_model(data['valid'], "valid", batch_size);

	nntrainer.nnmodel.save_model_parameters(nntrainer.file_path+'_last.pickle')

	# store metrics
	train_metrics= np.vstack(train_metrics)
	valid_metrics = np.vstack(valid_metrics)
	test_metrics = np.vstack(test_metrics)
	results = [train_metrics, valid_metrics, test_metrics]

	return nntrainer, results

	



def train_variable_learning_rate(nntrainer, train, valid, learning_rate_schedule, 
						batch_size=128, num_epochs=500, patience=10, verbose=1):
	"""Train a model with cross-validation data and test data
			learning_rate_schedule = {  0: 0.001
										2: 0.01,
										5: 0.001,
										15: 0.0001
										}
	"""
	# train model
	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# change learning rate if on schedule
		if epoch in learning_rate_schedule:
			lr = np.float32(learning_rate_schedule[epoch])
			nntrainer.set_learning_rate(lr)

		# training set
		train_loss = nntrainer.train_step(data['train'], batch_size, verbose)
		nntrainer.add_loss(train_loss, 'train') 

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(data['valid'], "valid", batch_size)
		
		# save model
		nntrainer.save_model()

		# check for early stopping				
		if patience:	
			status = nntrainer.early_stopping(valid_loss, patience)
			if not status:
				break

	return nntrainer


def train_variable_learning_rate_momentum(nntrainer, train, valid, learning_rate_schedule, 
						momenum_schedule, batch_size=128, num_epochs=500, patience=10, verbose=1):
	"""Train a model with cross-validation data and test data
			learning_rate_schedule = {  0: 0.001
										2: 0.01,
										5: 0.001,
										15: 0.0001
										}
	"""
	# train model
	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# change learning rate if on schedule
		if epoch in learning_rate_schedule:
			lr = np.float32(learning_rate_schedule[epoch])
			nntrainer.set_learning_rate(lr)

		# change momentum if on schedule
		if epoch in momenum_schedule:
			momentum = np.float32(momenum_schedule[epoch])
			nntrainer.set_momenum(momentum)

		# training set
		train_loss = nntrainer.train_step(data['train'], batch_size, verbose)
		nntrainer.add_loss(train_loss, 'train') 

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(data['valid'], "valid", batch_size)
		
		# save model
		nntrainer.save_model()

		# check for early stopping				
		if patience:	
			status = nntrainer.early_stopping(valid_loss, patience)
			if not status:
				break

	return nntrainer


def train_anneal_batch_size(nntrainer, train, valid, batch_schedule, 
						batch_size=128, num_epochs=500, patience=10, verbose=1):
	"""Train a model with cross-validation data and test data
			batch_schedule = {  0: 0.001
								2: 0.01,
								5: 0.001,
								15: 0.0001
								}
	"""
	# train model
	for epoch in range(num_epochs):
		if verbose >= 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# change learning rate if on schedule
		if epoch in batch_schedule:
			batch_size = batch_schedule[epoch]

		# training set
		train_loss = nntrainer.train_step(data['train'], batch_size, verbose)
		nntrainer.add_loss(train_loss, 'train') 

		# test current model with cross-validation data and store results
		if 'valid' in data.keys():
			valid_loss = nntrainer.test_model(data['valid'], "valid", batch_size)
		
		# save model
		nntrainer.save_model()

		# check for early stopping				
		if patience:	
			status = nntrainer.early_stopping(valid_loss, patience)
			if not status:
				break

	return nntrainer

		
def test_model_all(nntrainer, test, batch_size, num_train_epochs, filepath):
	"""loops through training parameters for epochs min_index 
	to max_index located in filepath and calculates metrics for 
	test data """
	print("Model performance for each training epoch on on test data set")

	for epoch in range(num_train_epochs):
		sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_train_epochs))

		# load model parameters for a given training epoch
		savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
		nntrainer.nnmodel.set_parameters_from_file(savepath)

		# test model with cross-validation data
		nntrainer.test_model(test, batch_size, "test")

	return nntrainer

