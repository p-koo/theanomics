#!/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import theano


__all__ = [
	"train_minibatch",
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
		if verbose == 1:
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
		if verbose == 1:
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
		if verbose == 1:
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
		if verbose == 1:
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

