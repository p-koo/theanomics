#!/bin/python
import sys
from neuralnetwork import MonitorPerformance
from utils import batch_generator
from six.moves import cPickle
import numpy as np
import theano

def train_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
			patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	# setup generator for mini-batches
	num_train_batches = len(train[0]) // batch_size
	train_batches = batch_generator(train[0], train[1], batch_size)

	num_valid_batches = len(valid[0]) // batch_size
	valid_batches = batch_generator(valid[0], valid[1], batch_size)

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nnmodel.train_step(train_batches, num_train_batches, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test current model with cross-validation data and store results
		valid_loss, valid_prediction = nnmodel.test_step_batch(valid)
		nnmodel.valid_monitor.update(valid_loss, valid_prediction, valid[1])
		nnmodel.valid_monitor.print_results("valid")
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel



def train_valid_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
			patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	"""
	# setup generator for mini-batches
	num_train_batches = len(train[0]) // batch_size
	train_batches = 

	num_valid_batches = len(valid[0]) // batch_size
	valid_batches = batch_generator(valid[0], valid[1], batch_size)
	"""
	learning_rate = np.linspace(.1,.001,30)
	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nnmodel.train_step(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test current model with cross-validation data and store results
		valid_loss, valid_prediction, valid_label = nnmodel.test_step_minibatch(valid, batch_size)
		nnmodel.valid_monitor.update(valid_loss, valid_prediction, valid_label)
		nnmodel.valid_monitor.print_results("valid")
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel
	

def test_model_all(nnmodel, test, batch_size, num_train_epochs, filepath):
	"""loops through training parameters for epochs min_index 
	to max_index located in filepath and calculates metrics for 
	test data """
	print "Model performance for each training epoch on on test data set"

	performance = MonitorPerformance("test_all")
	for epoch in range(num_train_epochs):
		sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_train_epochs))

		# load model parameters for a given training epoch
		savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
		nnmodel.set_parameters_from_file(savepath)

		# get test metrics 
		test_loss, test_prediction, test_label = nnmodel.test_step_minibatch(test, batch_size)
		performance.update(test_loss, test_prediction, test_label)
		performance.print_results(" test") 

	return performance




def anneal_train_valid_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
			patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	num_anneal = 30
	learning_rate = np.array(np.linspace(.1,.001,num_anneal), dtype=theano.config.floatX)

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		if epoch < num_anneal:
			nnmodel.set_learning_rate(learning_rate[epoch])
		else:
			nnmodel.set_learning_rate(learning_rate[-1])

		train_loss = nnmodel.train_step(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test current model with cross-validation data and store results
		valid_loss, valid_prediction, valid_label = nnmodel.test_step_minibatch(valid, batch_size)
		nnmodel.valid_monitor.update(valid_loss, valid_prediction, valid_label)
		nnmodel.valid_monitor.print_results("valid")
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel
	