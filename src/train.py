#!/bin/python
import sys
from neuralnetwork import MonitorPerformance
from utils import batch_generator
from six.moves import cPickle
import numpy as np
import theano

def train_minibatch_ae(nnmodel, train, batch_size=128, num_epochs=500, 
			patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nnmodel.train_step_ae(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)
		nnmodel.valid_monitor.add_loss(train_loss)
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(train_loss, epoch, patience)
		if not status:
			break

	return nnmodel

def train_minibatch(nnmodel, train, valid, batch_size=128, num_epochs=500, 
			patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		train_loss = nnmodel.train_step(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test current model with cross-validation data and store results
		valid_loss = nnmodel.test_model(valid, batch_size, "valid")
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel


def train_learning_decay(nnmodel, train, valid, learning_rate=.01, 
						batch_size=128, num_epochs=500, patience=14,  
						learn_patience=4, decay=0.5, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	# set learning rate and wait time before altering learning rate
	learn_wait = learn_patience
	nnmodel.set_learning_rate(learning_rate) 

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
				sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# train step
		train_loss = nnmodel.train_step(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test current model with cross-validation data and store results
		valid_loss = nnmodel.test_model(valid, batch_size, "valid")

		# check to see if learning rate should change
		min_loss, min_epoch = nnmodel.get_min_loss()	
		if min_loss < valid_loss:
			learn_wait -= 1
			if learn_wait == 0:
				learning_rate *= decay
				print('changing learning rate to: %f' % (learning_rate))
				nnmodel.set_learning_rate(learning_rate) 
				learn_wait = learn_patience
		else:
			learn_wait = learn_patience


		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel


def train_anneal_learning(nnmodel, train, valid, learning_schedule, 
						batch_size=128, num_epochs=500, 
						patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data
		learning_schedule = np.linspace(.1,.00001, 30)
	"""
	num_anneal = len(learning_schedule)
	learning_schedule = np.array(learning_schedule, dtype=theano.config.floatX)

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

		# training set
		if epoch < num_anneal:
			nnmodel.set_learning_rate(learning_schedule[epoch])
		else:
			nnmodel.set_learning_rate(learning_schedule[-1])

		# train step
		train_loss = nnmodel.train_step(train, batch_size, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test model with cross-validation data
		nnmodel.test_model(valid, batch_size, "valid")
		
		# save model
		if filepath:
			savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
			nnmodel.save_model_parameters(savepath)

		# check for early stopping					
		status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
		if not status:
			break

	return nnmodel
	


def train_autoencoder_minibatch(nnmodel, X, batch_size=128, num_epochs=500, 
									patience=10, verbose=1, filepath='.'):
	"""Train a model with cross-validation data and test data"""

	# setup generator for mini-batches
	num_train_batches = len(X) // batch_size
	def batch_generator(X, N):
		while True:
			idx = np.random.choice(len(X), N)
			yield X[idx].astype('float32')

	train_batches = batch_generator(X, batch_size)

	# train model
	for epoch in range(num_epochs):
		if verbose == 1:
			sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))


		# training set
		train_loss = nnmodel.train_step(train_batches, num_train_batches, verbose)
		nnmodel.train_monitor.add_loss(train_loss)

		# test model with cross-validation data
		nnmodel.test_model(valid, batch_size, "valid")
	
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

		# test model with cross-validation data
		nnmodel.test_model(test, batch_size, "test")

	return performance
