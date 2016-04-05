#!/bin/python

import os
import sys
import numpy as np
import h5py
import scipy.io
from six.moves import cPickle
	

def load_DeepSea(path, num_include=4400000, class_range=918):
	"""Loads DeepSea dataset"""
	
	def data_subset(y, class_range, negative=True):
		" gets a subset of data in the class_range"
		data_index = []
		for i in class_range:
			index = np.where(y[:, i] == 1)
			data_index = np.concatenate((data_index, index[0]), axis=0)
		unique_index = np.unique(data_index)
		num_data = y.shape[0]
		non_index = np.array(list(set(xrange(num_data)) - set(unique_index)))
		if negative:
		    index = np.concatenate((unique_index, non_index), axis=0)
		else:
		    index = unique_index
		return index.astype(int)

	print "loading training data"
	trainmat = h5py.File(os.path.join(path,'train.mat'))
	y_train = np.array(trainmat['traindata']).T
	y_train = y_train[:,class_range]
	index = data_subset(y_train, class_range)
	index = index[0:num_include]
	y_train = y_train[index,:]
	X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2,1,0))
	X_train = X_train[index,:,:]
	train = (X_train, y_train)

	print "loading validation data"  
	validmat = scipy.io.loadmat(os.path.join(path,'valid.mat'))
	y_valid = validmat['validdata']
	y_valid = y_valid[:, class_range]
	index = data_subset(y_valid,class_range, negative=False)
	y_valid = y_valid[index,:]
	X_valid = np.transpose(validmat['validxdata'],axes=(0,1,2)) 
	X_valid = X_valid[index,:,:]
	valid = (X_valid, y_valid)

	print "loading test data"
	testmat = scipy.io.loadmat(os.path.join(path,'test.mat'))
	y_test = testmat['testdata']
	y_test = y_test[:, class_range]
	index = data_subset(y_test,class_range, negative=False)
	y_test = y_test[index,:]
	X_test = np.transpose(testmat['testxdata'],axes=(0,1,2))
	X_test = X_test[index,:,:]
	test = (X_test, y_test)

	return train, valid, test 

def load_MotifSimulation(filename, directory, categorical=0):
	# setup paths for file handling
	filepath = os.path.join(directory,filename)

	# load training set
	print "loading data from: " + filepath
	f = open(filepath, 'rb')
	print "loading train data"
	train = cPickle.load(f)
	print "loading cross-validation data"
	cross_validation = cPickle.load(f)
	print "loading test data"
	test = cPickle.load(f)
	f.close()

	X_train = train[0].transpose((0,1,2))
	y_train = train[1]
	X_val = cross_validation[0].transpose((0,1,2))
	y_val = cross_validation[1]
	X_test = test[0].transpose((0,1,2))
	y_test = test[1]

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	if categorical == 1:
		y_train = np.argmax(y_train,axis=1)
		y_val = np.argmax(y_val,axis=1)
		y_test = np.argmax(y_test,axis=1)

	train = (X_train, y_train)
	valid = (X_val, y_val)
	test = (X_test, y_test)

	return train, valid, test

"""

class Data():
	def __init__(X_train, y_train, X_valid, y_valid, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_valid = X_valid
		self.y_valid = y_valid
		self.X_test = X_test
		self.y_test = y_test
		self.shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])

		y_train.shape[1]
		self.num_labels = max(y_train)+1


	def get_train():
		return self.X_train, self.y_train

	def get_test():
		return self.X_test, self.y_test

	def get_valid():
		return self.X_valid, self.y_valid

	def get_shape():
		return self.shape

	def get_num_labels():
		return self.num_labels
		
	def get_minibatch():


	def batch_generator(X, y, N):
	    while True:
	        idx = np.random.choice(len(y), N)
	        yield X[idx].astype('float32'), y[idx].astype('int32')


"""


