#!/bin/python
import os
import sys
import numpy as np
import h5py
import scipy.io
from six.moves import cPickle
	
def DeepSea(filepath, class_range=range(918), num_include=[]):
	"""Loads DeepSea dataset"""
	
	def data_subset(y, class_range, negative=True):
		" gets a subset of data in the class_range"
		data_index = []
		for i in class_range:
			index = np.where(y[:, i] == 1)
			data_index = np.concatenate((data_index, index[0]), axis=0)
		unique_index = np.unique(data_index)
		num_data = y.shape[0]
		non_index = np.array(list(set(range(num_data)) - set(unique_index)))
		if negative:
		    index = np.concatenate((unique_index, non_index), axis=0)
		else:
		    index = unique_index
		return index.astype(int)

	print "loading training data"
	trainmat = h5py.File(os.path.join(filepath,'train.mat'))
	y_train = np.array(trainmat['traindata']).T
	index = data_subset(y_train, class_range, negative=False)
	if num_include:
		index = index[0:num_include]
	y_train = y_train[:,class_range]
	y_train = y_train[index,:]
	X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2,1,0))
	X_train = X_train[index,:,:]
	X_train = np.expand_dims(X_train, axis=3)
	train = (X_train.astype(np.float32), y_train.astype(np.int32))

	print "loading validation data"  
	validmat = scipy.io.loadmat(os.path.join(filepath,'valid.mat'))
	y_valid = validmat['validdata']
	index = data_subset(y_valid,class_range, negative=False)
	y_valid = y_valid[:, class_range]
	y_valid = y_valid[index,:]
	X_valid = np.transpose(validmat['validxdata'],axes=(0,1,2)) 
	X_valid = X_valid[index,:,:]
	X_valid = np.expand_dims(X_valid, axis=3)
	valid = (X_valid.astype(np.float32), y_valid.astype(np.int32))

	print "loading test data"
	testmat = scipy.io.loadmat(os.path.join(filepath,'test.mat'))
	y_test = testmat['testdata']
	index = data_subset(y_test,class_range, negative=False)
	y_test = y_test[:, class_range]
	y_test = y_test[index,:]
	X_test = np.transpose(testmat['testxdata'],axes=(0,1,2))
	X_test = X_test[index,:,:]
	X_test = np.expand_dims(X_test, axis=3)
	test = (X_test.astype(np.float32), y_test.astype(np.int32))

	return train, valid, test 






	