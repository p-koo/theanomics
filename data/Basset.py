#!/bin/python
import os
import sys
import numpy as np
import h5py

def Basset(filepath, class_range=range(164), num_include=[]):
	"""Loads Basset dataset"""
	
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
	trainmat = h5py.File(os.path.join(filepath,'er.h5'), 'r')
	y_train = np.array(trainmat['train_out'])
	index = data_subset(y_train, class_range, negative=False)
	if num_include:
		index = index[0:num_include]
	y_train = y_train[:,class_range]
	y_train = y_train[index,:]
	X_train = np.transpose(np.array(trainmat['train_in']), axes=(0,1,3,2))
	X_train = X_train[index,:,:,:]
	train = (X_train, y_train)


	print "loading validation data"
	y_valid = np.array(trainmat['valid_out'])
	index = data_subset(y_valid,class_range, negative=False)
	y_valid = y_valid[:, class_range]
	y_valid = y_valid[index,:]
	X_valid = np.transpose(np.array(trainmat['valid_in']), axes=(0,1,3,2))
	X_valid = X_valid[index,:,:,:]
	valid = (X_valid, y_valid)

	print "loading testing data"
	y_test = np.array(trainmat['test_out'])
	index = data_subset(y_test,class_range, negative=False)
	y_test = y_test[:, class_range]
	y_test = y_test[index,:]
	X_test = np.transpose(np.array(trainmat['test_in']), axes=(0,1,3,2))
	X_test = X_test[index,:,:,:]
	test = (X_test, y_test)
	
	return train, valid, test





