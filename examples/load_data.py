#/bin/python
#!/bin/python
import os, sys, h5py
import numpy as np
import scipy.io
from six.moves import cPickle

def simulation_pickle(filepath, class_type='binary'):

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

	X_train = train[0].transpose((0,1,2)).astype(np.float32)
	y_train = train[1].astype(np.int32)
	X_val = cross_validation[0].transpose((0,1,2)).astype(np.float32)
	y_val = cross_validation[1].astype(np.int32)
	X_test = test[0].transpose((0,1,2)).astype(np.float32)
	y_test = test[1].astype(np.int32)

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	if class_type == 'categorical':
		y_train = np.argmax(y_train, axis=1)
		y_val = np.argmax(y_val, axis=1)
		y_test = np.argmax(y_test, axis=1)

	train = (X_train, y_train)
	valid = (X_val, y_val)
	test = (X_test, y_test)

	return train, valid, test


def simulation_hdf5(filepath, class_type='binary'):
	
	trainmat = h5py.File(filepath, 'r')
	X_train = np.array(trainmat['trainx']).astype(np.float32)
	y_train = np.array(trainmat['trainy']).astype(np.float32)
	model_train = np.array(trainmat['trainmodel']).astype(np.float32)

	X_val = np.array(trainmat['validx']).astype(np.float32)
	y_val = np.array(trainmat['validy']).astype(np.int32)
	model_val = np.array(trainmat['validmodel']).astype(np.float32)

	X_test = np.array(trainmat['testx']).astype(np.float32)
	y_test = np.array(trainmat['testy']).astype(np.int32)
	model_test = np.array(trainmat['testmodel']).astype(np.float32)

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	if class_type == 'categorical':
		y_train = np.argmax(y_train, axis=1)
		y_val = np.argmax(y_val, axis=1)
		y_test = np.argmax(y_test, axis=1)

	train = (X_train, y_train, model_train)
	valid = (X_val, y_val, model_val)
	test = (X_test, y_test, model_test)

	return train, valid, test

