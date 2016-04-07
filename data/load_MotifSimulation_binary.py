#!/bin/python
import os
import sys
import numpy as np
import h5py
import scipy.io
from six.moves import cPickle
	
def load_MotifSimulation_binary(filepath):
	# setup paths for file handling

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

	train = (X_train, y_train)
	valid = (X_val, y_val)
	test = (X_test, y_test)

	return train, valid, test






