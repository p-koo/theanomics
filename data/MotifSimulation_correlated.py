#!/bin/python
import os
import sys
import numpy as np
import h5py
import scipy.io
from six.moves import cPickle
	
def MotifSimulation_correlated(filepath):
	# setup paths for file handling

	trainmat = h5py.File(filepath, 'r')
	X_train = np.array(trainmat['trainx'])
	y_train = np.array(trainmat['trainy'])

	X_val = np.array(trainmat['validx'])
	y_val = np.array(trainmat['validy'])
	X_test = np.array(trainmat['testx'])
	y_test = np.array(trainmat['testy'])

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	train = (X_train, y_train)
	valid = (X_val, y_val)
	test = (X_test, y_test)

	return train, valid, test






