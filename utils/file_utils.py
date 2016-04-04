#!/bin/python

import os
import sys
import numpy as np
from six.moves import cPickle
from lasagne.layers import get_all_param_values, set_all_param_values

def make_directory(foldername, path, verbose=1):
	"""make a directory"""
	if not os.path.isdir(path):
		os.mkdir(path)
		print "making directory: " + path

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print "making directory: " + outdir
	return outdir


def save_model(network, filepath):
	print "saving model parameters to: " + filepath
	all_param_values = get_all_param_values(network)
	f = open(filepath, 'wb')
	cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


def load_model(network, filepath):
	print "loading model parameters from:" + filepath
	f = open(filepath, 'rb')
	all_param_values = cPickle.load(f)
	f.close()
	return set_all_param_values(network, all_param_values)


def save_train_performance(filepath, performance):
	print "saving training performance to:" + filepath
	f = open(filepath, 'wb')
	for data in performance:
		cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()



