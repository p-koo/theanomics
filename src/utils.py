#!/bin/python

import os
import sys
import numpy as np
from six.moves import cPickle

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




