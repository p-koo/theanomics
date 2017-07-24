#!/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from scipy import stats
import pandas as pd



__all__ = [
	"batch_generator",
	"make_directory",
	"normalize_pwm",
	"meme_generate",
	"load_JASPAR_motifs"
]


def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir


def batch_generator(X, batch_size=128, shuffle=True):
	"""python generator to get a randomized minibatch"""
	"""
	while True:
		idx = np.random.choice(len(y), N)
		yield X[idx].astype('float32'), y[idx].astype('int32')
	"""
	if isinstance(X, (list, tuple)):
		num_var = len(X)
		num_data = len(X[0])
	else:
		num_var = 1
		num_data = len(X)

	if shuffle:
		indices = np.arange(num_data)
		np.random.shuffle(indices)

	for start_idx in range(0, num_data-batch_size+1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx+batch_size]
		else:
			excerpt = slice(start_idx, start_idx+batch_size)

		if num_var > 1:
			X_batch = []
			for i in range(num_var):
				X_batch.append(X[i][excerpt])
			yield X_batch
		else:
			if isinstance(X, (list, tuple)):
				yield [X[0][excerpt]]
			else:
				yield [X[excerpt]]



def get_performance(savepath):
	with open(savepath, 'rb') as f:
		name = cPickle.load(f)
		cost = cPickle.load(f)
		metric = cPickle.load(f)
		metric_std = cPickle.load(f)
		roc = cPickle.load(f)
		pr = cPickle.load(f)
	return cost, metric, metric_std, roc, pr


def load_JASPAR_motifs(jaspar_path, MAX):

	with open(jaspar_path, 'rb') as f: 
		jaspar_motifs = cPickle.load(f)

	motifs = []
	for jaspar in jaspar_motifs:
		length = len(jaspar)
		if length < MAX:
			offset = MAX - length
			firstpart = offset // 2
			secondpart = offset - firstpart
			matrix = np.vstack([np.ones((firstpart,4))*.25, jaspar,  np.ones((secondpart,4))*.25])
			motifs.append(matrix)

		elif length > MAX:
			offset = length - MAX
			firstpart = offset // 2
			secondpart = offset - firstpart
			matrix = jaspar[0+firstpart:length-secondpart,:]
			motifs.append(matrix)

		elif length == MAX:
			motifs.append(jaspar)

	motifs = np.array(motifs)
	motifs = np.expand_dims(np.transpose(motifs, (0,2,1)), axis=3)

	return motifs


def normalize_pwm(pwm, method=2):
	if method == 1:
		pwm = pwm/np.max(np.abs(pwm))
		pwm += .25
		pwm[pwm<0] = 0
	elif method == 2:
		MAX = np.max(pwm)
		pwm = pwm/MAX*4
		pwm = np.exp(pwm)
	norm = np.outer(np.ones(pwm.shape[0]), np.sum(pwm, axis=0))
	return pwm/norm


def meme_generate(W, output_file='meme.txt', prefix='filter'):

    # background frequency        
    nt_freqs = [1./4 for i in range(4)]

    # open file for writing
    f = open(output_file, 'w')

    # print intro material
    print >> f, 'MEME version 4'
    print >> f, ''
    print >> f, 'ALPHABET= ACGT'
    print >> f, ''
    print >> f, 'Background letter frequencies:'
    print >> f, 'A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs)
    print >> f, ''

    for j in range(len(W)):
        pwm = np.array(W[j])

        print >> f, 'MOTIF %s%d' % (prefix, j)
        print >> f, 'letter-probability matrix: alength= 4 w= %d nsites= %d' % (pwm.shape[1], 7)
        for i in range(pwm.shape[1]):
            print >> f, '%.4f %.4f %.4f %.4f' % tuple(pwm[:,i])
        print >> f, ''

    f.close()
    


