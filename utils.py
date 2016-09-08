#!/bin/python
import os
import sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from scipy import stats
import pandas as pd


def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print "making directory: " + path

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print "making directory: " + outdir
	return outdir


def batch_generator(X, y, batch_size=128, shuffle=True):
	"""python generator to get a randomized minibatch"""
	"""
	while True:
		idx = np.random.choice(len(y), N)
		yield X[idx].astype('float32'), y[idx].astype('int32')
	"""
	if shuffle:
		indices = np.arange(len(X))
		np.random.shuffle(indices)
	for start_idx in range(0, len(X)-batch_size+1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx+batch_size]
		else:
			excerpt = slice(start_idx, start_idx+batch_size)
		yield X[excerpt], y[excerpt]


def one_hot_labels(label):
	"""convert categorical labels to one hot"""
	num_data = label.shape[0]
	num_labels = max(label)+1
	label_expand = np.zeros((num_data, num_labels))
	for i in range(num_data):
		label_expand[i, label[i]] = 1
	return label_expand


def pearson_corr_metric(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		corr = [stats.pearsonr(label, prediction)]
	else:		
		num_labels = label.shape[1]
		corr = []
		for i in range(num_labels):
			#corr.append(np.corrcoef(label[:,i], prediction[:,i]))
			corr.append(stats.pearsonr(label[:,i], prediction[:,i])[0])
		
	return corr


def rsquare_metric(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		y = label
		X = prediction
		m = np.dot(X,y)/np.dot(X, X)
		resid = y - m*X; 
		ym = y - np.mean(y); 
		rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
		rsquare = [rsqr2]
		slope = [m]
	else:		
		num_labels = label.shape[1]
		rsquare = []
		slope = []
		for i in range(num_labels):
			y = label[:,i]
			X = prediction[:,i]
			m = np.dot(X,y)/np.dot(X, X)
			resid = y - m*X; 
			ym = y - np.mean(y); 
			rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
			rsquare.append(rsqr2)
			slope.append(m)
	return rsquare, slope


def accuracy_metrics(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		score = accuracy_score(label, np.round(prediction))
		accuracy = np.array(score)
	else:
		num_labels = label.shape[1]
		accuracy = np.zeros((num_labels))
		for i in range(num_labels):
			score = accuracy_score(label[:,i], np.round(prediction[:,i]))
			accuracy[i] = score
	return accuracy


def roc_metrics(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		fpr, tpr, thresholds = roc_curve(label, prediction)
		score = auc(fpr, tpr)
		auc_roc= np.array(score)
		roc = [(fpr, tpr)]
	else:
		num_labels = label.shape[1]
		roc = []
		auc_roc = np.zeros((num_labels))
		for i in range(num_labels):
			fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
			score = auc(fpr, tpr)
			auc_roc[i]= score
			roc.append((fpr, tpr))
	return auc_roc, roc


def pr_metrics(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		precision, recall, thresholds = precision_recall_curve(label, prediction)
		score = auc(recall, precision)
		auc_pr = np.array(score)
		pr = [(precision, recall)]
	else:
		num_labels = label.shape[1]
		pr = []
		auc_pr = np.zeros((num_labels))
		for i in range(num_labels):
			precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
			score = auc(recall, precision)
			auc_pr[i] = score
			pr.append((precision, recall))
	return auc_pr, pr


def calculate_metrics(label, prediction, objective):
	"""calculate metrics for classification"""

	if (objective == "binary") | (objective == "categorical") | (objective == 'hinge'):
		ndim = np.ndim(label)
		if ndim == 1:
			label = one_hot_labels(label)
		accuracy = accuracy_metrics(label, prediction)
		auc_roc, roc = roc_metrics(label, prediction)
		auc_pr, pr = pr_metrics(label, prediction)
		mean = [np.nanmean(accuracy), np.nanmean(auc_roc), np.nanmean(auc_pr)]
		std = [np.nanstd(accuracy), np.nanstd(auc_roc), np.nanstd(auc_pr)]
		print "ROC"
		print auc_roc
		print "PR"
		print auc_pr
	elif (objective == 'squared_error'):
		corr = pearson_corr_metric(label, prediction)
		rsquare, slope = rsquare_metric(label, prediction)
		mean = [np.nanmean(corr), np.nanmean(rsquare), np.nanmean(slope)]
		std = [np.nanstd(corr), np.nanstd(rsquare), np.nanstd(slope)]
	return [mean, std]


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
	norm = np.outer(np.ones(4), np.sum(pwm, axis=0))
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
    


