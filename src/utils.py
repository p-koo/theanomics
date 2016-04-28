#!/bin/python
import os
import sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score


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
		yield X[excerpt].astype('float32'), y[excerpt].astype('int32')


def one_hot_labels(label):
	"""convert categorical labels to one hot"""
	num_data = label.shape[0]
	num_labels = max(label)+1
	label_expand = np.zeros((num_data, num_labels))
	for i in range(num_data):
		label_expand[i, label[i]] = 1
	return label_expand


def calculate_metrics(label, prediction):
	"""calculate metrics for classification"""

	def accuracy_metrics(label, prediction):
		num_labels = label.shape[1]
		accuracy = np.zeros((num_labels))
		for i in range(num_labels):
			score = accuracy_score(label[:,i], np.round(prediction[:,i]))
			accuracy[i] = score
		return accuracy

	def roc_metrics(label, prediction):
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
		num_labels = label.shape[1]
		pr = []
		auc_pr = np.zeros((num_labels))
		for i in range(num_labels):
			precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
			score = auc(recall, precision)
			auc_pr[i] = score
			pr.append((precision, recall))
		return auc_pr, pr

	num_samples = len(prediction)
	ndim = np.ndim(label)
	if ndim == 1:
		label = one_hot_labels(label)

	accuracy = accuracy_metrics(label, prediction)
	auc_roc, roc = roc_metrics(label, prediction)
	auc_pr, pr = pr_metrics(label, prediction)
	mean = [np.nanmean(accuracy), np.nanmean(auc_roc), np.nanmean(auc_pr)]
	std = [np.std(accuracy), np.std(auc_roc), np.std(auc_pr)]
	#print "ROC"
	#print auc_roc
	#print "PR"
	#print auc_pr
	return mean, std, roc, pr


def get_performance(savepath):
	with open(savepath, 'rb') as f:
		name = cPickle.load(f)
		cost = cPickle.load(f)
		metric = cPickle.load(f)
		metric_std = cPickle.load(f)
		roc = cPickle.load(f)
		pr = cPickle.load(f)
	return cost, metric, metric_std, roc, pr


def get_layer_activity(layer, x):

	# compile theano function
	input_var = T.tensor4('input').astype(theano.config.floatX)
	get_activity = theano.function([input_var], get_output(layer, input_var))

	# get activation info
	activity = get_activity(x)

	return activity



def load_JASPAR_motifs(jaspar_path, MAX):

	with open(jaspar_path, 'rb') as f: 
		jaspar_motifs = cPickle.load(f)

	motifs = []
	for jaspar in jaspar_motifs:
		length = len(jaspar)
		print length
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



