#!/bin/python

import os
import sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score

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


def one_hot_labels(label):
	num_data = label.shape[0]
	num_labels = max(label)+1
	label_expand = np.zeros((num_data, num_labels))
	for i in range(num_data):
		label_expand[i, label[i]] = 1
	return label_expand


def accuracy_metrics(label, prediction):
	num_labels = label.shape[1]
	accuracy = np.zeros((num_labels))
	for i in range(num_labels):
		# accuracy score
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

def calculate_metrics(label, prediction):
	num_samples = len(prediction)
	ndim = np.ndim(label)
	if ndim == 1:
		label = one_hot_labels(label)

	accuracy = accuracy_metrics(label, prediction)
	auc_roc, roc = roc_metrics(label, prediction)
	auc_pr, pr = pr_metrics(label, prediction)
	mean = [np.nanmean(accuracy), np.nanmean(auc_roc), np.nanmean(auc_pr)]
	std = [np.std(accuracy), np.std(auc_roc), np.std(auc_pr)]
	return mean, std, roc, pr