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
	num_labels = label.shape[1]
	corr = []
	for i in range(num_labels):
		#corr.append(np.corrcoef(label[:,i], prediction[:,i]))
		corr.append(stats.spearmanr(label[:,i], prediction[:,i])[0])
		
	return corr

def rsquare_metric(label, prediction):
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

def calculate_metrics(label, prediction, objective):
	"""calculate metrics for classification"""

	if (objective == "binary") | (objective == "categorical") | (objective == "multi-binary") | (objective == 'hinge'):
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
	elif (objective == 'ols') | (objective == 'gls') | (objective == 'autoencoder'):
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



def seq_logo(pwm, height=100, nt_width=20, norm=0, rna=1):
    """generate a sequence logo from a pwm"""
    
    def load_alphabet(filepath='./nt', rna):
        """load images of nucleotide alphabet """
        df = pd.read_table(os.path.join(filepath, 'A.txt'), header=None);
        A_img = df.as_matrix()
        A_img = np.reshape(A_img, [72, 65, 3], order="F").astype(np.uint8)

        df = pd.read_table(os.path.join(filepath, 'C.txt'), header=None);
        C_img = df.as_matrix()
        C_img = np.reshape(C_img, [76, 64, 3], order="F").astype(np.uint8)

        df = pd.read_table(os.path.join(filepath, 'G.txt'), header=None);
        G_img = df.as_matrix()
        G_img = np.reshape(G_img, [76, 67, 3], order="F").astype(np.uint8)

        if rna == 1:
            df = pd.read_table(os.path.join(filepath, 'U.txt'), header=None);
            T_img = df.as_matrix()
            T_img = np.reshape(T_img, [74, 57, 3], order="F").astype(np.uint8)
        else:
	        df = pd.read_table(os.path.join(filepath, 'T.txt'), header=None);
	        T_img = df.as_matrix()
	        T_img = np.reshape(T_img, [72, 59, 3], order="F").astype(np.uint8)

        return A_img, C_img, G_img, T_img


    def get_nt_height(pwm, height, norm):
        """get the heights of each nucleotide"""

        def entropy(p):
            """calculate entropy of each nucleotide"""
            s = 0
            for i in range(4):
                if p[i] > 0:
                    s -= p[i]*np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(4) - entropy(pwm[:, i]))*height;
            heights[:,i] = np.floor(pwm[:,i]*total_height);
        return heights.astype(int)

    
    # get the alphabet images of each nucleotide
    A_img, C_img, G_img, T_img = load_alphabet(filepath='./nt', rna=rna)
    
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)
    
    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    total_height = np.sum(heights,axis=0)
    max_height = np.max(total_height)
    logo = np.ones((height*2, width, 3)).astype(int)*255;
    for i in range(num_seq):
        remaining_height = total_height[i];
        offset = max_height-remaining_height
        nt_height = np.sort(heights[:,i]);
        index = np.argsort(heights[:,i])

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                resize = (nt_height[j], nt_width)
                if index[j] == 0:
                    nt_img = imresize(A_img, resize)
                elif index[j] == 1:
                    nt_img = imresize(C_img, resize)
                elif index[j] == 2:
                    nt_img = imresize(G_img, resize)
                elif index[j] == 3:
                    nt_img = imresize(T_img, resize)

                # determine location of image
                height_range = range(remaining_height-nt_height[j], remaining_height)
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range+offset, width_range[m],k] = nt_img[:,m,k];

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


