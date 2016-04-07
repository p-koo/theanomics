
import sys
import os
import numpy as np
import theano
import theano.tensor as T
from six.moves import cPickle
import matplotlib.pyplot as plt
sys.path.append('/home/peter/GenomeMotifs/src')
from neuralnetwork import NeuralNetworkModel
from data_utils import load_MotifSimulation
from utils import make_directory
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score

#------------------------------------------------------------------------------
# load data

filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=1)

#-------------------------------------------------------------------------------------
# load model

model_name="simple_genome_motif_model"

shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = max(train[1])+1

save = 'all' # final
savename = 'mytest'
savepath = os.path.join(dirpath,'Results',savename)

# load best model
filepath = savepath + "_3.pickle"
f = open(filepath, 'rb')
all_param_values = cPickle.load(f)
f.close()
nnmodel = NeuralNetworkModel(model_name, shape=shape, num_labels=num_labels);
nnmodel.set_model_parameters(all_param_values)
cost, prediction = nnmodel.test_results(test)

# convert predictions to matrix format
y = np.zeros((test[1].shape[0],nnmodel.num_labels))
for i in range(test[1].shape[0]):
	y[i,test[1][i]] = 1

# get accuracy, roc, pr metrics
roc = []
pr = []
accuracy = np.zeros((num_labels))
auc_roc = np.zeros((num_labels))
auc_pr = np.zeros((num_labels))
for i in range(num_labels):
	print i

	# accuracy score
	score = accuracy_score(y[:,i], np.round(prediction[:,i]))
	accuracy[i] = score
	print score

	# receiver-operator characteristic curve
	fpr, tpr, thresholds = roc_curve(y[:,i], prediction[:,i])
	score = auc(fpr, tpr)
	auc_roc[i]= score
	roc.append((fpr, tpr))
	print score

	# precision recall curve
	precision, recall, thresholds = precision_recall_curve(y[:,i], prediction[:,i])
	score = auc(recall, precision)
	auc_pr[i] = score
	pr.append((precision, recall))
	print score
	print '------------------------------------------'



print str(np.mean(accuracy))
print str(np.mean(auc_roc))
print str(np.mean(auc_pr))

