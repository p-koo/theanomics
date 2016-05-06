#/bin/python
import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNet
from src import train as fit
#from src import train_learning_decay
from src import make_directory 
from models import load_model
from data import load_data
np.random.seed(727) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'Basset' # 'DeepSea'
datapath = '/home/peter/Data/'+name
options = {"class_range": range(40)}# 
train, valid, test = load_data(name, datapath, options)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

"""
print "total number of training samples:"
print train[0].shape
print train[1].shape

print "total number of validation samples:"
print valid[0].shape

print "number of positive training samples for each class:"
print np.sum(train[1], axis=0)
print "number of positive validation samples for each class:"
print np.sum(valid[1], axis=0)
"""
#-------------------------------------------------------------------------------------

# load model parameters
model_name = "binary_genome_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)

nnmodel.print_layers()

# set output file paths
filename = model_name + "_new"
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, filename)

# train model
nnmodel = fit.train_learning_decay(nnmodel, train, valid, learning_rate=.01, 
								batch_size=68, num_epochs=500, patience=10, 
								learn_patience=1, decay=0.5, verbose=1, filepath=filepath)
nnmodel = fit.train_learning_decay(nnmodel, train, valid, learning_rate=.001, 
								batch_size=256, num_epochs=500, patience=5, 
								learn_patience=1, decay=0.5, verbose=1, filepath=filepath)

# save best model --> lowest cross-validation error
min_loss, min_index = nnmodel.get_min_loss()
savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
nnmodel.set_parameters_from_file(savepath)
savepath = filepath + "_best.pickle"
nnmodel.save_model_parameters(savepath)

# test model
nnmodel.test_model(test, batch_size, "test")

# save all performance metrics (train, valid, test)
nnmodel.save_all_metrics(filepath)

# monitor/save test performance with parameters for each training epoch
num_train_epochs = nnmodel.get_num_epochs()
performance = fit.test_model_all(nnmodel, test, batch_size, num_train_epochs, filepath)

# save test performance
performance.save_metrics(filepath)













