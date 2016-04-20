#/bin/python
import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNet
from src import train as fit
from src import make_directory 
from models import load_model
from data import load_data
#np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data
"""
name = 'Basset' # 'DeepSea'
datapath = '/home/peter/Data/'+name
options = {"class_range": range(51,55)}# 
train, valid, test = load_data(name, datapath, options)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

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

name = 'MotifSimulation_binary'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'N=100000_S=200_M=10_G=20_data.pickle')
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])
#"""

#-------------------------------------------------------------------------------------


# build model
model_name = "jaspar_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)

#nnmodel.print_layers()
params = nnmodel.get_model_parameters()

# set output file paths
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, model_name)

# train model
batch_size = 128
nnmodel = fit.train_valid_minibatch(nnmodel, train, valid, batch_size, 
				num_epochs=2, patience=0, verbose=1, filepath=filepath)

model_parameters = nnmodel.get_model_parameters()
print params[0].shape
print model_parameters[0].shape
print np.sum(params[0] == model_parameters[0])

model_name = "jaspar_motif_model2"
nnmodel = NeuralNet(model_name, shape, num_labels)
nnmodel.set_model_parameters(model_parameters)
nnmodel = fit.train_valid_minibatch(nnmodel, train, valid, batch_size, 
				num_epochs=100, patience=10, verbose=1, filepath=filepath)

# save best model --> lowest cross-validation error
min_loss, min_index = nnmodel.get_min_loss()
savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
nnmodel.set_parameters_from_file(savepath)

# test set perfomance
nnmodel.get_min_loss()

savepath = filepath + "_epoch_" + str(1) + ".pickle"
nnmodel.test_model(test, batch_size, "test")
nnmodel.save_all_metrics(filepath)

# save all performance metrics (train, valid, test)
savepath = filepath + "_best.pickle"
nnmodel.save_model_parameters(savepath)

# monitor/save test performance with parameters for each training epoch
num_train_epochs = nnmodel.get_num_epochs()
performance = fit.test_model_all(nnmodel, test, batch_size, num_train_epochs, filepath)

# save test performance
performance.save_metrics(filepath)

model_parameters = nnmodel.get_model_parameters()
print params[0].shape
print model_parameters[0].shape
print np.sum(params[0] == model_parameters[0])















