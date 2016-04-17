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

name = 'Basset' # 'DeepSea'
datapath = '/home/peter/Data/'+name
options = {"class_range": range(150)}# 
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

#-------------------------------------------------------------------------------------

# load model parameters
model_name = "conv_LSTM_model"
nnmodel = NeuralNet(model_name, shape, num_labels)

#nnmodel.print_layers()

# set output file paths
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, model_name)

# train model
batch_size = 128
nnmodel = fit.train_valid_minibatch(nnmodel, train, valid, batch_size, num_epochs=500, patience=10, verbose=1, filepath=filepath)

# save best model --> lowest cross-validation error
min_cost, min_index = nnmodel.get_min_cost()
savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
nnmodel.set_parameters_from_file(savepath)

# test set perfomance
nnmodel.get_min_cost()

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
















