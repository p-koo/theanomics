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
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'MotifSimulation_binary'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'N=100000_S=200_M=10_G=20_data.pickle')
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

#-------------------------------------------------------------------------------------


# build model
model_name = "binary_genome_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)


# set output file paths
outputname = 'binary'
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, outputname)

# train model
batch_size = 128
nnmodel = fit.train_valid_minibatch(nnmodel, train, valid, batch_size, num_epochs=500, patience=5, verbose=1, filepath=filepath)

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















