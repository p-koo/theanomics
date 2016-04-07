#/bin/python
import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNets, make_directory
from models import load_model
from data import load_data

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'genome_motif_model'
datapath = '/home/peter/Data/DeepSea'
options = {"num_include": 1000000, "class_range": range(100)}
train, valid, test = load_data(name, datapath, options)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

#-------------------------------------------------------------------------------------

# load model parameters
model_name = "genome_motif_model"
model_layers, input_var, target_var, optimization = load_model(model_name, shape, num_labels)

# build model
nnmodel = NeuralNets(model_layers, input_var, target_var, optimization)

# train model
filename = 'genome_motif_model'
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, filename)
nnmodel.train_model(train, valid, batch_size=500, num_epochs=500, patience=5, verbose=1, filepath=filepath)

# set and save best model 
nnmodel.save_best_model(filepath)

# test set perfomance
nnmodel.test_model(test)

# save all performance metrics (train, valid, test)
nnmodel.save_all_metrics(filepath)

# monitor/save test performance with parameters for each training epoch
nnmodel.test_model_all(test, filepath)

















