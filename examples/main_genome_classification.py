#/bin/python
from __future__ import print_function

import os, sys
import numpy as np
sys.path.append('/Users/juliankimura/Desktop/deepomics')
import deepomics.neuralnetwork as nn
from deepomics import learn, utils
from models import standard_model, all_conv_model


sys.path.append('/Users/juliankimura/Desktop/data')
import load_data

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

data_path = '/Users/juliankimura/Desktop/data'
file_path = os.path.join(data_path, 'Localized_N=100000_S=200_M=50_G=20_data.pickle')
train, valid, test = load_data.simulation_pickle(file_path)

#-------------------------------------------------------------------------------------

# build network
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = train[1].shape[1]
network, placeholders, optimization = standard_model.model(shape, num_labels)

# build neural network class
nnmodel = nn.NeuralNet(network, placeholders)
nnmodel.inspect_layers()

# set output file paths
output_name = 'test'
utils.make_directory(data_path, 'Results')
file_path = os.path.join(data_path, 'Results', output_name)
nntrainer = nn.NeuralTrainer(nnmodel, optimization, save='best', file_path=file_path)

# train model
learn.train_minibatch(nntrainer, data={'train': train, 'valid': valid}, 
                              batch_size=100, num_epochs=500, patience=10, verbose=1)

# load best model --> lowest cross-validation error
nntrainer.set_best_parameters()

# test model
nntrainer.test_model(test, name="test", batch_size=100)

# save all performance metrics (train, valid, test)
nntrainer.save_all_metrics(file_path)














