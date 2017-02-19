from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

sys.path.append('/Users/juliankimura/Desktop/deepomics')
import deepomics.neuralnetwork as nn
from deepomics import learn, utils
from models import vae_model

np.random.seed(247)   # for reproducibility

#------------------------------------------------------------------------------
# load data

filename = 'frey_rawface.mat'
data_path = '/Users/juliankimura/Desktop/data/FreyFaces'
matfile = loadmat(os.path.join(data_path, filename))
all_data = (matfile['ff'] / 255.).T

indices = np.arange(len(all_data))
np.random.shuffle(indices)
indices

width = 20
height = 28
X_train = all_data[indices[:1500]]
X_valid = all_data[indices[1500:]]

#-------------------------------------------------------------------------------------

# build network
shape = (None, X_train.shape[1])
network, placeholders, optimization = vae_model.model(shape)

# build neural network class
nnmodel = nn.NeuralNet(network, placeholders)
nnmodel.inspect_layers()

# set output file paths
output_name = 'test'
utils.make_directory(data_path, 'Results')
file_path = os.path.join(data_path, 'Results', output_name)
nntrainer = nn.NeuralTrainer(nnmodel, optimization, save='best', file_path=file_path)

# train model
learn.train_minibatch(nntrainer, data={'train': X_train, 'valid': X_valid}, 
                              batch_size=100, num_epochs=500, patience=10, verbose=1)

