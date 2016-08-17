#/bin/python
import os, sys
import numpy as np
sys.path.append('..')
from src.neuralnetwork import NeuralNet, NeuralTrainer
import src.train as fit 
import load_data
from models.test_motif_model import model

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'Unlocalized_N=100000_S=200_M=300_G=30_data.pickle')
train, valid, test = load_data.simulation_pickle(filepath)

#-------------------------------------------------------------------------------------

# build network
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = train[1].shape[1]
network, input_var, target_var, optimization = model(shape, num_labels)

# build neural network class
nnmodel = NeuralNet(network, input_var, target_var)
nnmodel.inspect_layers()

# set output file paths
output_name = 'test'
filepath = os.path.join(datapath, 'Results', output_name)
nntrainer = NeuralTrainer(nnmodel, optimization, save='best', filepath=filepath)

# train model
fit.train_minibatch(nntrainer, data={'train': train, 'valid': valid}, 
                              batch_size=100, num_epochs=500, patience=10, verbose=1)

# load best model --> lowest cross-validation error
nntrainer.set_best_parameters()

# test model
nntrainer.test_model(test, batch_size, "test")

# save all performance metrics (train, valid, test)
nntrainer.save_all_metrics(filepath)














