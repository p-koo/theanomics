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
from six.moves import cPickle
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'MotifSimulation_correlated'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'synthetic_correlated_motifs_100000_1.hdf5')
#filepath = os.path.join(datapath, 'synthetic_random_motifs_100000_1.hdf5')

"""
name = 'MotifSimulation_binary'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'N=100000_S=200_M=30_G=20_data.pickle')
"""
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])


# calculate correlations
labels = np.vstack([train[1], valid[1]])
N = labels.shape[0]
rho_ij = np.zeros((num_labels, num_labels))
for i in range(num_labels):
    p_i = np.sum(labels[:,i])/N
    for j in range(i):
        p_j = np.sum(labels[:,j])/N    
        p_ij = np.sum(labels[:,i]*labels[:,j])/N
        norm = np.sqrt(p_i*(1-p_i)) * np.sqrt(p_j*(1-p_j))
        rho_ij[j,i] = (p_ij - p_i*p_j)/norm

f = open('/home/peter/Code/Deepomics/examples/rho_ij.pickle','wb')
cPickle.dump(rho_ij, f)
f.close()


#-------------------------------------------------------------------------------------

# build model
model_name = "binary_genome_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)
#nnmodel.print_layers()

# set output file paths
outputname = 'random_2'
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, outputname)

# train model
batch_size = 100
#nnmodel = fit.anneal_train_valid_minibatch(nnmodel, train, valid, batch_size, num_epochs=500, patience=5, verbose=1, filepath=filepath)
nnmodel = fit.train_minibatch(nnmodel, train, valid, batch_size=batch_size, num_epochs=500, 
			patience=5, verbose=1, filepath=filepath)

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
#num_train_epochs = nnmodel.get_num_epochs()
#performance = fit.test_model_all(nnmodel, test, batch_size, num_train_epochs, filepath)

# save test performance
#performance.save_metrics(filepath)















