
import sys
import os
import numpy as np
import theano
import theano.tensor as T

sys.path.append('/home/peter/Code/GenomeMotifs/src')
from neuralnetwork import NeuralNetworkModel
from data_utils import load_MotifSimulation
from utils import make_directory
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=1)

#-------------------------------------------------------------------------------------
# train model

model_name="simple_genome_motif_model"

shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = max(train[1])+1
nnmodel = NeuralNetworkModel(model_name, shape=shape, num_labels=num_labels);

save = 'all' # final
filename = 'sim1'
newpath = make_directory('Results', dirpath, verbose=1)
filepath = os.path.join(newpath,filename)
nnmodel.train(train, valid, test, batch_size=500, num_epochs=500, 
                  patience=5, save=save, filepath=filepath, verbose=1)



