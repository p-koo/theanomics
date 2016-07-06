#/bin/python
import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNet
from src import train as fit
from src import make_directory 
from models import load_model
from six.moves import cPickle
import h5py

np.random.seed(247) # for reproducibility


#------------------------------------------------------------------------------

outputname = 'LinearRegression'
name = 'dataset_norm.hd5f'
datapath='/home/peter/Data/CMAP'
trainmat = h5py.File(os.path.join(datapath, name), 'r')

filepath = make_directory(datapath, 'Results')
filepath = os.path.join(filepath, outputname)

#-------------------------------------------------------------------------------------

# build model
model_name = "CMAP_model"
nnmodel = NeuralNet(model_name, shape=[], num_labels=[])
nnmodel.print_layers()


#-------------------------------------------------------------------------------------
# train model

batch_size = 100        
num_files = 5        
num_epochs = 500    
patience = 20
verbose = 1

for epoch in range(num_epochs):
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    train_loss = 0
    for i in range(4):
        sys.stdout.write("\r  File %d \n"%(i+1))
        landmark= np.array(trainmat['landmark'+str(i)]).astype(np.float32)
        nonlandmark = np.array(trainmat['nonlandmark'+str(i)]).astype(np.float32)

        # training set
        train_loss = nnmodel.train_step((landmark,nonlandmark), batch_size, verbose)
        nnmodel.train_monitor.add_loss(train_loss)

    # test current model with cross-validation data and store results
    landmark= np.array(trainmat['landmark4']).astype(np.float32)
    nonlandmark = np.array(trainmat['nonlandmark4']).astype(np.float32)
    valid_loss = nnmodel.test_model((landmark,nonlandmark), batch_size, "valid")

    # save model
    if filepath:
        savepath = filepath + "_epoch_" + str(epoch) + ".pickle"
        nnmodel.save_model_parameters(savepath)

    # check for early stopping                  
    status = nnmodel.valid_monitor.early_stopping(valid_loss, epoch, patience)
    if not status:
        break