#/bin/python
import os, sys, gzip
import numpy as np
import cPickle as pickle
sys.setrecursionlimit(10000)

sys.path.append('..')
from src import NeuralNet
from src import train as fit
from src import make_directory 
from models import load_model
from data import load_data
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

datapath = '/home/peter/Data/mnist/'
filename = 'mnist.pkl.gz'
f = gzip.open(os.path.join(datapath, filename), 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()
X_train, y_train = train_set
X_valid, y_valid = valid_set
X_test, y_test = test_set

X_train = np.reshape(X_train, (-1, 1, 28, 28))
X_valid = np.reshape(X_valid, (-1, 1, 28, 28))
X_test = np.reshape(X_test, (-1, 1, 28, 28))

train = (X_train, y_train)
valid = (X_valid, y_valid)
test = (X_test, y_test)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = max(train[1])+1

#-------------------------------------------------------------------------------------

# load model parameters
model_name = "MNIST_CNN_model"
nnmodel = NeuralNet(model_name, shape, num_labels)
nnmodel.print_layers()

# train model
outputname = 'new'
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, outputname)

# train model
batch_size = 128
nnmodel = fit.train_valid_minibatch(nnmodel, train, valid, batch_size, num_epochs=500, patience=5, verbose=1, filepath=filepath)

# load best model --> lowest cross-validation error
min_loss, min_index = nnmodel.get_min_loss()
savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
nnmodel.set_parameters_from_file(savepath)

# test set perfomance
nnmodel.test_model(test, batch_size, "test")


















