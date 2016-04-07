
import sys
import os
import time
import numpy as np
import h5py

import scipy.io
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_MotifSimulation
from train_utils import batch_generator, early_stopping, epoch_train, print_progress
from file_utils import make_directory

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=1)
num_data, dim, sequence_length,_ = train[0].shape
num_labels = max(train[1])+1   # number of labels (output units)

#-------------------------------------------------------------------------------------

print "building theano model function"

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = lasagne.layers.InputLayer(shape=(None,  dim, sequence_length, 1),
                                    input_var=input_var)


network = lasagne.layers.Conv2DLayer(network, num_filters=200, filter_size=(8, 1), W=lasagne.init.GlorotUniform(), b=None)
network = BatchNormLayer(network)
#network = ParametricRectifierLayer(network, alpha=Constant(0.25), shared_axes='auto')
network = NonlinearityLayer(network, nonlinearity=rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 1))


network = lasagne.layers.Conv2DLayer(network, num_filters=200, filter_size=(8, 1), W=lasagne.init.GlorotUniform(), b=None)
network = BatchNormLayer(network)
#network = ParametricRectifierLayer(network, alpha=Constant(0.25), shared_axes='auto')
network = NonlinearityLayer(network, nonlinearity=rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 1))

network = lasagne.layers.DenseLayer(network, num_units=200, nonlinearity=lasagne.nonlinearities.rectify, b=None)
network = BatchNormLayer(network)
#network = ParametricRectifierLayer(network, alpha=Constant(0.25), shared_axes='auto')
network = NonlinearityLayer(network, nonlinearity=rectify)
network = DropoutLayer(network, p=.5)

network = lasagne.layers.DenseLayer(network, num_units=num_labels, nonlinearity=lasagne.nonlinearities.softmax)


prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)


test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()


test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fun = theano.function([input_var, target_var], [loss, test_acc], updates=updates)

test_fun = theano.function([input_var, target_var], [test_loss, test_acc])


#-------------------------------------------------------------------------------------

batch_size = 500
num_train_batches = len(train[0]) // batch_size
num_valid_batches = len(valid[0]) // batch_size
train_batches = batch_generator(train[0], train[1], batch_size)
valid_batches = batch_generator(valid[0], valid[1], batch_size)

print("Starting training...")
num_epochs = 100
patience = 0
valid_memory = []
performance = []
for epoch in range(num_epochs):
    
    train_loss, train_accuracy = epoch_train(train_fun, train_batches, num_train_batches, 1)

    valid_loss, valid_accuracy = epoch_train(test_fun, valid_batches, num_train_batches)
    print_progress(valid_loss, valid_accuracy, "cross-validation", epoch, num_epochs)    
    
    # store training performance info
    valid_memory.append(valid_loss)
    performance.append([train_loss, train_accuracy, valid_loss, valid_accuracy])
    
    # check for early stopping
    status = early_stopping(valid_memory, patience)
    if not status:
        print "Patience ran out... Early stopping."
        break
            

# get test loss and accuracy
num_test_batches = len(test[0]) // batch_size
test_batches = batch_generator(test[0], test[1], batch_size)
test_loss, test_accuracy = epoch_train(test_fun, test_batches, num_test_batches)
print_progress(test_loss, test_accuracy, "test")    

