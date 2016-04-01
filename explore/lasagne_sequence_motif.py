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
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_DeepSea
from train_utils import batch_generator
from file_utils import make_directory

np.random.seed(247) # for reproducibility

# load data and munge 
num_labels = 100
num_include = 1000000
class_range = range(num_labels)
data_path = "/home/peter/Data/DeepSea"
train, valid, test = load_DeepSea(data_path, num_include, class_range)

num_data, dim, sequence_length = train[0].shape

#------------------------------------------------------------------------------
# setup model
#------------------------------------------------------------------------------

weight_norm = 7
learning_rate = 0.002
momentum=0.98


# setup variables
input_var = T.tensor3('inputs') # T.matrix()
target_var = T.dmatrix()

# Input layer
network = InputLayer(shape=(None, dim, sequence_length), input_var=input_var)

# 1st convolution layer
network = Conv1DLayer(network,
                      num_filters=200,
                      filter_size=19,
                      W=GlorotUniform(),
                      nonlinearity=rectify)
network = MaxPool1DLayer(network, pool_size=3, stride=None)

# 3rd convolution layer
network = Conv1DLayer(network,
                      num_filters=300,
                      filter_size=8,
                      W=GlorotUniform(),
                      nonlinearity=rectify)
network = MaxPool1DLayer(network, pool_size=4, stride=None)

# Dense layer
network = DenseLayer(network,
                    num_units=1000,
                    W=GlorotUniform(),
                    b=Constant(0.05))
network = DropoutLayer(network, p=0.5)

# output layer
network = DenseLayer(network,
                    num_units=num_labels,
                    W=GlorotUniform(),
                    b=Constant(0.05), 
                    nonlinearity=sigmoid)

# setup loss 
prediction = get_output(network, input_var)
loss = T.mean(binary_crossentropy(prediction, target_var))


# calculate gradient with clipping
weight_norm = 10
params = get_all_params(network, trainable=True)
updates = adagrad(loss, params)

# test loss
test_prediction = get_output(network, deterministic=True)
test_loss = T.mean(binary_crossentropy(test_prediction, target_var))

# accuracy
accuracy = T.mean(T.eq(prediction, target_var))

# build theano function
print "building model"
train_fun = theano.function([input_var, target_var], [loss, accuracy], updates=updates)
test_fun = theano.function([input_var, target_var], [test_loss, accuracy])


#------------------------------------------------------------------------------
# train model 
#------------------------------------------------------------------------------


batch_size = 128
num_epochs = 100

num_train_batches = len(train[0]) // batch_size
num_valid_batches = len(valid[0]) // batch_size

train_batches = batch_generator(train[0], train[1], batch_size)
valid_batches = batch_generator(valid[0], valid[1], batch_size)


print("Starting training...")
valid_memory = []
performance = []
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = 0
    train_accuracy = 0
    for index in range(num_train_batches):
        X, y = next(train_batches)
        loss, accuracy = train_fun(X, y)
        train_loss += loss
        train_accuracy += accuracy

        remaining_time = (time.time()-start_time)*(num_train_batches-index-1)/(index+1)
        print("Estimated remaining time: {:.3f}s (batch {} of {}) ".format(
            remaining_time, index+1, num_train_batches ))
        print("  training loss:\t\t{:.6f}".format(train_loss/(index+1)))
        print("  training accuracy:\t\t{:.2f} %".format(train_accuracy/(index+1)*100))

    train_loss /= num_train_batches
    train_accuracy /= num_train_batches

    valid_loss = 0
    valid_accuracy = 0
    for _ in range(num_valid_batches):
        X, y = next(val_batches)
        loss, accuracy = test_fun(X, y)
        valid_loss += loss
        valid_accuracy += accuracy
    valid_loss /= num_valid_batches
    valid_accuracy /= num_valid_batches
    valid_memory.append(valid_loss)

    # stor training performance info
    performance.append([train_loss, train_accuracy, valid_loss, valid_accuracy])

    print("Epoch {} of {} took {:.3f}s".format(
        epoch+1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_loss))
    print("  validation loss:\t\t{:.6f}".format(valid_loss))
    print("  validation accuracy:\t\t{:.2f} %".format(valid_accuracy*100))

            

# get test loss and accuracy
num_test_batches = len(X_est) // batch_size
test_batches = batch_gen(X_test, y_test, batch_size)
test_loss = 0
test_accuracy = 0
for _ in range(num_test_batches):
    X, y = next(test_batches)
    loss, accuracy = test_fun(X, y)
    test_loss += loss
    test_accuracy += accuracy
test_loss /= num_test_batches
test_accuracy /= num_test_batches

print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_loss))
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy*100))





