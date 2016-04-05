#!/bin/python
from __future__ import print_function

import sys
import os
import time
import numpy as np
import h5py
import scipy.io
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import binary_crossentropy
from lasagne.updates import nesterov_momentum

np.random.seed(247) # for reproducibility

def load_dataset():

    print 'loading cross-validation data'    
    validmat = scipy.io.loadmat('data/valid.mat')
    X_valid = np.transpose(validmat['validxdata'],axes=(0,2,1))
    y_valid = validmat['validdata']
    
    print 'loading test data'    
    testmat = scipy.io.loadmat('data/test.mat')
    X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
    y_test = testmat['testdata']
  
    print 'loading train data'    
    trainmat = h5py.File('data/train.mat')
    X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
    y_train = np.array(trainmat['traindata']).T

    return X_train, y_train, X_valid, y_valid, X_test, y_test




#------------------------------------------------------------------------------
# setup model
#------------------------------------------------------------------------------

weight_norm = 7
learning_rate = 0.002
momentum=0.98


# setup variables
input_var = T.matrix()
target_var = T.ivector()

# create model
network = conv_genome(input_var)


#------------------------------------------------------------------------------
# train model 
#------------------------------------------------------------------------------


batch_size = 128
num_epochs = 100

num_train_batches = len(X_train) // batch_size
num_valid_batches = len(X_valid) // batch_size

train_batches = batch_gen(X_train, y_train, batch_size)
valid_batches = batch_gen(X_valid, y_valid, batch_size)

early_stop = 1
patience = 5

print("Starting training...")
valid_memory = []
performance = []
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = 0
    train_accuracy = 0
    for _ in range(num_train_batches):
        X, y = next(train_batches)
        loss, accuracy = train_fun(X, y)
        train_loss += loss
        train_accuracy += accuracy
    train_loss /= num_train_batches
    train_accuracy /= num_train_batches

    valid_loss = 0
    valid_accuracy = 0
    for _ in range(num_valid_batches):
        X, y = next(val_batches)
        loss, accuracy = valid_fun(X, y)
        valid_loss += loss
        valid_acc += accuracy
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

    if early_stop == 1:
        status = early_stopping(valid_memory, patience)
        if status == 0:
            print "Early stopping: ran out of patience..."
            break
            

# get test loss and accuracy
num_test_batches = len(X_est) // batch_size
test_batches = batch_gen(X_test, y_test, batch_size)
test_loss = 0
test_accuracy = 0
for _ in range(num_test_batches):
    X, y = next(test_batches)
    loss, accuracy = valid_fun(X, y)
    test_loss += loss
    test_accuracy += accuracy
test_loss /= num_test_batches
test_accuracy /= num_test_batches

print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_loss))
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy*100))





