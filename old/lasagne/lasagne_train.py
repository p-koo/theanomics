
import sys
import os
import time
import numpy as np
import h5py

from six.moves import cPickle
import scipy.io
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_DeepSea
from train_utils import batch_generator
from file_utils import make_directory

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
filename = 'N=100000_S=200_M=10_G=20_data.pickle'

# setup paths for file handling
filepath = os.path.join('data',filename)
name, ext = os.path.splitext(filename)
outdir = os.path.join('data',name)
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print "making directory: " + outdir
outpath = os.path.join(outdir,'model_log.hdf5')

# load training set
print "loading data from: " + filepath
f = open(filepath, 'rb')
print "loading train data"
train = cPickle.load(f)
print "loading cross-validation data"
cross_validation = cPickle.load(f)
print "loading test data"
test = cPickle.load(f)
f.close()

X_train = train[0].transpose((0,1,2))
y_train = train[1]
X_valid = cross_validation[0].transpose((0,1,2))
y_valid = cross_validation[1]
X_test = test[0].transpose((0,1,2))
y_test = test[1]
num_data, dim, sequence_length = X_train.shape
num_labels = y_train.shape[1]   # number of labels (output units)

train = (X_train, y_train)
valid = (X_valid, y_valid)
test = (X_test, y_test)
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
                      num_filters=100,
                      filter_size=19,
                      W=GlorotUniform(),
                      nonlinearity=rectify)
network = MaxPool1DLayer(network, pool_size=3, stride=None)
get_output_shape(network)


"""
# 3rd convolution layer
network = Conv1DLayer(network,
                      num_filters=200,
                      filter_size=8,
                      W=GlorotUniform(),
                      nonlinearity=rectify)
network = MaxPool1DLayer(network, pool_size=4, stride=None)
"""
# Dense layer
network = DenseLayer(network,
                    num_units=100,
                    W=GlorotUniform(),
                    b=Constant(0.05))
network = DropoutLayer(network, p=0.5)
get_output_shape(network)

# output layer
network = DenseLayer(network,
                    num_units=num_labels,
                    W=GlorotUniform(),
                    b=Constant(0.05), 
                    nonlinearity=sigmoid)
get_output_shape(network)


# setup loss 
prediction = get_output(network, input_var)
prediction = T.clip(prediction, 1e-15, 1-1e-15)
loss =  T.mean(T.nnet.binary_crossentropy(prediction, target_var))
#T.mean(binary_crossentropy(prediction, target_var))

# calculate gradient with clipping
params = get_all_params(network, trainable=True)
updates = rmsprop(loss, params)

# test loss
test_prediction = get_output(network, deterministic=True)
#test_loss = T.mean(binary_crossentropy(test_prediction, target_var))
test_loss = T.mean(T.nnet.binary_crossentropy(test_prediction, target_var))

test_prediction = T.ge(test_prediction, .5)
test_accuracy = T.mean(T.eq(test_prediction, target_var))

# accuracy
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = T.mean(T.eq(prediction, target_var))

# build theano function
print "building model"
train_fun = theano.function([input_var, target_var], [loss, test_accuracy], updates=updates)
test_fun = theano.function([input_var, target_var], [test_loss, test_accuracy])


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
bar_length = 20
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
        
        # progress bar
        remaining_time = (time.time()-start_time)*(num_train_batches-index-1)/(index+1)
        percent = (index+1.)/num_train_batches
        progress = '#'*int(round(percent*bar_length))
        spaces = ' '*int(bar_length-round(percent*bar_length))
        sys.stdout.write("\rEpoch %d [%s] %.1f%% -- est.time=%ds -- loss=%.3f -- accuracy=%.2f%%" \
        %(epoch+1, progress+spaces, percent*100, remaining_time, train_loss/(index+1), train_accuracy/(index+1)*100))
        sys.stdout.flush()
    sys.stdout.write("\n")

    train_loss /= num_train_batches
    train_accuracy /= num_train_batches

    valid_loss = 0
    valid_accuracy = 0
    for _ in range(num_valid_batches):
        X, y = next(valid_batches)
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
num_test_batches = len(test[0]) // batch_size
test_batches = batch_generator(test[0], test[1], batch_size)
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





