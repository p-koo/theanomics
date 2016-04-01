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
from lasagne.nonlinearities import softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import binary_crossentropy
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint

np.random.seed(247) # for reproducibility

filename = 'Lasagne'
outdir = os.path.join('data',filename)
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print "making directory: " + outdir

def data_subset(y, class_range, negative=True):
    data_index = []
    for i in class_range:
        index = np.where(y[:, i] == 1)
        data_index = np.concatenate((data_index, index[0]), axis=0)
    unique_index = np.unique(data_index)
    num_data = y.shape[0]
    non_index = np.array(list(set(xrange(num_data)) - set(unique_index)))
    if all:
        index = np.concatenate((unique_index, non_index), axis=0)
    else:
        index = unique_index
    return index.astype(int)


# load data and munge 
num_labels = 100
num_include = 1000000
class_range = range(num_labels)

print "loading training data"
trainmat = h5py.File('train.mat')
y_train = np.array(trainmat['traindata']).T
y_train = y_train[:,class_range]
index = data_subset(y_train, class_range)
index = index[0:num_include]
y_train = y_train[index,:]
X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2,1,0))
X_train = X_train[index,:,:]
print X_train.shape

print "loading validation data"  
validmat = scipy.io.loadmat('valid.mat')
y_valid = validmat['validdata']
y_valid = y_valid[:, class_range]
index = data_subset(y_valid,class_range, negative=False)
y_valid = y_valid[index,:]
X_valid = np.transpose(validmat['validxdata'],axes=(0,1,2)) 
X_valid = X_valid[index,:,:]
print X_valid.shape


print "loading test data"
testmat = scipy.io.loadmat('test.mat')
y_test = testmat['testdata']
y_test = y_test[:, class_range]
index = data_subset(y_test,class_range, negative=False)
y_test = y_test[index,:]
X_test = np.transpose(testmat['testxdata'],axes=(0,1,2))
X_test = X_test[index,:,:]
print X_test.shape



#------------------------------------------------------------------------------
# setup model
#------------------------------------------------------------------------------

weight_norm = 7
learning_rate = 0.002
momentum=0.98


# setup variables
input_var = T.tensor3('inputs') # T.matrix()
target_var = T.ivector()

# Input layer
network = InputLayer(shape=(None, 4, 1000), input_var=input_var)

# 1st convolution layer
network = Conv1DLayer(network,
                      num_filters = 300,
                      filter_size = 19,
                      stride = 1,
                      pad = 0,
                      untie_biases=False,
                      W=GlorotUniform(),
                      b=None,
                      flip_filters=False)
network = BatchNormLayer(network)
network = ParametricRectifierLayer(network, alpha=Constant(0.1))
network = MaxPool1DLayer(network, pool_size = 3, stride=None, pad=0, ignore_border=True)

# 2nd convolution layer
network = Conv1DLayer(network,
                      num_filters = 200,
                      filter_size = 11,
                      stride = 1,
                      pad = 0,
                      untie_biases=False,
                      W=GlorotUniform(),
                      b=None,
                      flip_filters=False)
network = BatchNormLayer(network)
network = ParametricRectifierLayer(network, alpha=Constant(0.1))
network = MaxPool1DLayer(network, pool_size = 4, stride=None, pad=0, ignore_border=True)

# 3rd convolution layer
network = Conv1DLayer(network,
                      num_filters = 200,
                      filter_size = 7,
                      stride = 1,
                      pad = 0,
                      untie_biases=False,
                      W=GlorotUniform(),
                      b=None,
                      flip_filters=False)
network = BatchNormLayer(network)
network = ParametricRectifierLayer(network, alpha=Constant(0.1))
network = MaxPool1DLayer(network, pool_size = 4, stride=None, pad=0, ignore_border=True)

# Dense layer
network = DenseLayer(network,
                    num_units=1000,
                    W=GlorotUniform(),
                    b=Constant(0.))
network = BatchNormLayer(network)
network = ParametricRectifierLayer(network, alpha=Constant(0.1))
network = DropoutLayer(network, p=0.5)

# output layer
network = DenseLayer(network,
                    num_units=num_labels,
                    W=GlorotUniform(),
                    b=Constant(0.), 
                    nonlinearity=softmax)


# setup loss 
prediction = get_output(network)
loss = T.mean(binary_crossentropy(prediction, target_var))


# calculate gradient with clipping
weight_norm = 10
params = get_all_params(network, trainable=True)
grad = T.grad(loss, params)
grad = total_norm_constraint(grad, weight_norm)

# setup updates
updates = adagrad(grad, params)

# test loss
deterministic = True
test_prediction = get_output(network, trainable=True)
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

def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')


batch_size = 128
num_epochs = 100

num_train_batches = len(X_train) // batch_size
num_valid_batches = len(X_valid) // batch_size

train_batches = batch_gen(X_train, y_train, batch_size)
valid_batches = batch_gen(X_valid, y_valid, batch_size)


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
        print("  training accuracy:\t\t{:.2f} %".format(valid_accuracy/(index+1)*100))

    train_loss /= num_train_batches
    train_accuracy /= num_train_batches

    valid_loss = 0
    valid_accuracy = 0
    for _ in range(num_valid_batches):
        X, y = next(val_batches)
        loss, accuracy = test_fun(X, y)
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





