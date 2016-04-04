
import sys
import os
import time
import numpy as np
import h5py

import scipy.io
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify, softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint, sgd, adam

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



def base_layer(layer, network=[]):

    # input layer
    if layer['layer'] == 'input':
        network = InputLayer(layer['shape'], input_var=layer['input_var'])

    # dense layer
    elif layer['layer'] == 'dense':
        network = DenseLayer(network,
                            num_units=layer['num_units'],
                            W=layer['W'],
                            b=layer['b'])

    # convolution layer
    elif layer['layer'] == 'convolution':
        network = Conv2DLayer(network,
                              num_filters = layer['num_filters'],
                              filter_size = layer['filter_size'],
                              W=layer['W'],
                              b=layer['b'])

    return network


def build_model(layers, input_var):

    # loop to build each layer of network
    network = []
    for layer in layers:

        # create base layer
        network = base_layer(layer, network)
                
        # add Batch normalization layer
        if 'norm' in layer:
            if layer['norm'] == 'batch':
                network = BatchNormLayer(network)

        # add activation layer
        if 'activation' in layer:
            if layer['activation'] == 'prelu':
                network = ParametricRectifierLayer(network,
                                                  alpha=Constant(0.25),
                                                  shared_axes='auto')
            else:
                network = NonlinearityLayer(network, nonlinearity=layer['activation'])

        # add dropout layer
        if 'dropout' in layer:
            DropoutLayer(network, p=layer['dropout'])

        # add max-pooling layer
        if layer['layer'] == 'convolution':            
            network = MaxPool2DLayer(network, pool_size=layer['pool_size'])

    return network


def genome_motif_simple_model(shape=(None, 4, 200, 1)):

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # create model
    layer1 = {'layer': 'input', 
              'input_var': input_var,
              'shape': shape }
    layer2 = {'layer': 'convolution', 
              'num_filters': 200, 
              'filter_size': (8, 1),
              'W': GlorotUniform(),
              'b': None,
              #'dropout': .5,
              'norm': 'batch', 
              'activation': 'prelu',
              'pool_size': (4, 1)}
    layer3 = {'layer': 'convolution', 
              'num_filters': 200, 
              'filter_size': (8, 1),
              'W': GlorotUniform(),
              'b': None,
              #'dropout': .5,
              'norm': 'batch', 
              'activation': 'prelu',
              'pool_size': (4, 1)}
    layer4 = {'layer': 'dense', 
              'num_units': 200, 
              'default': True,
              'W': GlorotUniform(),
              'b': Constant(0.05), 
              'dropout': .5,
              'norm': 'batch',
              'activation': 'prelu'}
    layer5 = {'layer': 'dense', 
              'num_units': num_labels, 
              'default': True,
              'W': GlorotUniform(),
              'b': Constant(0.05),
              'noise': 'dropout',
              'activation': softmax}

    layers = [layer1, layer2, layer3, layer4, layer5]
    return layers, input_var, target_var



def build_cost(network, target_var, objective, deterministic=False):

    prediction = get_output(network, deterministic=deterministic)
    if objective == 'categorical':
        loss = categorical_crossentropy(prediction, target_var)
    elif objective == 'binary':
        loss = binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    return loss, prediction


def calculate_gradient(network, loss, params, weight_norm):

    # calculate gradients
    grad = T.grad(loss, params)

    # gradient clipping option
    if weight_norm > 0:
        grad = total_norm_constraint(grad, weight_norm)

    return grad



def optimizer(grad, params, update_params):

    if update_params['optimizer'] == 'sgd':
        updates = sgd(grad, params, learning_rate=update_params['learning_rate']) 
 
    elif update_params['optimizer'] == 'nesterov_momentum':
        updates = nesterov_momentum(grad, params, 
                                    learning_rate=update_params['learning_rate'], 
                                    momentum=update_params['momentum'])
    
    elif update_params['optimizer'] == 'adagrad':
        if "learning_rate" in update_params:
            updates = adagrad(grad, params, 
                              learning_rate=update_params['learning_rate'], 
                              epsilon=update_params['epsilon'])
        else:
            updates = adagrad(grad, params)

    elif update_params['optimizer'] == 'rmsprop':
        if "learning_rate" in update_params:
            updates = rmsprop(grad, params, 
                              learning_rate=update_params['learning_rate'], 
                              rho=update_params['rho'], 
                              epsilon=update_params['epsilon'])
        else:
            updates = rmsprop(grad, params)
    
    elif update_params['optimizer'] == 'adam':
        if "learning_rate" in update_params:
            updates = adam(grad, params, 
                            learning_rate=update_params['learning_rate'], 
                            beta1=update_params['beta1'], 
                            beta2=update_params['beta2'], 
                            epsilon=update['epsilon'])
        else:
            updates = adam(grad, params)
  
    return updates



def  prediction_accuracy(prediction, target_var, objective):

    if objective == "categorical":
        accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                                                  dtype=theano.config.floatX)
    elif objective == "binary":
        accuracy = T.mean(T.eq(prediction, target_var))
    elif objective == "meansquare":
        print "work in progress"
    return accuracy


#---------------------------------------

optimization = {"objective": "categorical",
                "optimizer": "sgd", 
                "learning_rate": 0.1}


print "building theano model function"

layers, input_var, target_var = genome_motif_simple_model(shape=(None,  dim, sequence_length, 1))
network = build_model(layers, input_var)

cost, prediction = build_cost(network, target_var, objective=optimization["objective"])

params = get_all_params(network, trainable=True)    
grad = calculate_gradient(network, cost, params, weight_norm=10)

update_params = {"optimizer": 'sgd', "learning_rate": 0.1, "momentum": 0.9 }
updates = optimizer(grad, params, optimization)

test_cost, test_prediction = build_cost(network, target_var, objective, deterministic=True)
test_accuracy = prediction_accuracy(test_prediction, target_var, objective)

train_fun = theano.function([input_var, target_var], [cost, test_accuracy], updates=updates)
test_fun = theano.function([input_var, target_var], [test_cost, test_accuracy])


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
    
    train_cost, train_accuracy = epoch_train(train_fun, train_batches, num_train_batches, 1)

    valid_cost, valid_accuracy = epoch_train(test_fun, valid_batches, num_train_batches)
    print_progress(valid_cost, valid_accuracy, "cross-validation", epoch, num_epochs)    

    # store training performance info
    valid_memory.append(valid_cost)
    performance.append([train_cost, train_accuracy, valid_cost, valid_accuracy])

    # check for early stopping
    status = early_stopping(valid_memory, patience)
    if not status:
        print "Patience ran out... Early stopping."
        break
            

# get test loss and accuracy
num_test_batches = len(test[0]) // batch_size
test_batches = batch_generator(test[0], test[1], batch_size)
test_cost, test_accuracy = epoch_train(test_fun, test_batches, num_test_batches)
print_progress(test_cost, test_accuracy, "test")    

