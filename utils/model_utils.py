#!/bin/python

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
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint, sgd, adam
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant

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



def build_cost(network, target_var, objective, deterministic=False):

    prediction = get_output(network, deterministic=deterministic)
    if objective == 'categorical':
        cost = categorical_crossentropy(prediction, target_var)
    elif objective == 'binary':
        cost = binary_crossentropy(prediction, target_var)
        
    cost = cost.mean()
    return cost, prediction


def calculate_gradient(network, cost, params, weight_norm):

    # calculate gradients
    grad = T.grad(cost, params)

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




