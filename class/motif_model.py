#!/bin/python

import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params
from lasagne.nonlinearities import softmax,
from lasagne.init import GlorotUniform Constant
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers import get_output, get_all_params


from lasagne.objectives import binary_crossentropy, binary_crossentropy, squared_error
from lasagne.updates import sgd, nesterov_momentum, adagrad, rmsprop, adam, norm_constraint


def build_model(input_var=None):
    
    # Input layer
    network = InputLayer((None, 1000, 4), input_var=input_var)

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
    network = BatchNormLayer(network,
                            axis='auto',
                            epsilon=1e-4,
                            alpha=0.1,
                            beta=Constant(0),
                            gamma=Constant(1),
                            mean=Constant(0),
                            inv_std=Constant(1))
    network = ParametricRectifierLayer(network,
                                      alpha=Constant(0.25),
                                      shared_axes='auto')
    network = MaxPool1DLayer(network,
                             pool_size = 3,
                             stride=None,
                             pad=0,
                             ignore_border=True)

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
    network = BatchNormLayer(network,
                            axis='auto',
                            epsilon=1e-4,
                            alpha=0.1,
                            beta=Constant(0),
                            gamma=Constant(1),
                            mean=Constant(0),
                            inv_std=Constant(1))
    network = ParametricRectifierLayer(network,
                                      alpha=Constant(0.25),
                                      shared_axes='auto')
    network = MaxPool1DLayer(network,
                             pool_size = 4,
                             stride=None,
                             pad=0,
                             ignore_border=True)

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
    network = BatchNormLayer(network,
                            axis='auto',
                            epsilon=1e-4,
                            alpha=0.1,
                            beta=Constant(0),
                            gamma=Constant(1),
                            mean=Constant(0),
                            inv_std=Constant(1))
    network = ParametricRectifierLayer(network,
                                      alpha=Constant(0.25),
                                      shared_axes='auto')
    network = MaxPool1DLayer(network,
                             pool_size = 4,
                             stride=None,
                             pad=0,
                             ignore_border=True)

    # Dense layer
    network = DenseLayer(network,
                        num_units=1000,
                        W=GlorotUniform(),
                        b=Constant(0.))

    # Batch normalization layer
    network = BatchNormLayer(network,
                            axis='auto',
                            epsilon=1e-4,
                            alpha=0.1,
                            beta=Constant(0),
                            gamma=Constant(1),
                            mean=Constant(0),
                            inv_std=Constant(1))
    network = ParametricRectifierLayer(network,
                                      alpha=Constant(0.25),
                                      shared_axes='auto')
    DropoutLayer(network, p=0.5)

    # output layer
    network = DenseLayer(network,
                        num_units=919,
                        W=GlorotUniform(),
                        b=Constant(0.), 
                        nonlinearity=softmax)
    return network

