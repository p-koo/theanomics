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
from lasagne.objectives import binary_crossentropy, categorical_crossentropy, 
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint, sgd, adam
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant


