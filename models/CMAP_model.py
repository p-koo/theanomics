import sys
import theano.tensor as T
import numpy as np
from lasagne import layers, init, nonlinearities

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer, LocalResponseNormalization2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.nonlinearities import softmax, sigmoid, rectify, linear
from lasagne.nonlinearities import leaky_rectify, tanh, very_leaky_rectify

from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

from six.moves import cPickle
sys.path.append('..')


#---------------------------------------------------------------------------------------------------------------

def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 1
    net['corrupt1'] = layers.GaussianNoiseLayer(net['input'], sigma=0.1)
    net['encode1'] = layers.DenseLayer(net['corrupt1'], num_units=2000, W=init.GlorotUniform(), 
                                      b=init.Constant(.0), nonlinearity=None)
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1'], nonlinearity=nonlinearities.rectify)

    # encode layer 2
    net['corrupt2'] = layers.GaussianNoiseLayer(net['encode1_active'], sigma=0.1)
    net['encode2'] = layers.DenseLayer(net['corrupt2'], num_units=4000, W=init.GlorotUniform(), 
                                      b=init.Constant(.0), nonlinearity=None)
    net['encode2_active'] = layers.NonlinearityLayer(net['encode2'], nonlinearity=nonlinearities.rectify)

    # encode layer
    net['encode'] = layers.DenseLayer(net['encode2_active'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(.0), nonlinearity=None)
    net['encode_active'] = layers.NonlinearityLayer(net['encode'], nonlinearity=nonlinearities.linear)

    # decode layer
    net['decode'] = layers.DenseLayer(net['encode_active'], num_units=4000, W=net['encode'].W.T, 
                                      b=init.Constant(.0), nonlinearity=None)
    net['decode_active'] = layers.NonlinearityLayer(net['decode'], nonlinearity=nonlinearities.rectify)

    # decode layer 1
    net['decode1'] = layers.DenseLayer(net['decode_active'], num_units=2000, W=net['encode2'].W.T, 
                                      b=init.Constant(.0), nonlinearity=None)
    net['decode1_active'] = layers.NonlinearityLayer(net['decode1'], nonlinearity=nonlinearities.rectify)

    # decode layer 2
    net['decode2'] = layers.DenseLayer(net['decode1_active'], num_units=970, W=net['encode1'].W.T, 
                                      b=init.Constant(.0), nonlinearity=None)
    net['decode2_active'] = layers.NonlinearityLayer(net['decode2'], nonlinearity=nonlinearities.linear)
    net['output'] = net['decode2_active']


    # optimization parameters
    optimization = {"objective": "autoencoder",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization

#---------------------------------------------------------------------------------------------------------------

"""
def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

# create model
    inputs = {'layer': 'input',
              'input_var': input_var,
              'shape': (None, 970),
              'name': 'input'
              }
    dense1 = {'layer': 'dense', 
              'num_units': 2500, 
              'W': GlorotUniform(),
              'b': Constant(0.05),
              #'norm': 'batch', 
              'dropout': .5,
              'activation': 'prelu',
              'name': 'dense1'
              }
    dense2 = {'layer': 'dense', 
              'num_units': 5000, 
              'W': GlorotUniform(),
              'b': Constant(0.05),
              #'norm': 'batch', 
              'dropout': .5,
              'activation': 'prelu',
              'name': 'dense1'
              }
    output = {'layer': 'dense', 
              'num_units': 11350, 
              'W': GlorotUniform(),
              'b': Constant(0.05),
              'activation': 'sigmoid',
              'name': 'dense2'
              }
          
    model_layers = [inputs, dense1, output]
    net = build_network(model_layers)

    # optimization parameters
    optimization = {"objective": "ols",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization
"""
#---------------------------------------------------------------------------------------------------------------

# Mean RIS: 1188564.27

"""
def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = InputLayer(shape=(None, 970), input_var=input_var)
    net['dense1'] = DenseLayer(net['input'], num_units=11350, W=GlorotUniform(), 
                                      b=Constant(.0), nonlinearity=None)
    net['dense1_active'] = NonlinearityLayer(net['dense1'], nonlinearity=linear)
    net['output'] = net['dense1_active']

    # optimization parameters
    optimization = {"objective": "ols",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
                    "epsilon": 1e-8,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization

"""

