
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.nonlinearities import softmax, sigmoid, rectify, linear
from lasagne.nonlinearities import leaky_rectify, tanh, very_leaky_rectify

from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform

# Examples:
#   net['conv1'] = ConvLayer(net['input'], num_filters=200, filter_size=(12, 1), stride=(1, 1),
#                                          W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
#   net['pool'] = PoolLayer(net['something'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
#   net['batch'] = BatchNormLayer(net['something'])
#   net['active'] = NonlinearityLayer(net['something'], sigmoid)
#   net['dense'] = DenseLayer(net['something'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
#   net['drop4'] = DropoutLayer(net['something'], p=0.5)
#   net['prelu'] = ParametricRectifierLayer(net['something'], alpha=Constant(0.25), shared_axes='auto')


import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.nonlinearities import softmax, sigmoid, rectify, linear
from lasagne.nonlinearities import leaky_rectify, tanh, very_leaky_rectify

from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform

# Examples:
#   net['conv1'] = ConvLayer(net['input'], num_filters=200, filter_size=(12, 1), stride=(1, 1),
#                                          W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
#   net['pool'] = PoolLayer(net['something'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
#   net['batch'] = BatchNormLayer(net['something'])
#   net['active'] = NonlinearityLayer(net['something'], sigmoid)
#   net['dense'] = DenseLayer(net['something'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
#   net['drop4'] = DropoutLayer(net['something'], p=0.5)
#   net['prelu'] = ParametricRectifierLayer(net['something'], alpha=Constant(0.25), shared_axes='auto')


def inception_module(input_layer, num_filters, filter_size):
    net = {}
    net['filt6'] = ConvLayer(input_layer, num_filters=num_filters[0], filter_size=(filter_size[0], 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['batch6'] = BatchNormLayer(net['filt6'], epsilon=0.001)
    net['active6'] = ParametricRectifierLayer(net['batch6'], alpha=Constant(0.25), shared_axes='auto')

    net['filt9'] = ConvLayer(input_layer, num_filters=num_filters[1], filter_size=(filter_size[1], 1), 
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['batch9'] = BatchNormLayer(net['filt9'], epsilon=0.001)
    net['active9'] = ParametricRectifierLayer(net['batch9'], alpha=Constant(0.25), shared_axes='auto')

    net['filt12'] = ConvLayer(input_layer, num_filters=num_filters[2], filter_size=(filter_size[2], 1), 
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['batch12'] = BatchNormLayer(net['filt12'], epsilon=0.001)
    net['active12'] = ParametricRectifierLayer(net['batch12'], alpha=Constant(0.25), shared_axes='auto')


    # recurrent layer
    net['filt6_2'] = ConvLayer(input_layer, num_filters=num_filters[0], filter_size=(filter_size[0], 1),
                                           W=net['filt6'].W, b=net['filt6'].b, nonlinearity=None, 
                                           pad='same')
    net['batch6_2'] = BatchNormLayer(net['filt6_2'], epsilon=0.001)
    net['active6_2'] = ParametricRectifierLayer(net['batch6_2'], alpha=Constant(0.25), shared_axes='auto')

    net['filt9_2'] = ConvLayer(input_layer, num_filters=num_filters[1], filter_size=(filter_size[1], 1), 
                                           W=net['filt9'].W, b=net['filt9'].b, nonlinearity=None, 
                                           pad='same')
    net['batch9_2'] = BatchNormLayer(net['filt9_2'], epsilon=0.001)
    net['active9_2'] = ParametricRectifierLayer(net['batch9_2'], alpha=Constant(0.25), shared_axes='auto')

    net['filt12_2'] = ConvLayer(input_layer, num_filters=num_filters[2], filter_size=(filter_size[2], 1), 
                                           W=net['filt12'].W, b=net['filt12'].b, nonlinearity=None, 
                                           pad='same')
    net['batch12_2'] = BatchNormLayer(net['filt12_2'], epsilon=0.001)
    net['active12_2'] = ParametricRectifierLayer(net['batch12_2'], alpha=Constant(0.25), shared_axes='auto')



    net['output'] = ConcatLayer((net['active6'], net['active9']))#, net['active12']])



    return net['output']
    

def recurrent_inception_motif_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    net['input'] = InputLayer(input_var=input_var, shape=shape)

    # inception module on genome
    net['incept1'] = inception_module(net['input'], num_filters=[150, 100, 50], filter_size=[5, 9, 13])
    net['pool1'] = PoolLayer(net['incept1'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    # inception module on motifs
    net['incept2'] = inception_module(net['pool1'], num_filters=[150, 100, 50], filter_size=[5, 9, 13])
    net['pool2'] = PoolLayer(net['incept2'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    # 
    net['conv3'] = ConvLayer(net['pool2'], num_filters=200, filter_size=(8, 1), 
                                           W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch3'] = BatchNormLayer(net['conv3'])
    net['active3'] = ParametricRectifierLayer(net['batch3'], alpha=Constant(0.25), shared_axes='auto')
    net['pool3'] = PoolLayer(net['active3'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)


    net['dense6'] = DenseLayer(net['pool3'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch6'] = BatchNormLayer(net['dense6'], epsilon=0.001)
    net['active6'] = ParametricRectifierLayer(net['batch6'], alpha=Constant(0.25), shared_axes='auto')

    net['dense7'] = DenseLayer(net['active6'], num_units=num_labels, W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch7'] = BatchNormLayer(net['dense7'], epsilon=0.001)
    net['output'] = NonlinearityLayer(net['batch7'], sigmoid)
    
    optimization = {"objective": "binary",
                    "optimizer": "adam"
#                   "learning_rate": 0.1,
#                   "momentum": 0.9
#                   "weight_norm": 10
                    #"l1": 1e-7,
                    #"l2": 1e-8
                    }
    return net, input_var, target_var, optimization

