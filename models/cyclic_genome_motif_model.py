
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

    
def cyclic_genome_motif_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    net['input'] = InputLayer(input_var=input_var, shape=shape)

    # stage 1 - 2 convolution layers   
    net['conv1'] = ConvLayer(net['input'], num_filters=200, filter_size=(9, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['active1'] = NonlinearityLayer(net['conv1'], leaky_rectify)
    net['conv2'] = ConvLayer(net['active1'], num_filters=200, filter_size=(9, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None,
                                           pad='same')
    net['batch2'] = BatchNormLayer(net['conv2'], epsilon=0.001)
    net['active2'] = NonlinearityLayer(net['batch2'], leaky_rectify)
    net['pool2'] = PoolLayer(net['active2'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    
    # stage 2 - 2 convolution layers initialized from stage 1 weights   
    net['conv3'] = ConvLayer(net['input'], num_filters=200, filter_size=(9, 1), stride=(1, 1),
                                           W=net['conv1'].W, b=net['conv1'].b, nonlinearity=None, 
                                           pad='same')
    net['active3'] = NonlinearityLayer(net['conv3'], leaky_rectify)
    net['conv4'] = ConvLayer(net['active3'], num_filters=200, filter_size=(9, 1), stride=(1, 1),
                                           W=net['conv2'].W, b=net['conv2'].b, nonlinearity=None,
                                           pad='same')
    net['batch4'] = BatchNormLayer(net['conv4'], epsilon=0.001)
    net['active4'] = NonlinearityLayer(net['batch4'], leaky_rectify)
    net['pool4'] = PoolLayer(net['active4'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
    

    # merge initial and layers
    net['concat'] = ConcatLayer([net['pool2'], net['pool4']])

    # stage 3  - deeper convolutions w/ max-pooling for wider spatial context
    net['conv5'] = ConvLayer(net['pool4'], num_filters=200, filter_size=(8, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch5'] = BatchNormLayer(net['conv5'], epsilon=0.001)
    net['active5'] = NonlinearityLayer(net['batch5'], leaky_rectify)
    net['pool5'] = PoolLayer(net['active5'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)


    # stage 4 - 2 dense layers for classification            
    net['dense6'] = DenseLayer(net['pool5'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
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

