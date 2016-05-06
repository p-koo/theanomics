
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

    
filter_size = 8
pool_size = 6
num_filters = 100

def simple_exon_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}

    #---------------------------------------------------------------------------------------------------------------

    # region 1: upstream exon region
    net['input1'] = InputLayer(input_var=input_var, shape=shape)
    net['conv1'] = ConvLayer(net['input1'], num_filters=num_filters, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=Constant(0.0),  pad='valid')
    net['batch1'] = BatchNormLayer(net['conv1'], epsilon=0.001)    
    net['active1'] = ParametricRectifierLayer(net['batch1'], alpha=Constant(0.25), shared_axes='auto')
    net['pool1'] = PoolLayer(net['active1'], pool_size=(pool_size, 1), stride=(pool_size, 1), ignore_border=False)

	# region 2: upstream flank region
    net['input2'] = InputLayer(input_var=input_var, shape=shape)
    net['conv2'] = ConvLayer(net['input2'], num_filters=num_filters, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=Constant(0.0), pad='valid')
    net['batch2'] = BatchNormLayer(net['conv2'], epsilon=0.001)    
    net['active2'] = ParametricRectifierLayer(net['batch2'], alpha=Constant(0.25), shared_axes='auto')
    net['pool2'] = PoolLayer(net['active2'], pool_size=(pool_size, 1), stride=(pool_size, 1), ignore_border=False)


    # region 3: downstream flank region
    net['input3'] = InputLayer(input_var=input_var, shape=shape)
    net['conv3'] = ConvLayer(net['input3'], num_filters=num_filters, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=Constant(0.0), pad='valid')
    net['batch3'] = BatchNormLayer(net['conv3'], epsilon=0.001)    
    net['active3'] = ParametricRectifierLayer(net['batch3'], alpha=Constant(0.25), shared_axes='auto')
    net['pool3'] = PoolLayer(net['active3'], pool_size=(pool_size, 1), stride=(pool_size, 1), ignore_border=False)

    # region 4: downstream exon region
    net['input4'] = InputLayer(input_var=input_var, shape=shape)
    net['conv4'] = ConvLayer(net['input4'], num_filters=num_filters, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=Constant(0.0), pad='valid')
    net['batch4'] = BatchNormLayer(net['conv4'], epsilon=0.001)    
    net['active4'] = ParametricRectifierLayer(net['batch4'], alpha=Constant(0.25), shared_axes='auto')
    net['pool4'] = PoolLayer(net['active4'], pool_size=(pool_size, 1), stride=(pool_size, 1), ignore_border=False)

    #---------------------------------------------------------------------------------------------------------------
    # combine to a single layer

    net['concat'] = ConcatLayer([net['pool1'], net['pool2'], net['pool3'], net['pool4']])

    # also try to merge conv layers first and then normalize/activate/pool

    #---------------------------------------------------------------------------------------------------------------

    filter_size = 6
    pool_size = 3
    # 2nd conv layer

    net['conv5'] = ConvLayer(net['concat'], num_filters=256, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch5'] = BatchNormLayer(net['conv5'], epsilon=0.001)
    net['active5'] = ParametricRectifierLayer(net['batch5'], alpha=Constant(0.25), shared_axes='auto')
    net['pool5'] = PoolLayer(net['active5'], pool_size=(pool_size, 1), stride=(pool_size, 1), ignore_border=False)

    # 3rd conv layer
    net['conv6'] = ConvLayer(net['pool5'], num_filters=512, filter_size=(filter_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch6'] = BatchNormLayer(net['conv6'], epsilon=0.001)
    net['active6'] = ParametricRectifierLayer(net['batch6'], alpha=Constant(0.25), shared_axes='auto')
    net['pool6'] = PoolLayer(net['active6'], pool_size=(pool_size, 1), stride=(pool_size 1), ignore_border=False)

    # 1st dense layer  
    net['dense7'] = DenseLayer(net['pool7'], num_units=512, W=GlorotUniform(), b=Constant(0.0), nonlinearity=None)
    net['batch7'] = BatchNormLayer(net['dense7'], epsilon=0.001)
    net['active7'] = ParametricRectifierLayer(net['batch7'], alpha=Constant(0.25), shared_axes='auto')

    # 2nd dense layer
    net['dense8'] = DenseLayer(net['active7'], num_units=512, W=GlorotUniform(), b=Constant(0.0), nonlinearity=None)
    net['batch8'] = BatchNormLayer(net['dense8'], epsilon=0.001)
    net['active8'] = NonlinearityLayer(net['batch8'], sigmoid)

    # output layer
	net['dense9'] = DenseLayer(net['active8'], num_units=num_labels, W=GlorotUniform(), b=Constant(0.0), nonlinearity=None)
    net['batch9'] = BatchNormLayer(net['dense9'], epsilon=0.001)
    net['output'] = NonlinearityLayer(net['batch9'], sigmoid)


    optimization = {"objective": "binary",
                    "optimizer": "adam"
#                   "learning_rate": 0.1,
#                   "momentum": 0.9
#                   "weight_norm": 10
                    #"l1": 1e-7,
                    #"l2": 1e-8
                    }
    return net, input_var, target_var, optimization

