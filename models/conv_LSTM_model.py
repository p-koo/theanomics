
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer, LSTMLayer, get_output_shape, LocalResponseNormalization2DLayer
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

    
def bidirectionalLSTM(l_in, num_units, grad_clipping):
    l_forward = LSTMLayer(l_in, num_units=num_units, grad_clipping=grad_clipping)
    l_backward = LSTMLayer(l_in, num_units=num_units, grad_clipping=grad_clipping, backwards=True)
    return ConcatLayer([l_forward, l_backward])



def inception_module(input_layer, num_filters, filter_size):
    net = {}
    net['filt6'] = ConvLayer(input_layer, num_filters=num_filters[0], filter_size=(filter_size[0], 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['norm6'] = LocalResponseNormalization2DLayer(net['filt6'], alpha=.001/9.0, k=1., beta=0.75, n=5)
    #net['norm1'] = BatchNormLayer(net['concat'], epsilon=0.001)
    net['active6'] = ParametricRectifierLayer(net['norm6'], alpha=Constant(0.25), shared_axes='auto')

    net['filt9'] = ConvLayer(input_layer, num_filters=num_filters[1], filter_size=(filter_size[1], 1), 
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['norm9'] = LocalResponseNormalization2DLayer(net['filt9'], alpha=.001/9.0, k=1., beta=0.75, n=5)
    #net['norm1'] = BatchNormLayer(net['concat'], epsilon=0.001)
    net['active9'] = ParametricRectifierLayer(net['norm9'], alpha=Constant(0.25), shared_axes='auto')
    
    net['filt12'] = ConvLayer(input_layer, num_filters=num_filters[2], filter_size=(filter_size[2], 1), 
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None, 
                                           pad='same')
    net['norm12'] = LocalResponseNormalization2DLayer(net['filt12'], alpha=.001/9.0, k=1., beta=0.75, n=5)
    #net['norm1'] = BatchNormLayer(net['concat'], epsilon=0.001)
    net['active12'] = ParametricRectifierLayer(net['norm12'], alpha=Constant(0.25), shared_axes='auto')
    
    net['output'] = ConcatLayer([net['active6'], net['active9'], net['active12']])
    return net['output']
    


def conv_LSTM_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    net['input'] = InputLayer(input_var=input_var, shape=shape)
    
    # stage 1 - 2 convolution layers   
    net['incept1'] = inception_module(net['input'], num_filters=[150, 100, 50], filter_size=[5, 9, 13])
    net['pool1'] = PoolLayer(net['incept1'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
    """
    # inception module on motifs
    net['incept2'] = inception_module(net['pool1'], num_filters=[150, 100, 50], filter_size=[5, 9, 13])
    net['pool2'] = PoolLayer(net['incept2'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
    """

    net['conv2'] = ConvLayer(net['pool1'], num_filters=300, filter_size=(8, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['norm2'] = LocalResponseNormalization2DLayer(net['conv2'], alpha=.001/9.0, k=1., beta=0.75, n=5)
    #net['norm1'] = BatchNormLayer(net['concat'], epsilon=0.001)
    net['active2'] = NonlinearityLayer(net['norm2'], leaky_rectify)
    net['pool2'] = PoolLayer(net['active2'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)


    net['lstm'] = bidirectionalLSTM(net['pool2'], num_units=200, grad_clipping=100)
    net['drop5'] = DropoutLayer(net['lstm'], p=0.3)

    # stage 4 - 2 dense layers for classification            
    net['dense6'] = DenseLayer(net['drop5'], num_units=300, W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch6'] = BatchNormLayer(net['dense6'], epsilon=0.001)
    net['active6'] = ParametricRectifierLayer(net['batch6'], alpha=Constant(0.25), shared_axes='auto')
    net['drop6'] = DropoutLayer(net['active6'], p=0.5)

    net['dense7'] = DenseLayer(net['drop6'], num_units=num_labels, W=GlorotUniform(), b=None, nonlinearity=None)
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

