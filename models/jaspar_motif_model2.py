import sys
import theano.tensor as T
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.nonlinearities import softmax, sigmoid, rectify, linear
from lasagne.nonlinearities import leaky_rectify, tanh, very_leaky_rectify

from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform

from six.moves import cPickle
sys.path.append('..')
#from src import load_JASPAR_motifs

# Examples:
#   net['conv1'] = ConvLayer(net['input'], num_filters=200, filter_size=(12, 1), stride=(1, 1),
#                                          W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
#   net['pool'] = PoolLayer(net['something'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
#   net['batch'] = BatchNormLayer(net['something'])
#   net['active'] = NonlinearityLayer(net['something'], sigmoid)
#   net['dense'] = DenseLayer(net['something'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
#   net['drop4'] = DropoutLayer(net['something'], p=0.5)
#   net['prelu'] = ParametricRectifierLayer(net['something'], alpha=Constant(0.25), shared_axes='auto')

def load_JASPAR_motifs(jaspar_path, MAX):

    with open(jaspar_path, 'rb') as f: 
        jaspar_motifs = cPickle.load(f)

    motifs = []
    for jaspar in jaspar_motifs:
        length = len(jaspar)
        if length < MAX:
            offset = MAX - length
            firstpart = offset // 2
            secondpart = offset - firstpart
            matrix = np.vstack([np.ones((firstpart,4))*.25, jaspar,  np.ones((secondpart,4))*.25])
            motifs.append(matrix)

        elif length > MAX:
            offset = length - MAX
            firstpart = offset // 2
            secondpart = offset - firstpart
            matrix = jaspar[0+firstpart:length-secondpart,:]
            motifs.append(matrix)

        elif length == MAX:
            motifs.append(jaspar)

    motifs = np.array(motifs)
    motifs = np.expand_dims(np.transpose(motifs, (0,2,1)), axis=3)

    return motifs
    
def jaspar_motif_model2(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    net['input'] = InputLayer(input_var=input_var, shape=shape)

    # load JASPAR motifs
    motif_size = 12
    motifs = load_JASPAR_motifs('/home/peter/Code/DeepMotifs/models/JASPAR_motifs.cpickle', motif_size)
    num_motifs = len(motifs)

    # initialize weights with JASPAR motifs
    net['conv1_1'] = ConvLayer(net['input'], num_filters=num_motifs, filter_size=(motif_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['conv1_2'] = ConvLayer(net['input'], num_filters=50, filter_size=(motif_size, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)

    # merge convolution motifs
    net['concat'] = ConcatLayer([net['conv1_1'], net['conv1_2']])
    net['batch1'] = BatchNormLayer(net['concat'], epsilon=0.001)
    net['active1'] = ParametricRectifierLayer(net['batch1'], alpha=Constant(0.25), shared_axes='auto')
    #net['pool1'] = PoolLayer(net['active1'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    net['conv2'] = ConvLayer(net['active1'], num_filters=200, filter_size=(8, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch2'] = BatchNormLayer(net['conv2'], epsilon=0.001)
    net['active2'] = ParametricRectifierLayer(net['batch2'], alpha=Constant(0.25), shared_axes='auto')
    net['pool2'] = PoolLayer(net['active2'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    net['conv3'] = ConvLayer(net['pool2'], num_filters=200, filter_size=(8, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch3'] = BatchNormLayer(net['conv3'], epsilon=0.001)
    net['active3'] = ParametricRectifierLayer(net['batch3'], alpha=Constant(0.25), shared_axes='auto')
    #net['pool3'] = PoolLayer(net['active3'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    net['conv4'] = ConvLayer(net['active3'], num_filters=200, filter_size=(8, 1), stride=(1, 1),
                                           W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
    net['batch4'] = BatchNormLayer(net['conv4'], epsilon=0.001)
    net['active4'] = ParametricRectifierLayer(net['batch4'], alpha=Constant(0.25), shared_axes='auto')
    net['pool4'] = PoolLayer(net['active4'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)

    net['dense4'] = DenseLayer(net['pool4'], num_units=500, W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch4'] = BatchNormLayer(net['dense4'], epsilon=0.001)
    net['active4'] = ParametricRectifierLayer(net['batch4'], alpha=Constant(0.25), shared_axes='auto')

    net['dense5'] = DenseLayer(net['active4'], num_units=num_labels, W=GlorotUniform(), b=None, nonlinearity=None)
    net['batch5'] = BatchNormLayer(net['dense5'], epsilon=0.001)
    net['output'] = NonlinearityLayer(net['batch5'], sigmoid)


    
    optimization = {"objective": "binary",
                    "optimizer": "adam"
#                   "learning_rate": 0.1,
#                   "momentum": 0.9
#                   "weight_norm": 10
                    #"l1": 1e-7,
                    #"l2": 1e-8
                    }
    return net, input_var, target_var, optimization

