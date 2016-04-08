
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, sigmoid

def binary_genome_motif_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    net['input'] = InputLayer(input_var=input_var, shape=shape)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=200,
                             filter_size=(12, 1),
                             stride=(1, 1)
                             flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1'],
                             pool_size=(4, 1),
                             stride=(4, 1)
                             ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'],
                             num_filters=200,
                             filter_size=(8, 1),
                             flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=(4,1),
                             stride=(4,1),
                             ignore_border=False)
    net['fc4'] = DenseLayer(net['pool3'], num_units=200)
    net['drop4'] = DropoutLayer(net['fc4'], p=0.5)
    net['fc5'] = DenseLayer(net['drop4'], num_units=num_labels, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc5'], sigmoid)
    
    optimization = {"objective": "binary",
                    "optimizer": "adam",
     #               "learning_rate": 0.1,
    #                "momentum": 0.9
                    "weight_norm": 10
                    #"l1": 1e-7,
                    #"l2": 1e-8
                    }

    return network, input_var, target_var, optimization

