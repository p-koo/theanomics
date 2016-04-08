
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import NonlinearityLayer, InverseLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, sigmoid

def cae_genome_motif_model(shape, num_labels):

    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')

    net = {}
    input_1 = InputLayer(input_var=input_var, shape=shape)
    conv_1 = ConvLayer(input_1,
                     num_filters=200,
                     filter_size=(12, 1),
                     stride=(1, 1)
                     flip_filters=False)
    pool_1 = PoolLayer(conv_1,
                     pool_size=(4, 1),
                     stride=(4, 1)
                     ignore_border=False)
    conv_2 = ConvLayer(pool_1,
                     num_filters=200,
                     filter_size=(8, 1),
                     flip_filters=False)
    pool_2 = PoolLayer(conv_2,
                     pool_size=(4,1),
                     stride=(4,1),
                     ignore_border=False)
    fc_3 = DenseLayer(pool_2, num_units=200)
    drop_3 = DropoutLayer(fc_3, p=0.5)
    fc_4 = DenseLayer(drop_3, num_units=num_labels, nonlinearity=None)
    output = NonlinearityLayer(fc_4, sigmoid)
    fc_3_inv = InverseLayer(output, fc_3)
    fc_3_inv = InverseLayer(output, fc_3)
    pool2_inv = InverseLayer(fc_3_inv, pool_2)
    conv_2_inv = InverseLayer(pool2_inv, conv_2)
    pool_1_inv = InverseLayer(conv_2_inv, pool_1)
    conv_1_inv = InverseLayer(pool_1_inv, conv_1)
    conv_2_inv = InverseLayer(pool2_inv, conv_2)

    optimization = {"objective": "mse",
                    "optimizer": "adam",
                    "weight_norm": 10
                    }

    return network, input_var, target_var, optimization

