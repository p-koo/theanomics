
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


input_var = T.tensor4('input')

l_in = layers.InputLayer(shape=shape, input_var=input_var, name='input')
l_conv1 = layers.Conv2DLayer(l_in, num_filters=200, filter_size=(9,1), W=init.GlorotUniform(), 
                             nonlinearity=None, b=None, pad='valid', name='conv1')
l_norm1 = layers.BatchNormLayer(l_conv1)
l_bias1 = layers.BiasLayer(l_norm1, b=init.Constant(0.05))
l_nonlin1 = layers.NonlinearityLayer(l_bias1, nonlinearity=nonlinearities.rectify)
l_pool1 = layers.MaxPool2DLayer(l_nonlin1, pool_size=(4,1))

l_conv3 = layers.Conv2DLayer(l_pool1, num_filters=200, filter_size=(7,1), W=init.GlorotUniform(), 
                             nonlinearity=None, b=None, pad='valid', name='conv1')
l_norm3 = layers.BatchNormLayer(l_conv3)
l_bias3 = layers.BiasLayer(l_norm3, b=init.Constant(0.05))
l_nonlin3 = layers.NonlinearityLayer(l_bias3, nonlinearity=nonlinearities.rectify)
l_pool3 = layers.MaxPool2DLayer(l_nonlin3, pool_size=(4,1))
l_drop3 = layers.DropoutLayer(l_pool3, p=0.3)

l_dense3 = layers.DenseLayer(l_drop3, num_units=200, W=init.GlorotUniform(), name='inter')
l_norm4 = layers.BatchNormLayer(l_dense3)
l_bias3 = layers.BiasLayer(l_norm4, init.Constant(0.05))
l_nonlin3 = layers.NonlinearityLayer(l_bias3, nonlinearity=nonlinearities.sigmoid)
l_drop4 = layers.DropoutLayer(l_nonlin3, p=0.5)

l_dense4 = layers.DenseLayer(l_drop4, num_units=3, W=init.GlorotUniform(), name='inter')
l_bias4 = layers.BiasLayer(l_dense4, init.Constant(0.05))
l_nonlin4 = layers.NonlinearityLayer(l_bias4, nonlinearity=nonlinearities.sigmoid)

l_enc = layers.NonlinearityLayer(l_nonlin4, nonlinearity=None)

l_dec5 = layers.InverseLayer(l_enc, l_dense4, name='decode')
l_bias5 = layers.BiasLayer(l_dec5, init.Constant(0.05))
l_nonlin5 = layers.NonlinearityLayer(l_bias5, nonlinearity=nonlinearities.sigmoid)

l_dec6 = layers.InverseLayer(l_nonlin5, l_dense3, name='decode')
l_norm6 = layers.BatchNormLayer(l_dec6)
l_bias6 = layers.BiasLayer(l_norm6, init.Constant(0.05))
l_nonlin6 = layers.NonlinearityLayer(l_bias6, nonlinearity=nonlinearities.sigmoid)

l_depool9 = layers.InverseLayer(l_nonlin6, l_pool3, name='depool')
l_deconv9 = layers.InverseLayer(l_depool9, l_conv3, name='deconv')
l_norm9 = layers.BatchNormLayer(l_deconv9)
l_bias9 = layers.BiasLayer(l_norm9, init.Constant(0.05))
l_nonlin9 = layers.NonlinearityLayer(l_bias9, nonlinearity=nonlinearities.rectify)
                                     
#l_depool7 = layers.InverseLayer(l_nonlin9, l_pool2, name='depool')
#l_deconv7 = layers.InverseLayer(l_nonlin9, l_conv2, name='deconv')
#l_bias7 = layers.BiasLayer(l_deconv7, init.Constant(0.05))
#l_nonlin7 = layers.NonlinearityLayer(l_bias7, nonlinearity=nonlinearities.rectify)

l_depool8 = layers.InverseLayer(l_nonlin9, l_pool1, name='depool')
l_deconv8 = layers.InverseLayer(l_depool8, l_conv1, name='deconv')
l_norm8 = layers.BatchNormLayer(l_deconv8)
l_bias8 = layers.BiasLayer(l_norm8, init.Constant(0.05))
l_out = layers.NonlinearityLayer(l_bias8, nonlinearity=nonlinearities.rectify)

