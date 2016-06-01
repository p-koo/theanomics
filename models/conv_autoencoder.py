#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network
from lasagne.layers import Conv2DLayer#, TransposedConv2DLayer


def conv_autoencoder(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = input_var
	"""

	net['conv1'] = Conv2DLayer(shape, num_filters=20, filter_size=(12,1), stride=1, pad=0)
	net['output'] = TransposedConv2DLayer(net['conv1'], net['conv1'].input_shape[1],
	                                net['conv1'].filter_size, stride=net['conv1'].stride, crop=net['conv1'].pad,
	                                W=net['conv1'].W, flip_filters=not net['conv1'].flip_filters)
	"""
	# optimization parameters
	optimization = {"objective": "mse",
				 "optimizer": "adam"
#                "learning_rate": 0.1,
#                "momentum": 0.9, 
 #               "weight_norm": 10,
#                "l1": 1e-7,
 #               "l2": 1e-8,
				}
				
	return network, input_var, target_var, optimization
