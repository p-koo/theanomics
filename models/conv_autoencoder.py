#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def conv_autoencoder(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = input_var

	# create model
	layer1 = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			 }
	layer2 = {'layer': 'convolution', 
			  'num_filters': 2, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (8, 1),
			  'name': 'conv1'
			  }

	model_layers = [layer1, layer2]
	network = build_network(model_layers, autoencode=1)

	# optimization parameters
	optimization = {"objective": "autoencoder",
				 "optimizer": "adam"
#                "learning_rate": 0.1,
#                "momentum": 0.9, 
 #               "weight_norm": 10,
#                "l1": 1e-7,
 #               "l2": 1e-8,
				}
				
	return network, input_var, target_var, optimization
