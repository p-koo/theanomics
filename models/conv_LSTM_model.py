#/bin/python
import sys
sys.path.append('..')
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from lasagne import layers, nonlinearities, init
from build_network import build_network

def model(shape, num_labels):
	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	input_layer = {'layer': 'input',
				   'input_var': input_var,
				   'shape': shape,
				   'name': 'input'
				   }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (7, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv2'
			  }

	conv3 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv3'
			  }

	lstm = {'layer': 'lstm', 
			  'num_units': 25, 
			  'grad_clipping': 50,
			  'name': 'lstm'
			  }

	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': None,
			  'activation': 'sigmoid', 
			  'name': 'dense'
			  }
			  
	model_layers = [input_layer, conv1, conv2, conv3, lstm, output] 
	network = build_network(model_layers)


	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                    
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-8,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
					"l1": 1e-4,
					"l2": 1e-5
					}

	return network, input_var, target_var, optimization


