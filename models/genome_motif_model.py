#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def genome_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	input_layer = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 32, 
			  'filter_size': (5, 1), # 194 , 
			  'W': GlorotUniform(),
			  'b': None, # Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (2, 1), # 97, 148 
			  'pad': 'valid',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (5, 1),  #93
			  'W': GlorotUniform(),
			  'b': None, # Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),  # 31, 72
			  'pad': 'valid',
			  'dropout': 0.2,
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (5, 1), # 27
			  'W': GlorotUniform(),
			  'b': None, # Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),  # 9, 34
			  'pad': 'valid',
			  'dropout': 0.2,
			  'name': 'conv3'
			  }
	conv4 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (5, 1),  #8
			  'W': GlorotUniform(),
			  'b': None, # Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (2, 1), # 4, 
			  'pad': 'valid',
			  'dropout': 0.2,
			  'name': 'conv4'
			  }
	conv5 = {'layer': 'convolution', 
			  'num_filters': 1028, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None, # Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pad': 'valid',
			  'dropout': 0.,
			  'name': 'conv5'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'sigmoid',
			  'name': 'dense'
			  }

	model_layers = [input_layer, conv1, conv2, conv3, conv4,  conv5, output]
	network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                 
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-6,
					"l1": 1e-5,
					"l2": 1e-6
					}

	return network, input_var, target_var, optimization

"""

	input_layer = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (7, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (7, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (2, 1),
			  'name': 'conv2'
			  }
	conv3= {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv3'
			  }
	conv4 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (2, 1),
			  'name': 'conv4'
			  }
	conv5 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv5'
			  }
	conv6 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (2, 1),
			  'name': 'conv6'
			  }
	conv7 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv7'
			  }
	conv8 = {'layer': 'convolution', 
			  'num_filters': 512, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1), #5
			  'name': 'conv8'
			  }
	conv9 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv9'
			  }
	conv10 = {'layer': 'convolution', 
			  'num_filters': 768, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1), #5
			  'name': 'conv10'
			  }
	conv11 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv11'
			  }
	conv12 = {'layer': 'convolution', 
			  'num_filters': 1028, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (5, 1),
			  'name': 'conv12'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'activation': 'sigmoid',
			  'name': 'dense'
			  }

	model_layers = [input_layer, conv1, conv2, conv3, conv4, conv5, conv6, 
					conv7, conv8, conv9, conv10, conv11, conv12, output]
	network = build_network(model_layers)
"""