#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def deepsea_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	layer1 = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	layer2 = {'layer': 'convolution', 
			  'num_filters': 100,  #240
			  'filter_size': (19, 1),
			  'pool_size': (3, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'pad': 'same',
			  'norm': 'batch', 
			  'activation': 'relu',
			  'name': 'conv1'
			  }
	layer3 = {'layer': 'convolution', 
			  'num_filters': 200,  #480
			  'filter_size': (9, 1),
			  'pool_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'pad': 'same',
			  'norm': 'batch', 
			  'activation': 'relu',
			  'name': 'conv2'
			  }
	layer4 = {'layer': 'convolution', 
			  'num_filters': 300,  #960
			  'filter_size': (7, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'pad': 'same',
			  'pool_size': (3,1),
			  'norm': 'batch', 
			  'activation': 'relu',
			  'name': 'conv3'
			  }
	layer5 = {'layer': 'dense', 
			  'num_units': 1000, 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'dropout': .5,
			  'activation': 'relu',
			  'name': 'dense1'
			  }
	layer6 = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'activation': 'sigmoid',
			  'name': 'dense2'
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4, layer5, layer6]
	network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                 
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-6,
					"l1": 1e-7,
					"l2": 1e-8, 
					}


	return network, input_var, target_var, optimization

