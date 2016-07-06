#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def test_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	input_layer = {'layer': 'input',
				   'input_var': input_var,
				   'shape': shape,
				   'name': 'input'
				   }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 32, #30
			  'filter_size': (11, 1), #189
			  'W': GlorotUniform(),
			  'b': None,
			  'pad': 'same',
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (13, 1), #20
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 50,  #14
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (4, 1), #8
			  'pad': 'same',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 128,  #30
			  'filter_size': (15, 1),
			  'W': GlorotUniform(),
			  'b': None,
#			  'norm': 'batch', 
			  'activation': 'relu',
			  'pad': 'valid',
			  'name': 'conv3'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(.05),
			  'norm': 'batch', 
			  'activation': 'sigmoid', 
			  'name': 'dense2'
			  }
			  
	model_layers = [input_layer,output] 
	network = build_network(model_layers)



	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,	                
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-8,
#	                "weight_norm": 7, 
#	                "momentum": 0.9
					"l1": 1e-5,
					"l2": 1e-6
					}

	return network, input_var, target_var, optimization

"""
	conv1 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (10, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (16, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv3'
			  }


	conv1 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (19, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (6, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv3'
			  }


	conv10 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv3'
			  }
	conv4 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv4'
			  }
	conv5 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv5'
			  }
	conv6 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv6'
			  }
	conv8 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv8'
			  }"""