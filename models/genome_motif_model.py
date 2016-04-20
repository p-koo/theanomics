#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def genome_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	layer1 = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	layer2 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'local_norm': 'local', 
			  #'batch_norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv1'
			  }
	layer3 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'local_norm': 'local', 
	          #'batch_norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv2'
			  }
			  
	layer4 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'local_norm': 'local', 
	          #'batch_norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'dropout': .3,
			  'name': 'conv3'
			  }
  	"""
  	layer5 = {'layer': 'convolution', 
			  'num_filters': 200, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'local_norm': 'local', 
			  #'batch_norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv4'
			  }
 	""" 
	layer6 = {'layer': 'dense', 
			  'num_units': 500, 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'batch_norm': 'batch', 
			  'activation': 'prelu', 
			  'dropout': .5,
			  'name': 'dense1'
			  }


  	layer7 = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'activation': 'sigmoid', 
			  'name': 'output'
			  }
			  
	model_layers = [layer1, layer2, layer3, layer4, layer6, layer7]
	network = build_network(model_layers)


	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
	                #"optimizer": "rmsprop",
	                "learning_rate": 0.01,	                
	                "beta1": 0.9,
	                "beta2": 0.999,
	                #"rho": .9,
	                "epsilon": 1e-6,
	                "weight_norm": 7, 
#	                "momentum": 0.9
	                "l1": 1e-7,
	                "l2": 1e-8
					}

	return network, input_var, target_var, optimization

