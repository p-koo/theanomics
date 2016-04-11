#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
sys.path.append('..')
from src import build_network

def binary_genome_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

    # create model
	layer1 = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape,
  			  'name': 'input'
  			  }
	layer2 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': (8, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (4, 1),
  			  'name': 'conv1'
  			  }
	layer3 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': (8, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'dropout': .2,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (4, 1),
  			  'name': 'conv2'
  			  }
	layer4 = {'layer': 'dense', 
	          'num_units': 200, 
	          'default': True,
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'dropout': .5,
	          'norm': 'batch',
	          'activation': 'prelu',
  			  'name': 'dense3'
  			  }
	layer5 = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'default': True,
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'activation': 'sigmoid',
  			  'name': 'output'
  			  }

	model_layers = [layer1, layer2, layer3, layer4, layer5]
    network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
	                "optimizer": "adam",
	 #               "learning_rate": 0.1,
	#                "momentum": 0.9
	                "weight_norm": 10
	                #"l1": 1e-7,
	                #"l2": 1e-8
	                }

	return network, input_var, target_var, optimization
