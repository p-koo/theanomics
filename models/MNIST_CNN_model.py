#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network

def MNIST_CNN_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

    # create model
	layer1 = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape,
  			  'name': 'input'
			 }
	layer2 = {'layer': 'convolution', 
	          'num_filters': 32, 
	          'filter_size': (5, 5),
	          'W': GlorotUniform(),
	          'b': Constant(.05),
	          'activation': 'relu',
	          'pool_size': (2, 2),
  			  'name': 'conv1'
  			  }
	layer3 = {'layer': 'convolution', 
	          'num_filters': 64, 
	          'filter_size': (5, 5),
	          'W': GlorotUniform(),
	          'b': Constant(.05), 
	          'activation': 'relu',
	          'pool_size': (2, 2),
  			  'name': 'conv2'
  			  }
	layer4 = {'layer': 'dense', 
	          'num_units': 500, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'dropout': .5,
	          'activation': 'relu',
  			  'name': 'dense'
  			  }
	layer5 = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'activation': 'softmax',
  			  'name': 'output'
  			  }
	"""
	 # create model
	layer1 = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape,
  			  'name': 'input'
			 }
	layer2 = {'layer': 'convolution', 
	          'num_filters': 32, 
	          'filter_size': (5, 5),
	          'W': GlorotUniform(),
	          'b': None,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 2),
  			  'name': 'conv1'
  			  }
	layer3 = {'layer': 'convolution', 
	          'num_filters': 64, 
	          'filter_size': (5, 5),
	          'W': GlorotUniform(),
	          'b': None,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'dropout': .5,
	          'pool_size': (2, 2),
  			  'name': 'conv2'
  			  }
	layer4 = {'layer': 'dense', 
	          'num_units': 100, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'dropout': .5,
	          'norm': 'batch',
	          'activation': 'prelu',
  			  'name': 'dense'
  			  }
	layer5 = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'activation': 'softmax',
  			  'name': 'output'
  			  }
  			  """

	model_layers = [layer1, layer2, layer3, layer4, layer5]
	network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "categorical",
                "optimizer": "adam"
#                "learning_rate": 0.1,
#                "momentum": 0.9, 
                #"weight_norm": 10,
                #"l1": 1e-4,
                #"l2": 1e-5
                }
                
	return network, input_var, target_var, optimization
