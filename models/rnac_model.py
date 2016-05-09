#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network
from six.moves import cPickle

def rnac_model(shape, num_labels):
	"""
	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

    # create model
	input_layer = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape,
  			  'name': 'input'
  			  }
	conv1 = {'layer': 'convolution', 
	          'num_filters': 68, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
  			  'name': 'conv1'
  			  }
	conv2 = {'layer': 'convolution', 
	          'num_filters': 256, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 1),
  			  'name': 'conv2'
  			  }
  	conv3= {'layer': 'convolution', 
	          'num_filters': 512, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
  			  'name': 'conv3'
  			  }
  	conv4= {'layer': 'convolution', 
	          'num_filters': 768, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 1),
  			  'name': 'conv4'
  			  }
  	conv5= {'layer': 'convolution', 
	          'num_filters': 1028, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
  			  'name': 'conv5'
  			  }
  	dense1 = {'layer': 'dense', 
	          'num_units': 1028, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'norm': 'batch',
	          'activation': 'prelu',
	          'dropout': .5,
  			  'name': 'dense'
  			  }
	dense2 = {'layer': 'dense', 
	          'num_units': 512, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'norm': 'batch',
	          'activation': 'prelu',
	          'dropout': .5,
  			  'name': 'dense'
  			  }
  	output = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotNormal(),
	          'b': Constant(0.05),
	          'activation': 'linear',
  			  'name': 'output'
  			  }

	model_layers = [input_layer, conv1, conv2, conv3, conv4, conv5, dense1, dense2, output]
	network = build_network(model_layers)

	f = open('/home/peter/Code/Deepomics/examples/Linv.pickle','rb')
	Linv = cPickle.load(f)
	f.close()

	# optimization parameters
	optimization = {"objective": "gls",
	                "optimizer": "adam",
	                "Linv": Linv,
	                "learning_rate": 0.001,	                
	                "beta1": .9,
	                "beta2": .999,
	                "epsilon": 1e-6
#	                "weight_norm": 7, 
#	                "momentum": 0.975,
#	                "l1": 1e-5,
#	                "l2": 1e-5
	                }
	                
	return network, input_var, target_var, optimization
	
	"""


	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

    # create model
	input_layer = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape,
  			  'name': 'input'
  			  }
	conv1 = {'layer': 'convolution', 
	          'num_filters': 256, 
	          'filter_size': (8, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 1),
  			  'name': 'conv1'
  			  }
	conv2 = {'layer': 'convolution', 
	          'num_filters': 512, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 1),
  			  'name': 'conv2'
  			  }
  	conv3= {'layer': 'convolution', 
	          'num_filters': 758, 
	          'filter_size': (4, 1),
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'pool_size': (2, 1),
	          'dropout': .2,
  			  'name': 'conv3'
  			  }
	dense1 = {'layer': 'dense', 
	          'num_units': 1028, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'norm': 'batch',
	          'activation': 'prelu',
	          'dropout': .5,
  			  'name': 'dense'
  			  }
	dense2 = {'layer': 'dense', 
	          'num_units': 512, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'norm': 'batch',
	          'activation': 'prelu',
	          'dropout': .5,
  			  'name': 'dense'
  			  }
  	output = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotNormal(),
	          'b': Constant(0.05),
	          'activation': 'linear',
  			  'name': 'output'
  			  }

	model_layers = [input_layer, conv1, conv2, conv3, dense1, dense2, output]
	network = build_network(model_layers)

	f = open('/home/peter/Code/Deepomics/examples/Linv.pickle','rb')
	Linv = cPickle.load(f)
	f.close()

	# optimization parameters
	optimization = {"objective": "ols",
	                "optimizer": "adam",
	                "Linv": Linv,
	                "learning_rate": 0.001,	                
	                "beta1": .9,
	                "beta2": .999,
	                "epsilon": 1e-6
#	                "weight_norm": 7, 
#	                "momentum": 0.975,
#	                "l1": 1e-5,
#	                "l2": 1e-5
	                }
	                
	return network, input_var, target_var, optimization
	#"""