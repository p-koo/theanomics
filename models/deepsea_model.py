#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal, 
from lasagne.init import GlorotUniform, HeNormal, HeUniform

def DeepSea_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

    # create model
	layer1 = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape
	          }
	layer2 = {'layer': 'convolution', 
	          'num_filters': 240, 
	          'filter_size': (8, 1),
	          'pool_size': (4, 1)
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'dropout': .2,
	          'activation': 'relu'
	          }
	layer3 = {'layer': 'convolution', 
	          'num_filters': 480, 
	          'filter_size': (8, 1),
	          'pool_size': (4, 1)
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'dropout': .2,
	          'activation': 'relu'
	          }
	layer4 = {'layer': 'convolution', 
	          'num_filters': 960, 
	          'filter_size': (8, 1),
	          'pool_size': (4, 1)
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'dropout': .2,
	          'activation': 'relu'
	          }
	layer5 = {'layer': 'dense', 
	          'num_units': 1000, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'dropout': .5,
	          'activation': 'relu'
	          }
	layer6 = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'activation': sigmoid
	          }
	          
	layers = [layer1, layer2, layer3, layer4, layer5, layer6]

	# optimization parameters
	optimization = {"objective": "binary",
	                "optimizer": "nesterov_momentum", 
	                "learning_rate": 0.1,
	                "momentum": 0.9
	                "weight_norm": 10,
	                "l1": 1e-7,
	                "l2": 1e-8, 
	                }

	return layers, input_var, target_var, optimization

