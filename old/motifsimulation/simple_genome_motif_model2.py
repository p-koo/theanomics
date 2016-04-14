#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal, 
from lasagne.init import GlorotUniform, HeNormal, HeUniform

def simple_genome_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

    # create model
	layer1 = {'layer': 'input',
	          'input_var': input_var,
	          'shape': shape
	          }
  	layer2 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': (8, 1),
	          'pool_size': (4, 1)
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch'
	          'activation': 'prelu'
	          }
	layer3 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': (8, 1),
	          'pool_size': (4, 1)
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'norm': 'batch'
	          'dropout': .2,
	          'activation': 'prelu'
	          }
	layer4 = {'layer': 'dense', 
	          'num_units': 200, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05), 
	          'norm': 'batch'
	          'dropout': .5,
	          'activation': 'prelu'
	          }
	layer5 = {'layer': 'dense', 
	          'num_units': num_labels, 
	          'W': GlorotUniform(),
	          'b': Constant(0.05),
	          'activation': softmax
	          }
	          
	layers = [layer1, layer2, layer3, layer4, layer5]

	# optimization parameters
	optimization = {"objective": "categorical",
	                "optimizer": "adam", 
	                "weight_norm": 5,
	                "l1": 1e-7,
	                "l2": 1e-8, 
	                }

	return layers, input_var, target_var, optimization

