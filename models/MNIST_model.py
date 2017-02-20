import sys
import collections

sys.path.append('..')
from deepomics.build_network import build_network 
import theano.tensor as T

def model(shape, num_labels):

	placeholders = collections.OrderedDict()
	placeholders['inputs'] = T.tensor4('inputs')
	placeholders['targets'] = T.dmatrix('targets')

	# create model
	layer1 = {'layer': 'input',
			  'input_var': placeholders['inputs'],
			  'shape': shape,
			  'name': 'input'
			  }
	layer2 = {'layer': 'conv2d', 
			  'num_filters': 16,  #240
			  'filter_size': (5, 5),
			  'norm': 'batch',
			  'activation': 'relu',
			  'pool_size': (2,2),
			  'pad': 'same',
			  'dropout': 0.2,
			  'name': 'conv1'
			  }
	layer3 = {'layer': 'conv2d', 
			  'num_filters': 32,  #240
			  'filter_size': (5, 5),
			  'norm': 'batch',
			  'activation': 'relu',
			  'pool_size': (2,2),
			  'pad': 'same',
			  'dropout': 0.2,
			  'name': 'conv2'
			  }
	layer4 = {'layer': 'dense', 
			  'num_units': 512,
			  'norm': 'batch',
			  'activation': 'relu',
			  'name': 'dense1'
			  }
	layer5 = {'layer': 'dense', 
			  'num_units': num_labels,
			  'activation': 'softmax',
			  'name': 'dense2'
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4, layer5]
	network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "categorical",			  
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}


	return network, placeholders, optimization


