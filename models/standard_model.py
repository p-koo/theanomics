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
	layer2 = {'layer': 'conv1d', 
			  'num_filters': 30,  #240
			  'filter_size': 19,
			  'norm': 'batch',
			  'activation': 'relu',
			  'pool_size': 20,
			  'dropout': 0.2,
			  'name': 'conv1'
			  }
	layer3 = {'layer': 'dense', 
			  'num_units': 512,
			  'norm': 'batch',
			  'activation': 'relu',
			  'dropout': 0.5,
			  'name': 'dense1'
			  }
	layer4 = {'layer': 'dense', 
			  'num_units': num_labels,
			  'activation': 'sigmoid',
			  'name': 'dense1'
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4]
	network = build_network(model_layers)

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}


	return network, placeholders, optimization

