import sys
import collections

sys.path.append('..')
from deepomics.build_network import build_network 
import theano.tensor as T

def model(shape):

	placeholders = collections.OrderedDict()
	placeholders['inputs'] = T.dmatrix('inputs')

	# create model
	layer1 = {'layer': 'input',
			  'input_var': placeholders['inputs'],
			  'shape': shape,
			  'name': 'input'
			  }
	layer2 = {'layer': 'dense', 
			  'num_units': 200,  
			  'norm': 'batch',
			  'activation': 'relu',
			  'dropout': 0.3,
			  'name': 'dense1'
			  }
	layer3 = {'layer': 'dense', 
			  'num_units': 100,
			  'norm': 'batch',
			  'activation': 'relu',
			  'dropout': 0.3,
			  'name': 'dense2'  
			  }
	layer4 = {'layer': 'variational', 
			  'num_units': 4,
			  'activation': 'relu',
			  'name': 'encode'  
			  }
	layer5 = {'layer': 'dense', 
			  'num_units': 100,
			  'activation': 'relu',
			  'name': 'dense3'
			  }
	layer6 = {'layer': 'dense', 
			  'num_units': 200,
			  'activation': 'relu',
			  'name': 'dense4'
			  }
	layer7 = {'layer': 'dense', 
			  'num_units': shape[1],
			  'activation': 'relu',
			  'name': 'decode_mu'
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
	network = build_network(model_layers, supervised=False)

	# optimization parameters
	optimization = {"objective": "lower_bound",
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}


	return network, placeholders, optimization


