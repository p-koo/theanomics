import sys
import collections

sys.path.append('..')
from theanomics.build_network import build_network 

def model(input_shape, output_shape):

	layer1 = {'layer': 'input',
			  'shape': input_shape
			  }
	layer2 = {'layer': 'conv2d', 
			  'num_filters': 16,  #240
			  'filter_size': (5, 5),
			  'norm': 'batch',
			  'activation': 'relu',
			  'pool_size': (2,2),
			  'pad': 'same',
			  'dropout': 0.2,
			  }
	layer3 = {'layer': 'conv2d', 
			  'num_filters': 32,  #240
			  'filter_size': (5, 5),
			  'norm': 'batch',
			  'activation': 'relu',
			  'pool_size': (2,2),
			  'pad': 'same',
			  'dropout': 0.2,
			  }
	layer4 = {'layer': 'dense', 
			  'num_units': 512,
			  'norm': 'batch',
			  'activation': 'relu',
			  }
	layer5 = {'layer': 'dense', 
			  'num_units': output_shape[1],
			  'activation': 'softmax',
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4, layer5]
	network, placeholders = build_network(model_layers, output_shape, supervised=True)

	# optimization parameters
	optimization = {"objective": "categorical",			  
					"optimizer": "adam",
					"learning_rate": 0.001,      
					"l2": 1e-6,
					# "l1": 0, 
					}


	return network, placeholders, optimization


