import sys
sys.path.append('..')
from tfomics import utils, init
import tensorflow as tf



def model(input_shape, output_shape):

    # create model
    layer1 = {'layer': 'input',
                'input_shape': input_shape,
                'name': 'input'
                }
    layer2 = {'layer': 'conv1d', 
                'num_filters': {'start': 20, 'bounds': [1, 200], 'scale': 20},
                'filter_size': {'start': 19, 'bounds': [5, 27], 'scale': 10, 'multiples': 2, 'offset': 1},
                'norm': 'batch',
                'padding': 'same',
                'activation': 'relu',
                'pool_size': {'start': 20, 'bounds': [1, 200], 'scale': 10, 'multiples': 4},
                }
    layer3 = {'layer': 'dense', 
                'num_units': {'start': 64, 'bounds': [16, 1000], 'scale': 50},
                'norm': 'batch',
                'activation': 'relu',
                }
    layer3 = {'layer': 'dense', 
                'num_units': output_shape[1],
                'activation': 'sigmoid',
                }

    #from tfomics import build_network
    model_layers = [layer1, layer2, layer3]

    # optimization parameters
    optimization = {"objective": "binary",
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "l2": 1e-6
                    #"learning_rate": {'start': -3, 'bounds': [-4, -1], 'scale': 1.5, 'transform': 'log'},      
                    #"l2": {'start': -6, 'bounds': [-8, -2], 'scale': 3, 'transform': 'log'},
                    # "l1": 0, 
                    }
    return model_layers, optimization