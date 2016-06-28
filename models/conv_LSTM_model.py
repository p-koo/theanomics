
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import ConcatLayer, LSTMLayer, get_output_shape, LocalResponseNormalization2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

from lasagne.nonlinearities import softmax, sigmoid, rectify, linear
from lasagne.nonlinearities import leaky_rectify, tanh, very_leaky_rectify

from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform

# Examples:
#   net['conv1'] = ConvLayer(net['input'], num_filters=200, filter_size=(12, 1), stride=(1, 1),
#                                          W=GlorotUniform(), b=Constant(.05), nonlinearity=None)
#   net['pool'] = PoolLayer(net['something'], pool_size=(4, 1), stride=(4, 1), ignore_border=False)
#   net['batch'] = BatchNormLayer(net['something'])
#   net['active'] = NonlinearityLayer(net['something'], sigmoid)
#   net['dense'] = DenseLayer(net['something'], num_units=200, W=GlorotUniform(), b=None, nonlinearity=None)
#   net['drop4'] = DropoutLayer(net['something'], p=0.5)
#   net['prelu'] = ParametricRectifierLayer(net['something'], alpha=Constant(0.25), shared_axes='auto')



def conv_LSTM_model(shape, num_labels):
	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	# create model
	input_layer = {'layer': 'input',
				   'input_var': input_var,
				   'shape': shape,
				   'name': 'input'
				   }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (7, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv2'
			  }

	conv3 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (5, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (2, 1),
			  'name': 'conv3'
			  }

	lstm = {'layer': 'lstm', 
			  'num_units': 25, 
			  'grad_clipping': 50,
			  'name': 'lstm'
			  }

	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': None,
			  'activation': 'sigmoid', 
			  'name': 'dense'
			  }
			  
	model_layers = [input_layer, conv1, conv2, conv3, lstm, output] 
	network = build_network(model_layers)


	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                    
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-8,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
					"l1": 1e-4,
					"l2": 1e-5
					}

	return network, input_var, target_var, optimization



"""
	
def bidirectionalLSTM(l_in, num_units, grad_clipping):
	l_forward = LSTMLayer(l_in, num_units=num_units, grad_clipping=grad_clipping)
	l_backward = LSTMLayer(l_in, num_units=num_units, grad_clipping=grad_clipping, backwards=True)
	return ConcatLayer([l_forward, l_backward])


def conv_LSTM_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	net = {}
	net['input'] = InputLayer(input_var=input_var, shape=shape)

	net['conv1'] = ConvLayer(net['input'], num_filters=64, filter_size=(5, 1), stride=(1, 1),
										   W=GlorotUniform(), b=Constant(.05), pad='same', nonlinearity=None)
	net['conv1_norm'] = BatchNormLayer(net['conv1'], epsilon=0.001)
	net['conv1_active'] = NonlinearityLayer(net['conv1_norm'], leaky_rectify)
	net['conv1_pool'] = PoolLayer(net['conv1_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False)
	
	net['conv2'] = ConvLayer(net['conv1_pool'], num_filters=128, filter_size=(5, 1), stride=(1, 1),
										   W=GlorotUniform(), b=Constant(.05), pad='same', nonlinearity=None)
	net['conv2_norm'] = BatchNormLayer(net['conv2'], epsilon=0.001)
	net['conv2_active'] = NonlinearityLayer(net['conv2_norm'], leaky_rectify)
	net['conv2_pool'] = PoolLayer(net['conv2_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False)


	net['lstm'] = bidirectionalLSTM(net['conv2_pool'], num_units=50, grad_clipping=100)
	net['lstm2'] = bidirectionalLSTM(net['lstm'], num_units=50, grad_clipping=100)	

	net['dense'] = DenseLayer(net['lstm2'], num_units=num_labels, W=GlorotUniform(), b=Constant(0.05), nonlinearity=None)
	net['output'] = NonlinearityLayer(net['dense'], sigmoid)

	
	network = net


	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                    
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-8,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
					"l1": 1e-4,
					"l2": 1e-5
					}

	return network, input_var, target_var, optimization

"""