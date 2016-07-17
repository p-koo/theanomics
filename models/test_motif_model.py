#/bin/python
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from build_network import build_network
from lasagne import layers, nonlinearities

def test_motif_model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(5, 1), stride=(1, 1),    # 196
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False) # 98
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=128, filter_size=(5, 1), stride=(1, 1), 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False) # 47
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=256, filter_size=(6, 1), stride=(1, 1),  #42
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False)  # 21
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.3)
	
	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=512, filter_size=(6, 1), stride=(1, 1), #16
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_active'], pool_size=(4, 1), stride=(4, 1), ignore_border=False) #5
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.3)
	
	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=2056, filter_size=(4, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)

	net['conv6'] = layers.Conv2DLayer(net['conv5_active'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=nonlinearities.sigmoid, pad='valid')

	net['output'] = layers.ReshapeLayer(net['conv6'], [-1, num_labels])

	

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,	                
					"beta1": .9,
					"beta2": .999
#	                "weight_norm": 7, 
#	                "momentum": 0.9
					#"l1": 1e-5,
					#"l2": 1e-6
					}

	return net, input_var, target_var, optimization

"""
	# very deep -- strided convolutions
	
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(11, 1), stride=(1, 1), # 64
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_active'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=512, filter_size=(11, 1), stride=(10, 1), # 18
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_active'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=1028, filter_size=(7, 1), stride=(2, 1), # 6
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.3)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=2056, filter_size=(6, 1), stride=(1, 1), # 4
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_active'], p=0.3)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv5_active'], [-1, num_labels])


"""
"""
# shallow
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(9, 1), stride=(1, 1),    # 196
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(20, 1), stride=(20, 1), ignore_border=False) # 65
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=128, filter_size=(10, 1), stride=(1, 1), 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_active'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),  #26
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3'], nonlinearity=nonlinearities.sigmoid)
	
	net['output'] = layers.ReshapeLayer(net['conv3_active'], [-1, num_labels])

"""

"""
# medium
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(9, 1), stride=(1, 1),    # 196
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(22, 1), stride=(12, 1), ignore_border=False) # 98
	#net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_pool'], num_filters=512, filter_size=(5, 1), stride=(1, 1), 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 47
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=2056, filter_size=(4, 1), stride=(1, 1),  #42
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.3)
	
	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1), #16
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv4_active'], [-1, num_labels])



"""




"""
# very deep
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(5, 1), stride=(1, 1),    # 196
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False) # 98
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=128, filter_size=(5, 1), stride=(1, 1), 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False) # 47
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=256, filter_size=(6, 1), stride=(1, 1),  #42
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_active'], pool_size=(2, 1), stride=(2, 1), ignore_border=False)  # 21
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.3)
	
	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=512, filter_size=(6, 1), stride=(1, 1), #16
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_active'], pool_size=(4, 1), stride=(4, 1), ignore_border=False) #5
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.3)
	
	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=2056, filter_size=(4, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)

	net['conv6'] = layers.Conv2DLayer(net['conv5_active'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=nonlinearities.sigmoid, pad='valid')

	net['output'] = layers.ReshapeLayer(net['conv6'], [-1, num_labels])

"""



"""

	# create model
	input_layer = {'layer': 'input',
				   'input_var': input_var,
				   'shape': shape,
				   'name': 'input'
				   }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 35, #30
			  'filter_size': (9, 1), #189
			  'W': GlorotUniform(),
			  'b': None,
			  'pad': 'same',
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (20, 1), #20
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 20,  #16
			  'filter_size': (10, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'norm': 'batch', 
			  'activation': 'linear',
			  #'pool_size': (5, 1), #20
			  'pad': 'valid',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 40,  #16
			  'filter_size': (3, 1),
			  'W': GlorotUniform(),
			  'b': None,
			  'activation': 'linear',
			  'pad': 'valid',
			  'name': 'conv3'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': Constant(1.),
			  'b': None,
			  'activation': 'sigmoid', 
			  'name': 'dense2'
			  }
			  
	model_layers = [input_layer, conv1, conv2, output] 
	network = build_network(model_layers)





	conv1 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (10, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (16, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv3'
			  }


	conv1 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (19, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (6, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (4, 1),
			  'name': 'conv3'
			  }


	conv10 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv3'
			  }
	conv4 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv4'
			  }
	conv5 = {'layer': 'convolution', 
			  'num_filters': 64, 
			  'filter_size': (4, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv5'
			  }
	conv6 = {'layer': 'convolution', 
			  'num_filters': 128, 
			  'filter_size': (12, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'pool_size': (3, 1),
			  'name': 'conv6'
			  }
	conv8 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (8, 1),
			  'W': GlorotUniform(),
			  'b': Constant(0.05),
			  'norm': 'batch', 
			  'activation': 'prelu',
			  'name': 'conv8'
			  }"""
