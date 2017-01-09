
from lasagne import layers, nonlinearities, init
import theano.tensor as T

	
def model(shape, num_labels):
	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=30, filter_size=(19, 1), stride=(1, 1),    # 1000
					 W=init.GlorotUniform(), b=None, nonlinearity=None, pad='same', flip_filters=False)
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(40, 1), stride=(40, 1), ignore_border=False) # 25
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=50, filter_size=(6, 1), stride=(1, 1), # 20
										W=init.GlorotUniform(), b=None, nonlinearity=None, pad='valid', flip_filters=False)
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(20, 1), stride=(20, 1), ignore_border=False) # 3
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.1)

	#net['conv3'] = layers.DenseLayer(net['conv2_dropout'], num_units=num_labels, W=init.GlorotUniform(), b=init.Constant(), nonlinearity=None)
	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1), #  1
						   W=init.GlorotUniform(), b=init.Constant(), nonlinearity=None, pad='valid', flip_filters=False)
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3'], nonlinearity=nonlinearities.sigmoid)
	net['output'] = layers.ReshapeLayer(net['conv3_active'], [-1, num_labels])

	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001, 
					"l2": 1e-5
					}

	return net, input_var, target_var, optimization
