#/bin/python
from lasagne import layers, nonlinearities, init

def build_network(model_layers):
	""" build all layers in the model """

	def single_layer(model_layer, network):
		""" build a single layer"""

		# input layer
		if model_layer['layer'] == 'input':
			network = layers.InputLayer(model_layer['shape'], input_var=model_layer['input_var'])

		# dense layer
		elif model_layer['layer'] == 'dense':
			network = layers.DenseLayer(network, num_units=model_layer['num_units'],
												 W=model_layer['W'],
												 b=model_layer['b'])

		# convolution layer
		elif model_layer['layer'] == 'convolution':
			network = layers.Conv2DLayer(network, num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
											 	  W=model_layer['W'],
										   		  b=model_layer['b'])
		return network

	# loop to build each layer of network
	network = {}
	for model_layer in model_layers:

		# create base layer
		network = single_layer(model_layer, network)
				
		# add Batch normalization layer
		if 'norm' in model_layer:
			if model_layer['norm'] == 'batch':
				network = layers.BatchNormLayer(network)

		# add activation layer
		if 'activation' in model_layer:
			network = activation_layer(network, model_layer['activation']) 
			
		# add dropout layer
		if 'dropout' in model_layer:
			layers.DropoutLayer(network, p=model_layer['dropout'])

		# add max-pooling layer
		if model_layer['layer'] == 'convolution':            
			network = layers.MaxPool2DLayer(network, pool_size=model_layer['pool_size'])

	return network


def activation_layer(network, activation):

	if activation == 'prelu':
		network = layers.ParametricRectifierLayer(network,
												  alpha=init.Constant(0.25),
												  shared_axes='auto')
	elif activation == 'sigmoid':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.sigmoid)

	elif activation == 'softmax':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.softmax)

	elif activation == 'linear':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.linear)

	elif activation == 'tanh':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.tanh)

	elif activation == 'softplus':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.softplus)

	elif activation == 'leakyrelu':
			network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.leaky_rectify)
	
	elif activation == 'veryleakyrelu':
			network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.very_leaky_rectify)
		
	elif activation == 'relu':
		network = layers.NonlinearityLayer(network, nonlinearity=nonlinearities.rectify)
	
	return network






