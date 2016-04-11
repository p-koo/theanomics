#/bin/python
from lasagne import layers, nonlinearities, init

def build_network(model_layers, name=[]):
	""" build all layers in the model """

	def single_layer(model_layer, network_last):
		""" build a single layer"""

		# input layer
		if model_layer['layer'] == 'input':
			network = layers.InputLayer(model_layer['shape'], input_var=model_layer['input_var'])

		# dense layer
		elif model_layer['layer'] == 'dense':
			network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
												 W=model_layer['W'],
												 b=model_layer['b'])

		# convolution layer
		elif model_layer['layer'] == 'convolution':
			network = layers.Conv2DLayer(network_last, num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
											 	  W=model_layer['W'],
										   		  b=model_layer['b'])
		return network

	# loop to build each layer of network
	network = {}
	lastname = ''
	for model_layer in model_layers:
		name = model_layer['name']

		if name == "input":
			network[name] = single_layer(model_layer, network)
			lastname = name
		else:
			network[name] = single_layer(model_layer, network[lastname])
			lastname = name
				
		# add Batch normalization layer
		if 'norm' in model_layer:
			if model_layer['norm'] == 'batch':
				network[name+'_batch'] = layers.BatchNormLayer(network[lastname])
				lastname = name+'_batch'

		# add activation layer
		if 'activation' in model_layer:
			network[name] = activation_layer(network[lastname], model_layer['activation']) 
			lastname = name
			
		# add dropout layer
		if 'dropout' in model_layer:
			network[name+'_dropout'] = layers.DropoutLayer(network[lastname], p=model_layer['dropout'])
			lastname = name+'_dropout'

		# add max-pooling layer
		if model_layer['layer'] == 'convolution':  
			network[name+'_pool'] = layers.MaxPool2DLayer(network[lastname], pool_size=model_layer['pool_size'])
			lastname = name+'_pool'          

	return network


def activation_layer(network_last, activation):

	if activation == 'prelu':
		network = layers.ParametricRectifierLayer(network_last,
												  alpha=init.Constant(0.25),
												  shared_axes='auto')
	elif activation == 'sigmoid':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.sigmoid)

	elif activation == 'softmax':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.softmax)

	elif activation == 'linear':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.linear)

	elif activation == 'tanh':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.tanh)

	elif activation == 'softplus':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.softplus)

	elif activation == 'leakyrelu':
			network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.leaky_rectify)
	
	elif activation == 'veryleakyrelu':
			network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.very_leaky_rectify)
		
	elif activation == 'relu':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.rectify)
	
	return network






