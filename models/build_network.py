#/bin/python
from lasagne import layers, nonlinearities, init

def build_network(model_layers):
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
	lastlayer = ''
	for model_layer in model_layers:
		name = model_layer['name']

		if name == "input":
			network[name] = single_layer(model_layer, network)
			lastlayer = name
			counter = 1
		else:
			newlayer = name #'# str(counter) + '_' + name + '_batch'
			network[newlayer] = single_layer(model_layer, network[lastlayer])
			lastlayer = newlayer
			counter += 1
				

		if 'local_norm' in model_layer:
			newlayer = name + '_local' # str(counter) + '_' + name + '_local'
			network[newlayer] = layers.LocalResponseNormalization2DLayer(network[lastlayer])
			lastlayer = newlayer

			
		# add Batch normalization layer
		if 'batch_norm' in model_layer:
			newlayer = name + '_batch' #str(counter) + '_' + name + '_batch'
			network[newlayer] = layers.BatchNormLayer(network[lastlayer])
			lastlayer = newlayer
			
		# add dropout layer
		if 'dropout' in model_layer:
			newlayer = name+'_dropout' # str(counter) + '_' + name+'_dropout'
			network[newlayer] = layers.DropoutLayer(network[lastlayer], p=model_layer['dropout'])
			lastlayer = newlayer

			
		# add activation layer
		if 'activation' in model_layer:
			if name == 'output':
				newlayer = name
			else:
				newlayer = name+'_active'
			network[newlayer] = activation_layer(network[lastlayer], model_layer['activation']) 
			lastlayer = newlayer
		

		# add max-pooling layer
		if model_layer['layer'] == 'convolution':  
			newlayer = name+'_pool'  # str(counter) + '_' + name+'_pool' 
			network[newlayer] = layers.MaxPool2DLayer(network[lastlayer], pool_size=model_layer['pool_size'])
			lastlayer = newlayer       
			counter += 1

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






