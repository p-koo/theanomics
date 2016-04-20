#/bin/python
from lasagne import layers, nonlinearities, init

def build_network(model_layers, autoencode=0):
	""" build all layers in the model """
	
	network, lastlayer = build_forward_layers(model_layers)
	if autoencode == 1:
		network = build_decode_layers(model_layers, network, lastlayer)

	return network


def build_decode_layers(model_layers, network, last_layer):

	def get_reverse_layers(model_layers):
		reverse_layers = []
		num_layers = len(model_layers)
		for i in range(num_layers-1, -1, -1):
			reverse_layers.append(model_layers[i])
		return reverse_layers

	reverse_layers = get_reverse_layers(model_layers)

	# encode layer
	network['encode'] = layers.NonlinearityLayer(network[last_layer], nonlinearity=None)
	lastlayer = 'encode'

	for model_layer in reverse_layers:
		name = model_layer['name']

		if name == "input":
			break
		else:

			# add max-pooling layer
			if 'pool_size' in model_layer:  
				newlayer = name+'_pool_inv'  # str(counter) + '_' + name+'_pool' 
				network[newlayer] = layers.InverseLayer(network[lastlayer], network[name+'_pool'])
				lastlayer = newlayer       


				# add core layer
				newlayer = name+'_inv'
				network[newlayer] = layers.InverseLayer(network[lastlayer], network[name])
				lastlayer = newlayer

				newlayer = name+'_bias_inv'
				network[newlayer] = layers.BiasLayer(network[lastlayer], b=model_layer['b'])
				lastlayer = newlayer	
				
			
		# add Batch normalization layer
		if 'norm' in model_layer:
			if model_layer['norm'] == 'batch':
				newlayer = name + '_batch_inv' #str(counter) + '_' + name + '_batch'
				network[newlayer] = layers.BatchNormLayer(network[lastlayer])
			elif model_layer['norm'] == 'local':
				newlayer = name + '_local_inv' # str(counter) + '_' + name + '_local'
				network[newlayer] = layers.LocalResponseNormalization2DLayer(network[lastlayer], 
													alpha=.001/9.0, k=1., beta=0.75, n=5)
			lastlayer = newlayer
			
		# add activation layer
		if 'activation' in model_layer:
			if name == 'output':
				newlayer = name
			else:
				newlayer = name+'_active_inv'
			network[newlayer] = activation_layer(network[lastlayer], model_layer['activation']) 
			lastlayer = newlayer
		
		# add dropout layer
		if 'dropout' in model_layer:
			newlayer = name+'_dropout_inv' # str(counter) + '_' + name+'_dropout'
			network[newlayer] = layers.DropoutLayer(network[lastlayer], p=model_layer['dropout'])
			lastlayer = newlayer


	return network



def build_forward_layers(model_layers, network={}):

	# loop to build each layer of network
	lastlayer = ''
	for model_layer in model_layers:
		name = model_layer['name']

		if name == "input":
			# add input layer
			network[name] = single_layer(model_layer, network)
			lastlayer = name
		else:
			# add core layer
			newlayer = name #'# str(counter) + '_' + name + '_batch'
			network[newlayer] = single_layer(model_layer, network[lastlayer])
			lastlayer = newlayer

			# add bias layer
			newlayer = name+'_bias'
			network[newlayer] = layers.BiasLayer(network[lastlayer], b=model_layer['b'])
			lastlayer = newlayer	
				
			
		# add Batch normalization layer
		if 'norm' in model_layer:
			if model_layer['norm'] == 'batch':
				newlayer = name + '_batch' #str(counter) + '_' + name + '_batch'
				network[newlayer] = layers.BatchNormLayer(network[lastlayer])
			elif model_layer['norm'] == 'local':
				newlayer = name + '_local' # str(counter) + '_' + name + '_local'
				network[newlayer] = layers.LocalResponseNormalization2DLayer(network[lastlayer], 
													alpha=.001/9.0, k=1., beta=0.75, n=5)
			lastlayer = newlayer
			
		# add activation layer
		if 'activation' in model_layer:
			if name == 'output':
				newlayer = name
			else:
				newlayer = name+'_active'
			network[newlayer] = activation_layer(network[lastlayer], model_layer['activation']) 
			lastlayer = newlayer
		
		# add dropout layer
		if 'dropout' in model_layer:
			newlayer = name+'_dropout' # str(counter) + '_' + name+'_dropout'
			network[newlayer] = layers.DropoutLayer(network[lastlayer], p=model_layer['dropout'])
			lastlayer = newlayer

		# add max-pooling layer
		if 'pool_size' in model_layer:  
			newlayer = name+'_pool'  # str(counter) + '_' + name+'_pool' 
			network[newlayer] = layers.MaxPool2DLayer(network[lastlayer], pool_size=model_layer['pool_size'])
			lastlayer = newlayer       

	return network, lastlayer



def single_layer(model_layer, network_last):
	""" build a single layer"""

	# input layer
	if model_layer['layer'] == 'input':
		network = layers.InputLayer(model_layer['shape'], input_var=model_layer['input_var'])

	# dense layer
	elif model_layer['layer'] == 'dense':
		network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
											 W=model_layer['W'],
											 b=None, 
											 nonlinearity=None)

	# convolution layer
	elif model_layer['layer'] == 'convolution':
		network = layers.dnn.Conv2DDNNLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=model_layer['filter_size'],
											  W=model_layer['W'],
											  b=None, 
											  nonlinearity=None)
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






