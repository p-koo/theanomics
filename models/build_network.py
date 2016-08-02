#/bin/python
from lasagne import layers, nonlinearities, init
from lasagne.layers.base import Layer

def build_network(model_layers, autoencode=0):
	""" build all layers in the model """
	
	network, lastlayer = build_layers(model_layers)
	network['output'] = network[lastlayer]
	return network


def build_layers(model_layers, network={}):

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
			if 'b' in model_layer:
				newlayer = name+'_bias'
				network[newlayer] = layers.BiasLayer(network[lastlayer], b=model_layer['b'])
				lastlayer = newlayer	
				
			
		# add Batch normalization layer
		if 'norm' in model_layer:
			if 'batch' in model_layer['norm']:
				newlayer = name + '_batch' #str(counter) + '_' + name + '_batch'
				network[newlayer] = layers.BatchNormLayer(network[lastlayer])
				lastlayer = newlayer
			
		# add activation layer
		if 'activation' in model_layer:
			newlayer = name+'_active'
			network[newlayer] = activation_layer(network[lastlayer], model_layer['activation']) 
			lastlayer = newlayer
		
		# add Batch normalization layer
		if 'norm' in model_layer:
			if 'local' in model_layer['norm']:
				newlayer = name + '_local' # str(counter) + '_' + name + '_local'
				network[newlayer] = layers.LocalResponseNormalization2DLayer(network[lastlayer], 
													alpha=.001/9.0, k=1., beta=0.75, n=5)
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
											  pad=model_layer['pad'],
											  nonlinearity=None)

	elif model_layer['layer'] == 'lstm':
		l_forward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
											grad_clipping=model_layer['grad_clipping'])
		l_backward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
											grad_clipping=model_layer['grad_clipping'], 
											backwards=True)
		network = layers.ConcatLayer([l_forward, l_backward])

	elif model_layer['layer'] == 'highway':
		network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
											 W=model_layer['W'],
											 b=None, 
											 nonlinearity=None)
		for k in range(model_layer['num_layers']):
			network = highway_dense(network)
		

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



class MultiplicativeGatingLayer(layers.MergeLayer):
	"""
	Generic layer that combines its 3 inputs t, h1, h2 as follows:
	y = t * h1 + (1 - t) * h2
	"""
	def __init__(self, gate, input1, input2, **kwargs):
		incomings = [gate, input1, input2]
		super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
		assert gate.output_shape == input1.output_shape == input2.output_shape
	
	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]
	
	def get_output_for(self, inputs, **kwargs):
		return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]


def highway_dense(incoming, Wh=init.Orthogonal(), bh=init.Constant(0.0),
				  Wt=init.Orthogonal(), bt=init.Constant(-4.0),
				  nonlinearity=nonlinearities.rectify, **kwargs):

	num_inputs = int(np.prod(incoming.output_shape[1:]))
	# regular layer
	l_h = layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
							   nonlinearity=nonlinearity)
	# gate layer
	l_t = layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
							   nonlinearity=T.nnet.sigmoid)
	
	return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)



class DecorrLayer():
	def __init__(self, incoming, L, **kwargs):
		self.L = L
		super(DecorrLayer, self).__init__(incoming, L, **kwargs)

	def get_output_shape_for(self, input_shape):
		return input_shape[0]

	def get_output_for(self, input, **kwargs):
		
		return T.dot(self.L, input.T).T


class DenoiseLayer(layers.MergeLayer):
	"""
		Special purpose layer used to construct the ladder network
		See the ladder_network example.
	"""
	def __init__(self, u_net, z_net,
				 nonlinearity=nonlinearities.sigmoid, **kwargs):
		super(DenoiseLayer, self).__init__([u_net, z_net], **kwargs)

		u_shp, z_shp = self.input_shapes


		if not u_shp[-1] == z_shp[-1]:
			raise ValueError("last dimension of u and z  must be equal"
							 " u was %s, z was %s" % (str(u_shp), str(z_shp)))
		self.num_inputs = z_shp[-1]
		self.nonlinearity = nonlinearity
		constant = init.Constant
		self.a1 = self.add_param(constant(0.), (self.num_inputs,), name="a1")
		self.a2 = self.add_param(constant(1.), (self.num_inputs,), name="a2")
		self.a3 = self.add_param(constant(0.), (self.num_inputs,), name="a3")
		self.a4 = self.add_param(constant(0.), (self.num_inputs,), name="a4")

		self.c1 = self.add_param(constant(0.), (self.num_inputs,), name="c1")
		self.c2 = self.add_param(constant(1.), (self.num_inputs,), name="c2")
		self.c3 = self.add_param(constant(0.), (self.num_inputs,), name="c3")

		self.c4 = self.add_param(constant(0.), (self.num_inputs,), name="c4")

		self.b1 = self.add_param(constant(0.), (self.num_inputs,),
								 name="b1", regularizable=False)

	def get_output_shape_for(self, input_shapes):
		output_shape = list(input_shapes[0])  # make a mutable copy
		return tuple(output_shape)

	def get_output_for(self, inputs, **kwargs):
		u, z_lat = inputs
		sigval = self.c1 + self.c2*z_lat
		sigval += self.c3*u + self.c4*z_lat*u
		sigval = self.nonlinearity(sigval)
		z_est = self.a1 + self.a2 * z_lat + self.b1*sigval
		z_est += self.a3*u + self.a4*z_lat*u
		return z_est

class DecoderNormalizeLayer(layers.MergeLayer):
	"""
		Special purpose layer used to construct the ladder network
		See the ladder_network example.
	"""
	def __init__(self, incoming, mean, var, **kwargs):
		super(DecoderNormalizeLayer, self).__init__(
			[incoming, mean, var], **kwargs)

	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]

	def get_output_for(self, inputs, **kwargs):
		input, mean, var = inputs
		return (input - mean) / T.sqrt(var)