#/bin/python
import collections
from lasagne import layers, nonlinearities, init
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


__all__ = [
	"build_network"
]

def build_network(model_layers, supervised=True):
	""" build all layers in the model """
	
	network, last_layer = build_layers(model_layers)
	if supervised:
		network['output'] = network[last_layer]
	else:
		network['X'] = network[last_layer]
	return network


def build_layers(model_layers, network=collections.OrderedDict()):

	# loop to build each layer of network
	last_layer = ''
	for model_layer in model_layers:
		name = model_layer['name']
		layer = model_layer['layer']

		if name == "input":
			# add input layer
			network[name] = single_layer(model_layer, network)
			last_layer = name
		else:

			if layer == 'conv1d_residual':
				if 'residual_dropout' in model_layer:
					dropout = model_layer['residual_dropout']
				else:
					dropout = None
				network = conv1d_residual(network, last_layer, model_layer['name'], model_layer['filter_size'], 
											nonlinearity=nonlinearities.rectify, dropout=dropout)
				new_layer = name+'_resid'
				last_layer = new_layer

			elif layer == 'conv2d_residual':
				if 'residual_dropout' in model_layer:
					dropout = model_layer['residual_dropout']
				else:
					dropout = None
				network = conv2d_residual(network, last_layer, model_layer['name'], model_layer['filter_size'], 
											nonlinearity=nonlinearities.rectify, dropout=dropout)
				new_layer = name+'_resid'
				last_layer = new_layer

			elif layer == 'dense_residual':
				if 'residual_dropout' in model_layer:
					dropout = model_layer['residual_dropout']
				else:
					dropout = None
				network = dense_residual(network, last_layer, model_layer['name'], 
											nonlinearity=nonlinearities.rectify, dropout=dropout)
				new_layer = name+'_resid'
				last_layer = new_layer

			elif layer == 'variational':
				network['encode_mu'] = layers.DenseLayer(network[last_layer], num_units=model_layer['num_units'])
				network['encode_logsigma'] = layers.DenseLayer(network[last_layer], num_units=model_layer['num_units'])
				network['Z'] = VariationalSampleLayer(network['encode_mu'], network['encode_logsigma'])
				last_layer = 'Z'
			else:

				# add core layer
				new_layer = name #'# str(counter) + '_' + name + '_batch'
				network[new_layer] = single_layer(model_layer, network[last_layer])
				last_layer = new_layer

		# add Batch normalization layer
		if 'norm' in model_layer:
			if 'batch' in model_layer['norm']:
				new_layer = name + '_batch' #str(counter) + '_' + name + '_batch'
				network[new_layer] = layers.BatchNormLayer(network[last_layer])
				last_layer = new_layer
		else:						# add bias layer
			if (model_layer['layer'] == 'dense') | (model_layer['layer'] == 'conv1d') | (model_layer['layer'] == 'conv2d'):		
				if ('b' in model_layer):
					if model_layer['b'] != None:
						if 'b' in model_layer:		
							b=model_layer['b']
					else:	
						b = init.Constant(0.05)		
				else:	
					b = init.Constant(0.05)		
				new_layer = name+'_bias'
				network[new_layer] = layers.BiasLayer(network[last_layer], b=b)
				last_layer = new_layer

		# add activation layer
		if 'activation' in model_layer:
			new_layer = name+'_active'
			network[new_layer] = activation_layer(network[last_layer], model_layer['activation']) 
			last_layer = new_layer

		# add Batch normalization layer
		if 'norm' in model_layer:
			if 'local' in model_layer['norm']:
				new_layer = name + '_local' # str(counter) + '_' + name + '_local'
				network[new_layer] = layers.LocalResponseNormalization2DLayer(network[last_layer], 
													alpha=.001/9.0, k=1., beta=0.75, n=5)
				last_layer = new_layer

		# add dropout layer
		if 'dropout' in model_layer:
			new_layer = name+'_dropout' # str(counter) + '_' + name+'_dropout'
			network[new_layer] = layers.DropoutLayer(network[last_layer], p=model_layer['dropout'])
			last_layer = new_layer

		# add max-pooling layer
		if 'pool_size' in model_layer:  
			if isinstance(model_layer['pool_size'], (list, tuple)):
				pool_size = model_layer['pool_size']
			else:
				if '1d' in layer:
					pool_size = (model_layer['pool_size'], 1)
				elif '2d' in layer:
					pool_size = (model_layer['pool_size'], model_layer['pool_size'])
			new_layer = name+'_pool'  # str(counter) + '_' + name+'_pool' 
			network[new_layer] = layers.MaxPool2DLayer(network[last_layer], pool_size=pool_size)
			last_layer = new_layer

		# global pooling layer
		if 'global_pool' in model_layers:
			new_layer = name+'_global_pool'  # str(counter) + '_' + name+'_pool' 
			if model_layers['global_pool'] == 'max':
				pool_function = T.max
			elif model_layers['global_pool'] == 'mean':
				pool_function = T.mean
			network[new_layer] = layers.GlobalPoolLayer(network[last_layer], pool_function=pool_function)
			last_layer = new_layer

		# unpooling layer
		if 'unpool_size' in model_layer:
			if isinstance(model_layer['unpool_size'], (list, tuple)):
				unpool_size = model_layer['unpool_size']
			else:
				if '1d' in name:
					unpool_size = (model_layer['unpool_size'], 1)
				elif '2d' in name:
					unpool_size = (model_layer['unpool_size'], model_layer['unpool_size'])
			new_layer = name+'_unpool'
			network[new_layer] = layers.Upscale2DLayer(network[last_layer], scale_factor=unpool_size)
			last_layer = new_layer

	return network, last_layer


def single_layer(model_layer, network_last):
	""" build a single layer"""

	# input layer
	if model_layer['layer'] == 'input':
		network = layers.InputLayer(model_layer['shape'], input_var=model_layer['input_var'])

	# dense layer
	elif model_layer['layer'] == 'dense':
		if 'W' in model_layer:
			W = model_layer['W']
		else:
			W = init.HeNormal()
		network = layers.DenseLayer(network_last, num_units=model_layer['num_units'],
											 W=W,
											 b=None, 
											 nonlinearity=None)

	# 1D convolution layer
	elif model_layer['layer'] == 'conv1d':
		if 'W' in model_layer:
			W = model_layer['W']
		else:
			W = init.HeUniform()
		if 'pad' in model_layer:
			pad = model_layer['pad']
		else:
			pad = 'valid'
		if 'stride' in model_layer:
			stride = (model_layer['stride'], 1)
		else:
			stride = (1,1)
		if isinstance(model_layer['filter_size'], (list, tuple)):
			filter_size = model_layer['filter_size']
		else:
			filter_size = (model_layer['filter_size'], 1)
		network = layers.Conv2DLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=filter_size,
											  W=W,
											  b=None, 
											  pad=pad,
											  stride=stride,
											  nonlinearity=None)
	# 2D convolution layer
	elif (model_layer['layer'] == 'conv2d') | (model_layer['layer'] == 'convolution'):
		if 'W' in model_layer:
			W = model_layer['W']
		else:
			W = init.HeUniform()
		if 'pad' in model_layer:
			pad = model_layer['pad']
		else:
			pad = 'valid'
		if 'stride' in model_layer:
			if isinstance(model_layer['stride'], (list, tuple)):
				stride = model_layer['stride']
			else:
				stride = (model_layer['stride'], model_layer['stride'])
		else:
			stride = (1,1)
		if isinstance(model_layer['filter_size'], (list, tuple)):
			filter_size = model_layer['filter_size']
		else:
			filter_size = (model_layer['filter_size'], model_layer['filter_size'])
		network = layers.Conv2DLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=model_layer['filter_size'],
											  W=W,
											  b=None, 
											  pad=pad,
											  stride=stride,
											  nonlinearity=None)

	# 1D convolution layer
	elif model_layer['layer'] == 'transpose-conv1d':
		if 'W' in model_layer:
			W = model_layer['W']
		else:
			W = init.HeUniform()
		if 'pad' in model_layer:
			pad = model_layer['pad']
		else:
			pad = 'valid'
		if 'stride' in model_layer:
			stride = (model_layer['stride'], 1)
		else:
			stride = (1,1)
		if isinstance(model_layer['filter_size'], (list, tuple)):
			filter_size = model_layer['filter_size']
		else:
			filter_size = (model_layer['filter_size'], 1)
		network = layers.TransposedConv2DLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=model_layer['filter_size'],
											  W=W,
											  b=None, 
											  crop=pad,
											  stride=stride,
											  nonlinearity=None)
	# 2D convolution layer
	elif (model_layer['layer'] == 'transpose-conv2d') | (model_layer['layer'] == 'transpose-convolution'):
		if 'W' in model_layer:
			W = model_layer['W']
		else:
			W = init.HeUniform()
		if 'pad' in model_layer:
			pad = model_layer['pad']
		else:
			pad = 'valid'
		if 'stride' in model_layer:
			if isinstance(model_layer['stride'], (list, tuple)):
				stride = model_layer['stride']
			else:
				stride = (model_layer['stride'], model_layer['stride'])
		else:
			stride = (1,1)
		if isinstance(model_layer['filter_size'], (list, tuple)):
			filter_size = model_layer['filter_size']
		else:
			filter_size = (model_layer['filter_size'], model_layer['filter_size'])
		network = layers.TransposedConv2DLayer(network_last, num_filters=model_layer['num_filters'],
											  filter_size=model_layer['filter_size'],
											  W=W,
											  b=None, 
											  crop=pad,
											  stride=stride,
											  nonlinearity=None)

	# concat layer
	elif model_layer['layer'] == 'concat':
		network = layers.ConcatLayer([network_last, model_layer['concat']])

	# element-size sum layer
	elif model_layer['layer'] == 'sum':
		network = layers.ElemwiseSumLayer([network_last, model_layer['sum']])

	# reshape layer
	elif model_layer['layer'] == 'reshape':
		network = layers.ReshapeLayer(network_last, model_layer['reshape'])

	# gaussian noise layer
	elif model_layer['layer'] == 'noise':
		if 'sigma' in model_layer:
			sigma = model_layer['sigma']
		else:
			sigma = 0.1
		network = layers.GaussianNoiseLayer(network_last, sigma=sigma)  

	# lstm layer
	elif model_layer['layer'] == 'lstm':
		if 'grad_clipping' in model_layer:
			grad_clipping = model_layer['grad_clipping']
		else:
			grad_clipping = 0

		network = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
											grad_clipping=grad_clipping)

	# bi-directional lstm layer
	elif model_layer['layer'] == 'bi-lstm':
		if 'grad_clipping' in model_layer:
			grad_clipping = model_layer['grad_clipping']
		else:
			grad_clipping = 0

		l_forward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
											grad_clipping=grad_clipping)
		l_backward = layers.LSTMLayer(network_last, num_units=model_layer['num_units'], 
											grad_clipping=grad_clipping, 
											backwards=True)
		network = layers.ConcatLayer([l_forward, l_backward])

	# highway network layer
	elif model_layer['layer'] == 'highway':
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

	elif activation == 'leaky_relu':
			network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.leaky_rectify)
	
	elif activation == 'veryleakyrelu':
			network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.very_leaky_rectify)
		
	elif activation == 'relu':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.rectify)

	elif activation == 'orthogonal':
		network = layers.NonlinearityLayer(network_last, nonlinearity=nonlinearities.orthogonal)
		
	return network



#---------------------------------------------------------------------------------------------------------
# variational sampling

class VariationalSampleLayer(layers.MergeLayer):
    def __init__(self, incoming_mu, incoming_logsigma, **kwargs):
        super(VariationalSampleLayer, self).__init__(incomings=[incoming_mu, incoming_logsigma], **kwargs)
        self.srng = RandomStreams(seed=234)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.srng.normal(shape, avg=0.0, std=1).astype(theano.config.floatX)    


#--------------------------------------------------------------------------------------------------------------------
# highway MLP layer

class MultiplicativeGatingLayer(layers.MergeLayer):
	def __init__(self, gate, input1, input2, **kwargs):
		incomings = [gate, input1, input2]
		super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
	
	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]
	
	def get_output_for(self, inputs, **kwargs):
		return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]



def highway_dense(incoming, W_dense=init.Orthogonal(), b_dense=init.Constant(0.0),
				  W_gate=init.Orthogonal(), b_gate=init.Constant(-4.0),
				  nonlinearity=nonlinearities.rectify, **kwargs):

	num_inputs = int(np.prod(incoming.output_shape[1:]))

	# regular layer
	l_dense = layers.DenseLayer(incoming, num_units=num_inputs, W=W_dense, b=b_dense, nonlinearity=nonlinearity)

	# gate layer
	l_gate = layers.DenseLayer(incoming, num_units=num_inputs, W=W_gate, b=b_gate, nonlinearity=nonlinearities.sigmoid)
	
	return MultiplicativeGatingLayer(gate=l_gate, input1=l_dense, input2=incoming)


#--------------------------------------------------------------------------------------------------------------------
# residual learning layer

def conv1D_residual(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify, dropout=None):


	if not isinstance(filter_size, (list, tuple)):
		filter_size = (filter_size, 1)

	# original residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeUniform(), b=None, nonlinearity=None, pad='same')
	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	if dropout:
		net[name+'_dropout'] = layers.DropoutLayer(net[name+'_1resid_active'], p=dropout)
		last_layer = name+'_dropout'
	else:
		last_layer = name+'_1resid_active'

	# bottleneck residual layer
	net[name+'_2resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeUniform(), b=None, nonlinearity=None, pad='same')
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	return net



def conv2D_residual(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify, dropout=None):

	if not isinstance(filter_size, (list, tuple)):
		filter_size = (filter_size, filter_size)

	# original residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeUniform(), b=None, nonlinearity=None, pad='same')
	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	if dropout:
		net[name+'_dropout'] = layers.DropoutLayer(net[name+'_1resid_active'], p=dropout)
		last_layer = name+'_dropout'
	else:
		last_layer = name+'_1resid_active'
		
	# bottleneck residual layer
	net[name+'_2resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeUniform(), b=None, nonlinearity=None, pad='same')
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	return net



def dense_residual(net, last_layer, name, nonlinearity=nonlinearities.rectify):

	# initial residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_units = shape[1]

	# original residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid'] = layers.DenseLayer(net[last_layer], num_units=num_units, 
											W=init.HeUniform(), b=None, nonlinearity=None)
	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	if dropout:
		net[name+'_dropout'] = layers.DropoutLayer(net[name+'_1resid_active'], p=dropout)
		last_layer = name+'_dropout'
	else:
		last_layer = name+'_1resid_active'
		
	# bottleneck residual layer
	net[name+'_2resid'] = layers.DenseLayer(net[last_layer], num_units=num_units, 
											 W=init.HeUniform(), b=None, nonlinearity=None)
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	return net

#--------------------------------------------------------------------------------------------------------------------
# Denoising layer for ladder network

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

#--------------------------------------------------------------------------------------------------------------------
# decorrelation layer 

class DecorrLayer():
	def __init__(self, incoming, L, **kwargs):
		self.L = L
		super(DecorrLayer, self).__init__(incoming, L, **kwargs)

	def get_output_shape_for(self, input_shape):
		return input_shape[0]

	def get_output_for(self, input, **kwargs):
		
		return T.dot(self.L, input.T).T



