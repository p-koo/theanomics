import sys
import collections

sys.path.append('..')
from deepomics.build_network import build_network 

from lasagne import layers, init, nonlinearities
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def model(shape):

	"""

	placeholders = collections.OrderedDict()
	placeholders['inputs'] = T.dmatrix('inputs')

	num_encode=2
	num_units=200

	input_var = T.dmatrix('inputs')
	network = collections.OrderedDict()
	network['input'] = layers.InputLayer(shape=shape, input_var=placeholders['inputs'])
	network['encode1'] = layers.DenseLayer(network['input'], num_units=200, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.leaky_rectify)
	network['encode2'] = layers.DenseLayer(network['encode1'], num_units=100, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.leaky_rectify)
	network['encode_mu'] = layers.DenseLayer(network['encode2'], num_units=num_encode, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.linear)
	network['encode_logsigma'] = layers.DenseLayer(network['encode2'], num_units=num_encode, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.linear)
	network['Z'] = VariationalSampleLayer(network['encode_mu'], network['encode_logsigma'])

	network['decode1'] = layers.DenseLayer(network['Z'], num_units=100, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.leaky_rectify)
	network['decode2'] = layers.DenseLayer(network['decode1'], num_units=200, W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.leaky_rectify)
	network['X'] = layers.DenseLayer(network['decode2'], num_units=shape[1],  W=init.GlorotUniform(), 
	                                  b=init.Constant(.0), nonlinearity=nonlinearities.sigmoid)
	#network['decode_logsigma'] = layers.DenseLayer(network['decode2'], num_units=x_dim, nonlinearity=nonlinearities.linear)
	#network['X'] = VariationalSampleLayer(network['decode_mu'], network['decode_logsigma'])
	"""
	
	#"""
	# create model
	layer1 = {'layer': 'input',
			  'shape': shape,
			  }
	layer2 = {'layer': 'dense', 
			  'num_units': 512,  
			  'activation': 'leaky_relu',
			  }
	layer3 = {'layer': 'dense', 
			  'num_units': 128,
			  'activation': 'leaky_relu',
			  }
	layer4 = {'layer': 'variational', 
			  'num_units': 10,
			  }
	layer5 = {'layer': 'dense', 
			  'num_units': 128,
			  'activation': 'leaky_relu',
			  }
	layer6 = {'layer': 'dense', 
			  'num_units': 512,  
			  'activation': 'leaky_relu',
			  }
	layer7 = {'layer': 'dense', 
			  'num_units': shape[1],
			  'activation': 'sigmoid',
			  }
		  
	model_layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]
	network, placeholders = build_network(model_layers, shape, supervised=False)
	#"""
	# optimization parameters
	optimization = {"objective": "lower_bound",			  
					'binary': True,
					"optimizer": "adam",
					"learning_rate": 0.001
					#"l2": 1e-6,
					# "l1": 0, 
					}

	return network, placeholders, optimization



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
