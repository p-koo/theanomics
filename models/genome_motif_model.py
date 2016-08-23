#/bin/python
import sys
sys.path.append('..')
import theano.tensor as T
from lasagne.init import Constant, Normal, Uniform, GlorotNormal
from lasagne.init import GlorotUniform, HeNormal, HeUniform
from lasagne import layers, nonlinearities, init
from src.build_network import build_network
import numpy as np

def residual_block(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify):

	"""
	# original residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	# bottleneck residual layer
	net[name+'_2resid'] = layers.Conv2DLayer(net[name+'_1resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	"""
	# new residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]

	net[name+'_1resid_norm'] = layers.BatchNormLayer(net[last_layer])
	net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)
	net[name+'_1resid'] = layers.Conv2DLayer(net[name+'_1resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')

	# bottleneck residual layer
	net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	net[name+'_2resid_active'] = layers.NonlinearityLayer(net[name+'_2resid_norm'], nonlinearity=nonlinearity)
	net[name+'_2resid'] = layers.Conv2DLayer(net[name+'_2resid_active'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')

	# combine input with residuals
	net[name+'_resid'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid']])
	
	return net


def residual_bottleneck(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify):

	# initial residual unit
	shape = layers.get_output_shape(net[last_layer])
	num_filters = shape[1]
	reduced_filters = np.round(num_filters/2)

	# 1st residual layer
	net[name] = layers.Conv2DLayer(net[last_layer], num_filters=reduced_filters, filter_size=(1,1), stride=(1, 1),  
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_norm'] = layers.BatchNormLayer(net[name])
	net[name+'_active'] = layers.NonlinearityLayer(net[name+'_norm'], nonlinearity=nonlinearities.rectify)

	net[name+'_resid'] = layers.Conv2DLayer(net[name+'_active'], num_filters=reduced_filters, filter_size=filter_size, stride=(1, 1),   
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_resid_norm'] = layers.BatchNormLayer(net[name+'_resid'])
	net[name+'_resid_active'] = layers.NonlinearityLayer(net[name+'_resid_norm'], nonlinearity=nonlinearity)

	# bottleneck residual layer
	net[name+'_bottle'] = layers.Conv2DLayer(net[name+'_resid_active'], num_filters=num_filters, filter_size=(1,1), stride=(1, 1),    
					 W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net[name+'_bottle_norm'] = layers.BatchNormLayer(net[name+'_bottle'])

	# combine input with residuals
	net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_bottle_norm']])
	net[name+'_residual_active'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	return net

	
def model(shape, num_labels):

	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')

	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(8, 1), stride=(1, 1),    # 993
					 					W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_block(net, 'conv1_active', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 331
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=64, filter_size=(8, 1), stride=(1, 1), # 324
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_block(net, 'conv2_active', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 108
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.2)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=128, filter_size=(7, 1), stride=(1, 1),  # 102
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_block(net, 'conv3_active', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 34
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.2)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=256, filter_size=(5, 1), stride=(1, 1), # 30
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_block(net, 'conv4_active', 'conv4_2', filter_size=(1,1), nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 10
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.2)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 6
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_block(net, 'conv5_active', 'conv5_2', filter_size=(1,1), nonlinearity=nonlinearities.rectify)
	net['conv5_pool'] = layers.MaxPool2DLayer(net['conv5_2_resid'], pool_size=(6, 1), stride=(6, 1), ignore_border=False) # 1

	net['conv6'] = layers.Conv2DLayer(net['conv5_pool'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv6_active'], [-1, num_labels])




	# optimization parameters
	optimization = {"objective": "binary",
					"optimizer": "adam",
					"learning_rate": 0.001,                 
					"beta1": .9,
					"beta2": .999,
					"epsilon": 1e-6
					#"l1": 1e-9,
					#"l2": 1e-9
					}

	return net, input_var, target_var, optimization

"""

	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(8, 1), stride=(1, 1),    # 993
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 331
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=64, filter_size=(8, 1), stride=(1, 1), # 324
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 108
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.2)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=128, filter_size=(7, 1), stride=(1, 1),  #102
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 34
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.2)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=256, filter_size=(5, 1), stride=(1, 1), # 30
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 10
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.2)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 6
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net['conv5_pool'] = layers.MaxPool2DLayer(net['conv5_active'], pool_size=(6, 1), stride=(6, 1), ignore_border=False) # 1

	net['conv6'] = layers.Conv2DLayer(net['conv5_pool'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv6_active'], [-1, num_labels])



	# residual model
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(8, 1), stride=(1, 1),    # 993
					 					W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_bottleneck(net, 'conv1_active', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 331

	net['conv2'] = layers.Conv2DLayer(net['conv1_pool'], num_filters=64, filter_size=(8, 1), stride=(1, 1), # 324
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_bottleneck(net, 'conv2_active', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 108

	net['conv3'] = layers.Conv2DLayer(net['conv2_pool'], num_filters=128, filter_size=(7, 1), stride=(1, 1),  # 102
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_bottleneck(net, 'conv3_active', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 34

	net['conv4'] = layers.Conv2DLayer(net['conv3_pool'], num_filters=256, filter_size=(5, 1), stride=(1, 1), # 30
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_bottleneck(net, 'conv4_active', 'conv4_2', filter_size=(1,1), nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 10

	net['conv5'] = layers.Conv2DLayer(net['conv4_pool'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 6
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net = residual_bottleneck(net, 'conv5_active', 'conv5_2', filter_size=(1,1), nonlinearity=nonlinearities.rectify)
	net['conv5_pool'] = layers.MaxPool2DLayer(net['conv5_2_resid'], pool_size=(6, 1), stride=(6, 1), ignore_border=False) # 1

	net['conv6'] = layers.Conv2DLayer(net['conv5_pool'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   				W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv6_active'], [-1, num_labels])




	# very shallow
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=160, filter_size=(19, 1), stride=(1, 1),    # 1000
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(40, 1), stride=(40, 1), ignore_border=False) # 25
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 21 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(9, 1), stride=(9, 1), ignore_border=False) # 3
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=2048, filter_size=(3, 1), stride=(1, 1),  #1
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv4_active'], [-1, num_labels])


	# very shallow
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=160, filter_size=(19, 1), stride=(1, 1),    # 1000
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(40, 1), stride=(40, 1), ignore_border=False) # 25
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 21 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(9, 1), stride=(9, 1), ignore_border=False) # 3
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=2048, filter_size=(3, 1), stride=(1, 1),  #1
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv4_active'], [-1, num_labels])


	# shallow
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=160, filter_size=(19, 1), stride=(1, 1),    # 1000
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(40, 1), stride=(40, 1), ignore_border=False) # 25
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 21 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(9, 1), stride=(9, 1), ignore_border=False) # 3
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=2048, filter_size=(3, 1), stride=(1, 1),  #1
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv4_active'], [-1, num_labels])

	# shallow
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=160, filter_size=(13, 1), stride=(1, 1),    # 1000
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(25, 1), stride=(25, 1), ignore_border=False) # 40
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 36 
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(9, 1), stride=(9, 1), ignore_border=False) # 4
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=2048, filter_size=(4, 1), stride=(1, 1),  #30
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_active'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv4_active'], [-1, num_labels])


	# medium model
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(11, 1), stride=(1, 1),    # 990
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(5, 1), stride=(5, 1), ignore_border=False) # 198
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=256, filter_size=(9, 1), stride=(1, 1), # 190
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(5, 1), stride=(5, 1), ignore_border=False) # 38
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=512, filter_size=(9, 1), stride=(1, 1),  #30
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_active'], pool_size=(5, 1), stride=(5, 1), ignore_border=False) # 6
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.3)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=1024, filter_size=(6, 1), stride=(1, 1), # 1
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_active'], p=0.3)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv5_active'], [-1, num_labels])


	# deep model
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=64, filter_size=(11, 1), stride=(1, 1),    # 990
					 W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 330
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=128, filter_size=(7, 1), stride=(1, 1), # 324
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 108
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.3)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=256, filter_size=(7, 1), stride=(1, 1),  #102
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False)  # 34
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.3)
	
	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=512, filter_size=(8, 1), stride=(1, 1), # 27
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 9
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.3)
	
	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=1024, filter_size=(5, 1), stride=(1, 1), #5
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm']= layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net['conv5_dropout'] = layers.DropoutLayer(net['conv5_active'], p=0.3)

	net['conv6'] = layers.Conv2DLayer(net['conv5_dropout'], num_filters=2048, filter_size=(5, 1), stride=(1, 1), #1
						   W=GlorotUniform(), b=None, nonlinearity=None, pad='valid')
	net['conv6_norm'] = layers.BatchNormLayer(net['conv6'])
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6_norm'], nonlinearity=nonlinearities.rectify)
	net['conv6_dropout'] = layers.DropoutLayer(net['conv6_active'], p=0.3)

	net['conv7'] = layers.Conv2DLayer(net['conv6_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
						   W=GlorotUniform(), b=None, nonlinearity=nonlinearities.sigmoid, pad='valid')

	net['output'] = layers.ReshapeLayer(net['conv7'], [-1, num_labels])


	# Basset model
	input_layer = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 200, 
			  'filter_size': (19, 1), # 992 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (3, 1), # 248 
			  'pad': 'valid',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 300, 
			  'filter_size': (9, 1),  # 240
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (4, 1),  # 60
			  'pad': 'valid',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 300,
			  'filter_size': (7, 1), #54 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'norm': 'batch', 
			  'activation': 'relu',
			  'pool_size': (4, 1),  # 18
			  'pad': 'valid',
			  'name': 'conv3'
			  }
	dense1 = {'layer': 'dense', 
			  'num_units': 1000, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'relu',
			  'dropout': 0.5,
			  'name': 'dense1'
			  }
	dense2 = {'layer': 'dense', 
			  'num_units': 1000, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'norm': 'batch', 
			  'activation': 'relu',
			  'dropout': 0.5,
			  'name': 'dense1'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'sigmoid',
			  'name': 'dense'
			  }

	model_layers = [input_layer, conv1, conv2, conv3,  dense1, dense2, output]
	net = build_network(model_layers)

	# DeepSea model
	input_layer = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 240, 
			  'filter_size': (8, 1), # 992 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'activation': 'relu',
			  'pool_size': (4, 1), # 82 
			  'dropout': 0.1,
			  'pad': 'valid',
			  'name': 'conv1'
			  }
	conv2 = {'layer': 'convolution', 
			  'num_filters': 460, 
			  'filter_size': (8, 1),  # 76
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'activation': 'relu',
			  'pool_size': (4, 1),  # 19
			  'dropout': 0.2,
			  'pad': 'valid',
			  'name': 'conv2'
			  }
	conv3 = {'layer': 'convolution', 
			  'num_filters': 980,
			  'filter_size': (8, 1), # 12 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'activation': 'relu',
			  'pool_size': (4, 1),  # 3
			  'dropout': 0.2,
			  'pad': 'valid',
			  'name': 'conv3'
			  }
	dense1 = {'layer': 'dense', 
			  'num_units': 1000, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'relu',
			  'name': 'dense1'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'sigmoid',
			  'name': 'dense'
			  }

	model_layers = [input_layer, conv1, conv2, conv3,  dense1, output]
	net = build_network(model_layers)


	# DeepBind model
	input_layer = {'layer': 'input',
			  'input_var': input_var,
			  'shape': shape,
			  'name': 'input'
			  }
	conv1 = {'layer': 'convolution', 
			  'num_filters': 256, 
			  'filter_size': (23, 1), # 977 
			  'W': GlorotUniform(),
			  'b': Constant(0.05), 
			  'activation': 'relu',
			  'pool_size': (977, 1), # 8
			  'pad': 'valid',
			  'name': 'conv1'
			  }
	dense1 = {'layer': 'dense', 
			  'num_units': 256, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'relu',
			  'name': 'dense1'
			  }
	output = {'layer': 'dense', 
			  'num_units': num_labels, 
			  'W': GlorotUniform(),
			  'b': Constant(0.01), 
			  'activation': 'sigmoid',
			  'name': 'dense'
			  }

	model_layers = [input_layer, conv1, dense1, output]
	net = build_network(model_layers)




	"""