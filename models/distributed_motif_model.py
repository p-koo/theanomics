from lasagne import layers, nonlinearities, init
import theano.tensor as T

def model(shape, num_labels):

	def residual_block(net, last_layer, name, filter_size, nonlinearity=nonlinearities.rectify):

	    # original residual unit
	    shape = layers.get_output_shape(net[last_layer])
	    num_filters = shape[1]

	    net[name+'_1resid'] = layers.Conv2DLayer(net[last_layer], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
	                     W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	    net[name+'_1resid_norm'] = layers.BatchNormLayer(net[name+'_1resid'])
	    net[name+'_1resid_active'] = layers.NonlinearityLayer(net[name+'_1resid_norm'], nonlinearity=nonlinearity)

	    net[name+'_1resid_dropout'] = layers.DropoutLayer(net[name+'_1resid_active'], p=0.1)

	    # bottleneck residual layer
	    net[name+'_2resid'] = layers.Conv2DLayer(net[name+'_1resid_dropout'], num_filters=num_filters, filter_size=filter_size, stride=(1, 1),    # 1000
	                     W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	    net[name+'_2resid_norm'] = layers.BatchNormLayer(net[name+'_2resid'])

	    # combine input with residuals
	    net[name+'_residual'] = layers.ElemwiseSumLayer([net[last_layer], net[name+'_2resid_norm']])
	    net[name+'_resid'] = layers.NonlinearityLayer(net[name+'_residual'], nonlinearity=nonlinearity)

	    return net



	input_var = T.tensor4('inputs')
	target_var = T.dmatrix('targets')
		
	
	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=30, filter_size=(19, 1), stride=(1, 1),    # 246
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout1'] = layers.DropoutLayer(net['conv1_active'], p=0.1)
	net = residual_block(net, 'conv1_dropout1', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_2_resid'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=100, filter_size=(8, 1), stride=(5, 1), # 61
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout1'] = layers.DropoutLayer(net['conv2_active'], p=0.1)
	net = residual_block(net, 'conv2_dropout1', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_2_resid'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=250, filter_size=(8, 1), stride=(5, 1),  # 15
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout1'] = layers.DropoutLayer(net['conv3_active'], p=0.1)
	net = residual_block(net, 'conv3_dropout1', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_2_resid'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=500, filter_size=(8, 1), stride=(5, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout1'] = layers.DropoutLayer(net['conv4_active'], p=0.1)
	net = residual_block(net, 'conv4_dropout1', 'conv4_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_2_resid'], p=0.1)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=num_labels, filter_size=(7, 1), stride=(1, 1), # 3
	                                    W=init.HeNormal(), b=init.Constant(0.05), nonlinearity=None, pad='valid')
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5'], nonlinearity=nonlinearities.sigmoid)
	
	net['output'] = layers.ReshapeLayer(net['conv5_active'], [-1, num_labels])



	# optimization parameters
	optimization = {"objective": "binary",
	                "optimizer": "adam",
	                "learning_rate": 0.001,
	                "l2": 1e-5
	                }

	return net, input_var, target_var, optimization


"""


	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=25, filter_size=(19, 1), stride=(1, 1),    # 246
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout1'] = layers.DropoutLayer(net['conv1_active'], p=0.1)
	net = residual_block(net, 'conv1_dropout1', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_2_resid'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=50, filter_size=(5, 1), stride=(2, 1), # 61
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout1'] = layers.DropoutLayer(net['conv2_active'], p=0.1)
	net = residual_block(net, 'conv2_dropout1', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_2_resid'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=100, filter_size=(5, 1), stride=(2, 1),  # 15
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout1'] = layers.DropoutLayer(net['conv3_active'], p=0.1)
	net = residual_block(net, 'conv3_dropout1', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_2_resid'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=200, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout1'] = layers.DropoutLayer(net['conv4_active'], p=0.1)
	net = residual_block(net, 'conv4_dropout1', 'conv4_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_2_resid'], p=0.1)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=400, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net['conv5_dropout1'] = layers.DropoutLayer(net['conv5_active'], p=0.1)
	net = residual_block(net, 'conv5_dropout1', 'conv45_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv5_dropout'] = layers.DropoutLayer(net['conv54_2_resid'], p=0.1)

	net['conv6'] = layers.Conv2DLayer(net['conv5_dropout'], num_filters=600, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv6_norm'] = layers.BatchNormLayer(net['conv6'])
	net['conv6_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv6_dropout1'] = layers.DropoutLayer(net['conv6_active'], p=0.1)
	net = residual_block(net, 'conv6_dropout1', 'conv6_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv6_dropout'] = layers.DropoutLayer(net['conv6_2_resid'], p=0.1)

	net['conv7'] = layers.Conv2DLayer(net['conv6_dropout'], num_filters=800, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv7_norm'] = layers.BatchNormLayer(net['conv7'])
	net['conv7_active'] = layers.NonlinearityLayer(net['conv7_norm'], nonlinearity=nonlinearities.rectify)
	net['conv7_dropout'] = layers.DropoutLayer(net['conv7_active'], p=0.1)

	net['dense'] = layers.DenseLayer(net['conv7_dropout'], num_units=500, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.rectify)
	net['dense_dropout'] = layers.DropoutLayer(net['dense'], p=0.2)

	net['output'] = layers.DenseLayer(net['dense_dropout'], num_units=num_labels, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.sigmoid)



	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=25, filter_size=(19, 1), stride=(1, 1),    # 246
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout1'] = layers.DropoutLayer(net['conv1_active'], p=0.1)
	net = residual_block(net, 'conv1_dropout1', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_2_resid'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=50, filter_size=(5, 1), stride=(5, 1), # 61
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout1'] = layers.DropoutLayer(net['conv2_active'], p=0.1)
	net = residual_block(net, 'conv2_dropout1', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_2_resid'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=100, filter_size=(5, 1), stride=(5, 1),  # 15
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout1'] = layers.DropoutLayer(net['conv3_active'], p=0.1)
	net = residual_block(net, 'conv3_dropout1', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_2_resid'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=200, filter_size=(5, 1), stride=(5, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout1'] = layers.DropoutLayer(net['conv4_active'], p=0.1)
	net = residual_block(net, 'conv4_dropout1', 'conv4_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_2_resid'], p=0.1)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=200, filter_size=(5, 1), stride=(4, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net['conv5_dropout'] = layers.DropoutLayer(net['conv5_active'], p=0.1)

	net['dense'] = layers.DenseLayer(net['conv5_dropout'], num_units=500, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.rectify)
	net['dense_dropout'] = layers.DropoutLayer(net['dense'], p=0.2)

	net['output'] = layers.DenseLayer(net['dense_dropout'], num_units=num_labels, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.sigmoid)




	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=30, filter_size=(19, 1), stride=(1, 1),    # 1000
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout1'] = layers.DropoutLayer(net['conv1_active'], p=0.1)
	net = residual_block(net, 'conv1_dropout1', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_pool'] = layers.MaxPool2DLayer(net['conv1_2_resid'], pool_size=(4, 1), stride=(4, 1), ignore_border=False) # 250
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_pool'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=64, filter_size=(5, 1), stride=(3, 1), # 246
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout1'] = layers.DropoutLayer(net['conv2_active'], p=0.1)
	net = residual_block(net, 'conv2_dropout1', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_pool'] = layers.MaxPool2DLayer(net['conv2_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 82
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_pool'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=128, filter_size=(5, 1), stride=(1, 1),  # 78
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout1'] = layers.DropoutLayer(net['conv3_active'], p=0.1)
	net = residual_block(net, 'conv3_dropout1', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_pool'] = layers.MaxPool2DLayer(net['conv3_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 26
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_pool'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=256, filter_size=(6, 1), stride=(1, 1), # 21
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout1'] = layers.DropoutLayer(net['conv4_active'], p=0.1)
	net = residual_block(net, 'conv4_dropout1', 'conv4_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv4_pool'] = layers.MaxPool2DLayer(net['conv4_2_resid'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 7
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_pool'], p=0.1)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=512, filter_size=(5, 1), stride=(1, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	#net['conv5_dropout1'] = layers.DropoutLayer(net['conv5_active'], p=0.1)
	#net = residual_block(net, 'conv5_dropout1', 'conv5_2', filter_size=(3,1), nonlinearity=nonlinearities.rectify)
	net['conv5_pool'] = layers.MaxPool2DLayer(net['conv5_active'], pool_size=(3, 1), stride=(3, 1), ignore_border=False) # 7
	net['conv5_dropout'] = layers.DropoutLayer(net['conv5_pool'], p=0.1)

	net['conv6'] = layers.Conv2DLayer(net['conv5_dropout'], num_filters=num_labels, filter_size=(1, 1), stride=(1, 1),
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6'], nonlinearity=nonlinearities.sigmoid)

	net['output'] = layers.ReshapeLayer(net['conv6_active'], [-1, num_labels])

	net = {}
	net['input'] = layers.InputLayer(input_var=input_var, shape=shape)
	net['conv1'] = layers.Conv2DLayer(net['input'], num_filters=25, filter_size=(19, 1), stride=(1, 1),    # 246
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='same')
	net['conv1_norm'] = layers.BatchNormLayer(net['conv1'])
	net['conv1_active'] = layers.NonlinearityLayer(net['conv1_norm'], nonlinearity=nonlinearities.rectify)
	net['conv1_dropout1'] = layers.DropoutLayer(net['conv1_active'], p=0.1)
	net = residual_block(net, 'conv1_dropout1', 'conv1_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv1_dropout'] = layers.DropoutLayer(net['conv1_2_resid'], p=0.1)

	net['conv2'] = layers.Conv2DLayer(net['conv1_dropout'], num_filters=50, filter_size=(5, 1), stride=(2, 1), # 61
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv2_norm'] = layers.BatchNormLayer(net['conv2'])
	net['conv2_active'] = layers.NonlinearityLayer(net['conv2_norm'], nonlinearity=nonlinearities.rectify)
	net['conv2_dropout1'] = layers.DropoutLayer(net['conv2_active'], p=0.1)
	net = residual_block(net, 'conv2_dropout1', 'conv2_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv2_dropout'] = layers.DropoutLayer(net['conv2_2_resid'], p=0.1)

	net['conv3'] = layers.Conv2DLayer(net['conv2_dropout'], num_filters=100, filter_size=(5, 1), stride=(2, 1),  # 15
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv3_norm'] = layers.BatchNormLayer(net['conv3'])
	net['conv3_active'] = layers.NonlinearityLayer(net['conv3_norm'], nonlinearity=nonlinearities.rectify)
	net['conv3_dropout1'] = layers.DropoutLayer(net['conv3_active'], p=0.1)
	net = residual_block(net, 'conv3_dropout1', 'conv3_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv3_dropout'] = layers.DropoutLayer(net['conv3_2_resid'], p=0.1)

	net['conv4'] = layers.Conv2DLayer(net['conv3_dropout'], num_filters=150, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv4_norm'] = layers.BatchNormLayer(net['conv4'])
	net['conv4_active'] = layers.NonlinearityLayer(net['conv4_norm'], nonlinearity=nonlinearities.rectify)
	net['conv4_dropout1'] = layers.DropoutLayer(net['conv4_active'], p=0.1)
	net = residual_block(net, 'conv4_dropout1', 'conv4_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv4_dropout'] = layers.DropoutLayer(net['conv4_2_resid'], p=0.1)

	net['conv5'] = layers.Conv2DLayer(net['conv4_dropout'], num_filters=200, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv5_norm'] = layers.BatchNormLayer(net['conv5'])
	net['conv5_active'] = layers.NonlinearityLayer(net['conv5_norm'], nonlinearity=nonlinearities.rectify)
	net['conv5_dropout1'] = layers.DropoutLayer(net['conv5_active'], p=0.1)
	net = residual_block(net, 'conv5_dropout1', 'conv5_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv5_dropout'] = layers.DropoutLayer(net['conv5_2_resid'], p=0.1)

	net['conv6'] = layers.Conv2DLayer(net['conv5_dropout'], num_filters=250, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv6_norm'] = layers.BatchNormLayer(net['conv6'])
	net['conv6_active'] = layers.NonlinearityLayer(net['conv6_norm'], nonlinearity=nonlinearities.rectify)
	net['conv6_dropout1'] = layers.DropoutLayer(net['conv6_active'], p=0.1)
	net = residual_block(net, 'conv6_dropout1', 'conv6_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv6_dropout'] = layers.DropoutLayer(net['conv6_2_resid'], p=0.1)

	net['conv7'] = layers.Conv2DLayer(net['conv6_dropout'], num_filters=350, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv7_norm'] = layers.BatchNormLayer(net['conv7'])
	net['conv7_active'] = layers.NonlinearityLayer(net['conv7_norm'], nonlinearity=nonlinearities.rectify)
	net['conv7_dropout1'] = layers.DropoutLayer(net['conv7_active'], p=0.1)
	net = residual_block(net, 'conv7_dropout1', 'conv7_2', filter_size=(5,1), nonlinearity=nonlinearities.rectify)
	net['conv7_dropout'] = layers.DropoutLayer(net['conv7_2_resid'], p=0.1)

	net['conv8'] = layers.Conv2DLayer(net['conv7_dropout'], num_filters=500, filter_size=(5, 1), stride=(2, 1), # 3
	                                    W=init.HeNormal(), b=None, nonlinearity=None, pad='valid')
	net['conv8_norm'] = layers.BatchNormLayer(net['conv8'])
	net['conv8_active'] = layers.NonlinearityLayer(net['conv8_norm'], nonlinearity=nonlinearities.rectify)
	net['conv8_dropout'] = layers.DropoutLayer(net['conv8_active'], p=0.1)

	net['dense'] = layers.DenseLayer(net['conv8_dropout'], num_units=500, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.rectify)
	net['dense_dropout'] = layers.DropoutLayer(net['dense'], p=0.2)

	net['output'] = layers.DenseLayer(net['dense_dropout'], num_units=num_labels, W=init.HeNormal(), 
	                                  b=init.Constant(0.05), nonlinearity=nonlinearities.sigmoid)




"""