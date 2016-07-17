from __future__ import division, print_function, absolute_import
import tflearn

import sys, os
import numpy as np
from lasagne import layers, init, nonlinearities, utils, regularization, objectives, updates
from six.moves import cPickle
sys.setrecursionlimit(10000)
from scipy import stats
import time
import h5py


# data file and output files
outputname = 'highway_final'
name = 'train_norm.hd5f'
datapath='/home/peter/Data/CMAP'
trainmat = h5py.File(os.path.join(datapath, name), 'r')
filepath = os.path.join(datapath, 'Results', outputname)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 970])
dense1 = tflearn.fully_connected(input_layer, 970, activation='elu',
								 regularizer='L2', weight_decay=0.001)
								  
#install a deep network of highway layers
highway = dense1                              
for i in range(10):
	highway = tflearn.highway(highway, 970, activation='elu',
							  regularizer='L2', weight_decay=0.001, transform_dropout=0.8)
							  
							  
output = tflearn.fully_connected(highway, 11350, activation='linear')

#sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
#net = tflearn.regression(net, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.1)
network = tflearn.regression(output, optimizer='adam', metric='R2', 
									loss='mean_square', learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path=filepath, keep_checkpoint_every_n_hours=1, 
												random_seed=247, tensorboard_verbose=0)

# Load a model
#model.load('highway.tflearn')

# train model
batch_size = 50     
bar_length = 30     
num_epochs = 500   
verbose = 1
train_performance = []
valid_performance = []
for epoch in range(num_epochs):
	sys.stdout.write("\rEpoch %d \n"%(epoch+1))
	
	train_loss = 0
	for i in range(2):
		sys.stdout.write("\r  File %d \n"%(i+1))
		landmark= np.array(trainmat['landmark'+str(i)]).astype(np.float32)
		nonlandmark = np.array(trainmat['nonlandmark'+str(i)]).astype(np.float32)
		model.fit(landmark, nonlandmark, n_epoch=20, snapshot_epoch=True, 
						batch_size=100, shuffle=True, show_metric=True, run_id="highway_dense_model")

	# test current model with cross-validation data and store results
	landmark= np.array(trainmat['landmark2']).astype(np.float32)
	nonlandmark = np.array(trainmat['nonlandmark2']).astype(np.float32)
	landmark_test = np.array(trainmat['landmark3']).astype(np.float32)
	nonlandmark_test = np.array(trainmat['nonlandmark3']).astype(np.float32)
	model.fit(landmark, nonlandmark, n_epoch=20, validation_set=(landmark_test, nonlandmark_test), snapshot_epoch=True, 
				batch_size=100, shuffle=True, show_metric=True, run_id="highway_dense_model")

	# Save a model
	model.save('highway.tflearn')



	

