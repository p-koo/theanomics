
import sys
import os
import time
import numpy as np
import h5py

from six.moves import cPickle
import scipy.io
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform, Constant
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.updates import nesterov_momentum, adagrad, rmsprop, total_norm_constraint

sys.path.append('/home/peter/GenomeMotifs/src')
from neuralnetwork import NeuralNetworkModel
from data_utils import load_MotifSimulation
from utils import make_directory, calculate_metrics

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=0)
num_data, dim, sequence_length,_ = train[0].shape
num_labels = train[1].shape[1]   # number of labels (output units)

#-------------------------------------------------------------------------------------


input_var = T.tensor4('inputs')
target_var = T.dmatrix('targets')

network = lasagne.layers.InputLayer(shape=(None,  dim, sequence_length, 1),
									input_var=input_var)


network = lasagne.layers.Conv2DLayer(
		network, num_filters=200, filter_size=(8, 1),
		nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 1))


network = lasagne.layers.Conv2DLayer(
		network, num_filters=200, filter_size=(8, 1),
		nonlinearity=lasagne.nonlinearities.rectify)
network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 1))


network = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(network, p=.5),
		num_units=200,
		nonlinearity=lasagne.nonlinearities.rectify)


network = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(network, p=.5),
		num_units=num_labels,
		nonlinearity=lasagne.nonlinearities.sigmoid)


prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)
#updates = lasagne.updates.nesterov_momentum(
#		loss, params, learning_rate=0.01, momentum=0.9)
updates = lasagne.updates.adam(loss, params)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()

#test_prediction = T.ge(test_prediction, .5)
#test_acc = T.mean(T.eq(test_prediction, target_var))


train_fun = theano.function([input_var, target_var], [loss, prediction], updates=updates)
test_fun = theano.function([input_var, target_var], [test_loss, test_prediction])


#-------------------------------------------------------------------------------------
def batch_generator(X, y, N):
	while True:
		idx = np.random.choice(len(y), N)
		yield X[idx].astype('float32'), y[idx].astype('int32')


def early_stopping(valid_memory, patience):
	min_loss = min(valid_memory)
	min_epoch = valid_memory.index(min_loss)
	current_loss = valid_memory[-1]
	current_epoch = len(valid_memory)
	status = 1
	if min_loss > current_loss:
		if patience + min_epoch < current_epoch:
			status = 0
	return status



batch_size = 500
num_epochs = 100

num_train_batches = len(train[0]) // batch_size
num_valid_batches = len(valid[0]) // batch_size

train_batches = batch_generator(train[0], train[1], batch_size)
valid_batches = batch_generator(valid[0], valid[1], batch_size)


print("Starting training...")
patience = 3
bar_length = 20
valid_memory = []
performance = []
for epoch in range(num_epochs):
	start_time = time.time()
	train_loss = 0
	train_accuracy = 0
	for index in range(num_train_batches):
		X, y = next(train_batches)
		loss, prediction = train_fun(X, y)
		train_loss += loss
		
		# progress bar
		remaining_time = (time.time()-start_time)*(num_train_batches-index-1)/(index+1)
		percent = (index+1.)/num_train_batches
		progress = '='*int(round(percent*bar_length))
		spaces = ' '*int(bar_length-round(percent*bar_length))
		sys.stdout.write("\rEpoch %d [%s] %.1f%% -- est.time=%ds -- loss=%.3f" \
		%(epoch+1, progress+spaces, percent*100, remaining_time, train_loss/(index+1)))
		sys.stdout.flush()
	sys.stdout.write("\n")

	train_loss /= num_train_batches
	train_accuracy /= num_train_batches

	"""
	valid_loss = 0
	valid_accuracy = 0
	for _ in range(num_valid_batches):
		X, y = next(valid_batches)
		loss, prediction = test_fun(X, y)
		valid_loss += loss
		valid_accuracy += accuracy
	valid_loss /= num_valid_batches
	valid_accuracy /= num_valid_batches
	valid_memory.append(valid_loss)
	"""

	valid_loss, valid_prediction = test_fun(valid[0].astype('float32'), valid[1].astype('int32'))
	accuracy, auc_roc, auc_pr = calculate_metrics(valid[1], valid_prediction)
	print [accuracy, auc_roc, auc_pr]

	# stor training performance info
	performance.append([train_loss, train_accuracy, valid_loss, accuracy])
	"""
	print("Epoch {} of {} took {:.3f}s".format(
		epoch+1, num_epochs, time.time() - start_time))
	print("  training loss:\t\t{:.6f}".format(train_loss))
	print("  validation loss:\t\t{:.6f}".format(valid_loss))
	status = early_stopping(valid_memory, patience)
	if not status:
		print "Patience ran out... Early stopping."
		break
	"""

# get test loss and accuracy
num_test_batches = len(test[0]) // batch_size
test_batches = batch_generator(test[0], test[1], batch_size)
test_loss = 0
test_accuracy = 0
for _ in range(num_test_batches):
	X, y = next(test_batches)
	loss, accuracy = test_fun(X, y)
	test_loss += loss
	test_accuracy += accuracy
test_loss /= num_test_batches
test_accuracy /= num_test_batches

print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_loss))
print("  test accuracy:\t\t{:.2f} %".format(test_accuracy*100))



