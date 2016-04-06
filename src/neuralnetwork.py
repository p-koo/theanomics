
import os
import sys
import numpy as np
from six.moves import cPickle
import theano
import time

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer, NonlinearityLayer
from lasagne.nonlinearities import sigmoid, rectify, softmax, linear, tanh, LeakyRectify, softplus
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.objectives import binary_crossentropy, categorical_crossentropy, squared_error
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.init import Constant
from lasagne.regularization import regularize_layer_params, regularize_network_params
from lasagne.updates import sgd, nesterov_momentum, rmsprop, adagrad, adam, total_norm_constraint
sys.path.append('/home/peter/GenomeMotifs/models')
sys.path.append('/home/peter/GenomeMotifs/src')
from utils import calculate_metrics

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNetworkModel:

	def __init__(self, model_name, shape, num_labels):

		# get model architecture (in models.py file)
		layers, input_var, target_var, optimization = get_model(model_name, shape, num_labels)

		# build model 
		network, train_fun, test_fun = build_model(layers, input_var, target_var, optimization)

		self.model_name = model_name
		self.shape = shape
		self.num_labels = num_labels
		self.layers = layers
		self.input_var = input_var
		self.target_var = target_var
		self.optimization = optimization
		self.network = network
		self.train_fun = train_fun
		self.test_fun = test_fun
		# self.monitor = MonitorTraining()
		self.best_parameters = []
		self.train_monitor = MonitorPerformance(name="train")
		self.test_monitor = MonitorPerformance(name=" test")
		self.valid_monitor = MonitorPerformance(name="cross-validation")

	def get_model_name(self):
		return self.model_name


	def get_model_parameters(self):
		return get_all_param_values(self.network)


	def set_model_parameters(self, all_param_values):
		self.network = set_all_param_values(self.network, all_param_values)


	def save_model_parameters(self, filepath):
		print "saving model parameters to: " + filepath
		all_param_values = get_all_param_values(self.network)
		f = open(filepath, 'wb')
		cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	def set_best_model(self, filepath):
		# load best parameters
		min_cost, min_index = self.valid_monitor.get_min_cost()    
		savepath = filepath + "_" + str(min_index) + ".pickle"
		f = open(savepath, 'rb')
		self.best_parameters = cPickle.load(f)
		f.close()
		self.set_model_parameters(self.best_parameters)

	def test_results(self, test):
		test_cost, test_prediction = self.test_fun(test[0].astype(np.float32), test[1].astype(np.int32)) 
		return test_cost, test_prediction


	def epoch_train(self,  mini_batches, num_batches, verbose):        
		
		# set timer for epoch run
		performance = MonitorPerformance(verbose)
		performance.set_start_time(start_time = time.time())

		# train on mini-batch with random shuffling
		epoch_cost = 0
		for index in range(num_batches):
			X, y = next(mini_batches)
			cost, prediction = self.train_fun(X, y)
			epoch_cost += cost
			performance.progress_bar(index, num_batches, epoch_cost/(index+1))
		print "" 
		return epoch_cost/num_batches


	def train(self, train, valid, test, batch_size=128, num_epochs=500, 
					patience=10, save='all', filepath='.', verbose=1):

		# setup generator for mini-batches
		num_train_batches = len(train[0]) // batch_size
		train_batches = batch_generator(train[0], train[1], batch_size)

		# train model
		for epoch in range(num_epochs):
			if verbose == 1:
				sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

			# training set
			train_cost = self.epoch_train(train_batches, num_train_batches, verbose)
			self.train_monitor.add(train_cost)

			# validation set
			valid_cost, valid_prediction = self.test_results(valid)		
			self.valid_monitor.update(valid_cost, valid_prediction, valid[1])
			self.valid_monitor.print_results("valid", epoch, num_epochs) 

			status = self.valid_monitor.early_stopping(valid_cost, epoch, patience)
			if not status:
				break
			"""                
			# store training performance info
			self.monitor.append_values(train_cost, 'train')
			self.monitor.append_values(valid_cost, 'valid')
			self.monitor.print_results("valid", epoch, num_epochs) 
			print [accuracy, auc_roc, auc_pr]

			self.update_best_model_parameters()

			# check for early stopping
			status = self.monitor.early_stopping(patience)
			if not status:
				break
			"""
			# save model
			if save == 'all':
				savepath = filepath + "_" + str(epoch) + ".pickle"
				self.save_model_parameters(savepath)
					
		# update model with best parameters on cross-validation
		self.set_best_model(filepath)
		savepath = filepath + "_best.pickle"
		self.save_model_parameters(savepath)

		# test performance
		test_cost, test_prediction = self.test_results(test)
		self.test_monitor.update(test_cost, test_prediction, test[1])
		self.test_monitor.print_results("test")   

		# save results
		self.train_monitor.save_performance(filepath)
		self.test_monitor.save_performance(filepath)
		self.valid_monitor.save_performance(filepath)


#----------------------------------------------------------------------------------------------------
# Monitor traning class
#----------------------------------------------------------------------------------------------------

class MonitorPerformance():
	def __init__(self, name = '', verbose=1):
		self.cost = []
		self.metric = np.empty(3)
		self.metric_std = np.empty(3)
		self.verbose = verbose
		self.name = name
		self.roc = []
		self.pr = []

	def set_verbose(self, verbose):
		self.verbose = verbose

	def get_metrics(self, prediction, label):
		mean, std, roc, pr = calculate_metrics(label, prediction)
		self.roc = roc
		self.pr = pr 
		return mean, std, roc, pr

	def add(self, cost, mean=[], std=[]):
		self.cost.append(cost)
		if mean:
			self.metric = np.vstack([self.metric, mean])
		if std:
			self.metric_std = np.vstack([self.metric_std, std])

	def update(self, cost, prediction, label):
		mean, std, roc, pr = self.get_metrics(prediction, label)
		self.add(cost, mean, std)

	def get_mean_values(self):
		results = self.metric[-1,:]
		return results[0], results[1], results[2]

	def get_error_values(self):
		results = self.metric_std[-1,:]
		return results[0], results[1], results[2]

	def get_min_cost(self):
		min_cost = min(self.cost)
		min_index = np.argmin(self.cost)
		return min_cost, min_index

	def early_stopping(self, current_cost, current_epoch, patience):
		min_cost, min_epoch = self.get_min_cost()
		status = True
		if min_cost < current_cost:
			if patience - (current_epoch - min_epoch) < 0:
				status = False
				print "Patience ran out... Early stopping."
		return status

	def set_start_time(self, start_time):
		if self.verbose == 1:
			self.start_time = start_time

	def print_results(self, name, epoch=0, num_epochs=0): 
		if self.verbose == 1:
			accuracy, auc_roc, auc_pr = self.get_mean_values()
			accuracy_std, auc_roc_std, auc_pr_std = self.get_error_values()
			print("  " + name + " cost:\t\t{:.4f}".format(self.cost[-1]/1.))
			print("  " + name + " accuracy:\t{:.4f}+/-{:.4f}".format(accuracy, accuracy_std))
			print("  " + name + " auc-roc:\t{:.4f}+/-{:.4f}".format(auc_roc, auc_roc_std))
			print("  " + name + " auc-pr:\t\t{:.4f}+/-{:.4f}".format(auc_pr, auc_pr_std))
			#print("  " + name + " accuracy:\t{:.2f} %".format(float(accuracy)*100))

	def progress_bar(self, index, num_batches, cost, bar_length=30):
		if self.verbose == 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-index-1)/(index+1)
			percent = (index+1.)/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- cost=%.4f" \
			%(progress+spaces, percent*100, remaining_time, cost))
			sys.stdout.flush()

	def save_performance(self, filepath):
		savepath = filepath + "_" + self.name +".pickle"
		
		f = open(savepath, 'wb')
		cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.cost, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.roc, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.pr, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

#------------------------------------------------------------------------------------------
# Model building functions
#------------------------------------------------------------------------------------------

def get_model(model_name, shape, num_labels):

	# load and build model parameters
	if model_name == "simple_genome_motif_model":
		from simple_genome_motif_model import simple_genome_motif_model
		layers, input_var, target_var, optimization = simple_genome_motif_model(shape, num_labels)

	return layers, input_var, target_var, optimization


def build_model(layers, input_var, target_var, optimization):

	# build model based on layers
	network = build_layers(layers, input_var)

	# build cost function
	cost, prediction = build_cost(network, target_var, objective=optimization["objective"])

	# calculate and clip gradients
	params = get_all_params(network, trainable=True)    
	if "weight_norm" in optimization:
		grad = calculate_gradient(network, cost, params, weight_norm=optimization["weight_norm"])
	else:
		grad = calculate_gradient(network, cost, params)
	  
	# setup parameter updates
	updates = optimizer(grad, params, optimization)

	# test/validation set 
	test_cost, test_prediction = build_cost(network, target_var, objective=optimization["objective"], deterministic=True)

	# weight-decay regularization
	if "l1" in optimization:
		l1_penalty = regularize_network_params(network, l1) * optimization["l1"]
		test_cost += l1_penalty
	if "l2" in optimization:
		l2_penalty = regularize_network_params(network, l2) * optimization["l2"]        
		test_cost += l2_penalty 

	# create theano function
	train_fun = theano.function([input_var, target_var], [cost, prediction], updates=updates)
	test_fun = theano.function([input_var, target_var], [test_cost, test_prediction])

	return network, train_fun, test_fun


def build_layers(layers, input_var):

	# build a single layer
	def single_layer(layer, network=[]):

		# input layer
		if layer['layer'] == 'input':
			network = InputLayer(layer['shape'], input_var=layer['input_var'])

		# dense layer
		elif layer['layer'] == 'dense':
			network = DenseLayer(network,
								num_units=layer['num_units'],
								W=layer['W'],
								b=layer['b'])

		# convolution layer
		elif layer['layer'] == 'convolution':
			network = Conv2DLayer(network,
								  num_filters = layer['num_filters'],
								  filter_size = layer['filter_size'],
								  W=layer['W'],
								  b=layer['b'])
		return network

	# loop to build each layer of network
	network = []
	for layer in layers:

		# create base layer
		network = single_layer(layer, network)
				
		# add Batch normalization layer
		if 'norm' in layer:
			if layer['norm'] == 'batch':
				network = BatchNormLayer(network)

		# add activation layer
		if 'activation' in layer:
			network = activation_layer(network, layer['activation']) 
			
		# add dropout layer
		if 'dropout' in layer:
			DropoutLayer(network, p=layer['dropout'])

		# add max-pooling layer
		if layer['layer'] == 'convolution':            
			network = MaxPool2DLayer(network, pool_size=layer['pool_size'])

	return network


def activation_layer(network, activation):

	if activation == 'prelu':
		network = ParametricRectifierLayer(network,
										  alpha=Constant(0.25),
										  shared_axes='auto')

	elif activation == 'sigmoid':
		network = NonlinearityLayer(network, nonlinearity=sigmoid)

	elif activation == 'softmax':
		network = NonlinearityLayer(network, nonlinearity=softmax)

	elif activation == 'linear':
		network = NonlinearityLayer(network, nonlinearity=linear)

	elif activation == 'tanh':
		network = NonlinearityLayer(network, nonlinearity=tanh)

	elif activation == 'softplus':
		network = NonlinearityLayer(network, nonlinearity=softplus)

	elif activation == 'leakyrelu':
		if 'leakiness' in layer:
			network = NonlinearityLayer(network, nonlinearity=LeakyRectify(leakiness))
		else:
			network = NonlinearityLayer(network, nonlinearity=LeakyRectify(.05))
		
	elif activation == 'relu':
		network = NonlinearityLayer(network, nonlinearity=rectify)
	
	return network

def build_cost(network, target_var, objective, deterministic=False):

	prediction = get_output(network, deterministic=deterministic)
	if objective == 'categorical':
		cost = categorical_crossentropy(prediction, target_var)
	elif objective == 'binary':
		cost = binary_crossentropy(prediction, target_var)
	elif objective == 'mse':
		cost = squared_error(prediction, target_var)
	cost = cost.mean()
	return cost, prediction


def calculate_gradient(network, cost, params, weight_norm=0):

	# calculate gradients
	grad = T.grad(cost, params)

	# gradient clipping option
	if weight_norm > 0:
		grad = total_norm_constraint(grad, weight_norm)

	return grad


def optimizer(grad, params, update_params):

	if update_params['optimizer'] == 'sgd':
		updates = sgd(grad, params, learning_rate=update_params['learning_rate']) 
 
	elif update_params['optimizer'] == 'nesterov_momentum':
		updates = nesterov_momentum(grad, params, 
									learning_rate=update_params['learning_rate'], 
									momentum=update_params['momentum'])
	
	elif update_params['optimizer'] == 'adagrad':
		if "learning_rate" in update_params:
			updates = adagrad(grad, params, 
							  learning_rate=update_params['learning_rate'], 
							  epsilon=update_params['epsilon'])
		else:
			updates = adagrad(grad, params)

	elif update_params['optimizer'] == 'rmsprop':
		if "learning_rate" in update_params:
			updates = rmsprop(grad, params, 
							  learning_rate=update_params['learning_rate'], 
							  rho=update_params['rho'], 
							  epsilon=update_params['epsilon'])
		else:
			updates = rmsprop(grad, params)
	
	elif update_params['optimizer'] == 'adam':
		if "learning_rate" in update_params:
			updates = adam(grad, params, 
							learning_rate=update_params['learning_rate'], 
							beta1=update_params['beta1'], 
							beta2=update_params['beta2'], 
							epsilon=update['epsilon'])
		else:
			updates = adam(grad, params)
  
	return updates



#------------------------------------------------------------------------------------------
# Training functions
#------------------------------------------------------------------------------------------




def batch_generator(X, y, N):
	while True:
		idx = np.random.choice(len(y), N)
		yield X[idx].astype('float32'), y[idx].astype('int32')



"""

def print_results(cost, accuracy, name, epoch=0, num_epochs=0): 
	if num_epochs != 0:   
		sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
	else:
		print('Final Results:')
	print("  " + name + " cost:\t{:.6f}".format(float(cost)))
	print("  " + name + " accuracy:\t{:.2f} %".format(float(accuracy)*100))
"""

