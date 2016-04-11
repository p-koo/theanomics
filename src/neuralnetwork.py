#!/bin/python
import os
import sys
import numpy as np
from six.moves import cPickle
import time
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates, regularization


sys.path.append(os.path.realpath('..'))
from models import load_model
sys.path.append('..')
from utils import calculate_metrics
from utils import batch_generator

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNet:
	"""Class to build and train a feed-forward neural network"""

	def __init__(self, model_name, shape, num_labels):
		self.model_name = model_name
		self.shape = shape
		self.num_labels = num_labels

		network, input_var, target_var, optimization = load_model(model_name, shape, num_labels)
		self.network = network
		self.input_var = input_var
		self.target_var = target_var
		self.optimization = optimization		

		# build model 
		train_fun, test_fun = build_optimizer(network, input_var, target_var, optimization)
		self.train_fun = train_fun
		self.test_fun = test_fun

		self.train_monitor = MonitorPerformance(name="train")
		self.test_monitor = MonitorPerformance(name="test")
		self.valid_monitor = MonitorPerformance(name="cross-validation")


	def reinitialize(self):
		network, input_var, target_var, optimization = load_model(self.model_name, self.shape, self.num_labels)
		self.network = network
		self.input_var = input_var
		self.target_var = target_var
		self.optimization = optimization

		train_fun, test_fun = build_optimizer(self.network, self.input_var, self.target_var, self.optimization)
		self.train_fun = train_fun
		self.test_fun = test_fun


	def get_model_parameters(self):
		return layers.get_all_param_values(self.network['output'])


	def set_model_parameters(self, all_param_values):
		self.network['output'] = layers.set_all_param_values(self.network['output'], all_param_values)


	def save_model_parameters(self, filepath):
		print "saving model parameters to: " + filepath
		all_param_values = layers.get_all_param_values(self.network['output'])
		f = open(filepath, 'wb')
		cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


	def set_best_model_parameters(self, filepath):
		min_cost, min_index = self.valid_monitor.get_min_cost()    
		savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
		f = open(savepath, 'rb')
		best_parameters = cPickle.load(f)
		self.set_model_parameters(best_parameters)
		

	def test_step_batch(self, test):
		test_cost, test_prediction = self.test_fun(test[0].astype(np.float32), test[1].astype(np.int32)) 
		return test_cost, test_prediction


	def test_step_minibatch(self, test, batch_size):

		performance = MonitorPerformance()

		if np.ndim(test[1]) == 2:
			label = np.empty(test[1].shape[1])
		else:
			label = np.empty(1)
		prediction = np.empty((1,max(test[1])+1))

		num_batches = test[1].shape[0] // batch_size
		batches = batch_generator(test[0], test[1], batch_size)
		for epoch in range(num_batches):
			X, y = next(batches)
			cost, prediction_minibatch = self.test_fun(X, y)

			performance.add_cost(cost)
			prediction = np.concatenate((prediction, prediction_minibatch), axis=0)
			label = np.concatenate((label, y), axis=0)

		return performance.get_mean_cost(), prediction[1::], label[1::]
		

	def train_step(self,  train, batch_size, verbose=1):        
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance(verbose)
		performance.set_start_time(start_time = time.time())

		# train on mini-batch with random shuffling
		num_batches = train[0].shape[0] // batch_size
		batches = batch_generator(train[0], train[1], batch_size)
		for epoch in range(num_batches):
			X, y = next(batches)
			cost, prediction = self.train_fun(X, y)
			performance.add_cost(cost)
			performance.progress_bar(epoch+1., num_batches)
		print "" 
		return performance.get_mean_cost()


	def test_model(self, test, batch_size, name):
		test_cost, test_prediction, test_label = self.test_step_minibatch(test, batch_size)
		if name == "train":
			self.train_monitor.update(test_cost, test_prediction, test_label)
			self.train_monitor.print_results(name)
		if name == "valid":
			self.valid_monitor.update(test_cost, test_prediction, test_label)
			self.valid_monitor.print_results(name)
		if name == "test":
			self.test_monitor.update(test_cost, test_prediction, test_label)
			self.test_monitor.print_results(name)


	def save_metrics(self, filepath, name):
		if name == "train":
			self.train_monitor.save_metrics(filepath)
		elif name == "test":
			self.test_monitor.save_metrics(filepath)
		elif name == "valid":
			self.valid_monitor.save_metrics(filepath)


	def save_all_metrics(self, filepath):
		self.save_metrics(filepath, "train")
		self.save_metrics(filepath, "test")
		self.save_metrics(filepath, "valid")


	def get_num_epochs(self):
		return self.train_monitor.get_length()


#----------------------------------------------------------------------------------------------------
# Monitor performance metrics class
#----------------------------------------------------------------------------------------------------

class MonitorPerformance():
	"""helper class to monitor and store performance metrics during 
	   training. This class uses the metrics for early stopping. """

	def __init__(self, name = '', verbose=1):
		self.cost = []
		self.metric = np.zeros(3)
		self.metric_std = np.zeros(3)
		self.verbose = verbose
		self.name = name
		self.roc = []
		self.pr = []


	def set_verbose(self, verbose):
		self.verbose = verbose


	def add_cost(self, cost):
		self.cost = np.append(self.cost, cost)


	def add_metrics(self, mean, std):
		if mean:
			self.metric = np.vstack([self.metric, mean])
		if std:
			self.metric_std = np.vstack([self.metric_std, std])


	def get_length(self):
		return len(self.cost)

	def update(self, cost, prediction, label):
		mean, std, roc, pr = calculate_metrics(label, prediction)
		self.add_cost(cost)
		self.add_metrics(mean, std)
		self.roc = roc
		self.pr = pr

	def get_mean_cost(self):
		return np.mean(self.cost)

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


	def print_results(self, name): 
		if self.verbose == 1:
			print("  " + name + " cost:\t\t{:.4f}".format(self.cost[-1]/1.))
			if self.metric.any():
				accuracy, auc_roc, auc_pr = self.get_mean_values()
				accuracy_std, auc_roc_std, auc_pr_std = self.get_error_values()
				print("  " + name + " accuracy:\t{:.4f}+/-{:.4f}".format(accuracy, accuracy_std))
				print("  " + name + " auc-roc:\t{:.4f}+/-{:.4f}".format(auc_roc, auc_roc_std))
				print("  " + name + " auc-pr:\t\t{:.4f}+/-{:.4f}".format(auc_pr, auc_pr_std))


	def progress_bar(self, epoch, num_batches, bar_length=30):
		if self.verbose == 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-epoch)/epoch
			percent = epoch/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- cost=%.5f     " \
			%(progress+spaces, percent*100, remaining_time, self.get_mean_cost()))
			sys.stdout.flush()


	def save_metrics(self, filepath):
		savepath = filepath + "_" + self.name +"_performance.pickle"
		print "saving metrics to " + savepath

		f = open(savepath, 'wb')
		cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.cost, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.roc, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.pr, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


#------------------------------------------------------------------------------------------
# Neural network model building functions
#------------------------------------------------------------------------------------------

def build_optimizer(network, input_var, target_var, optimization):
	# build cost function
	prediction = layers.get_output(network["output"], deterministic=False)
	cost = build_cost(network, target_var, prediction, optimization)

	# calculate and clip gradients
	params = layers.get_all_params(network["output"], trainable=True)    
	if "weight_norm" in optimization:
		grad = calculate_gradient(cost, params, weight_norm=optimization["weight_norm"])
	else:
		grad = calculate_gradient(cost, params)
	  
	# setup parameter updates
	update_op = build_updates(grad, params, optimization)

	# test/validation set 
	test_prediction = layers.get_output(network["output"], deterministic=True)
	test_cost = build_cost(network, target_var, test_prediction, optimization)

	# create theano function
	train_fun = theano.function([input_var, target_var], [cost, prediction], updates=update_op)
	test_fun = theano.function([input_var, target_var], [test_cost, test_prediction])

	return train_fun, test_fun


def build_cost(network, target_var, prediction, optimization):
	""" setup cost function with weight decay regularization """

	if optimization["objective"] == 'categorical':
		cost = objectives.categorical_crossentropy(prediction, target_var)

	elif optimization["objective"] == 'binary':
		cost = objectives.binary_crossentropy(prediction, target_var)

	elif optimization["objective"] == 'mse':
		cost = objectives.squared_error(prediction, target_var)

	#cost = cost.mean()
	cost = objectives.aggregate(cost, mode='mean')

	# weight-decay regularization
	if "l1" in optimization:
		l1_penalty = regularization.regularize_network_params(network, regularization.l1) * optimization["l1"]
		cost += l1_penalty
	if "l2" in optimization:
		l2_penalty = regularization.regularize_network_params(network, regularization.l2) * optimization["l2"]        
		cost += l2_penalty 

	return cost


def calculate_gradient(cost, params, weight_norm=[]):
	""" calculate gradients with option to clip norm """

	grad = T.grad(cost, params)

	# gradient clipping option
	if weight_norm:
		grad = updates.total_norm_constraint(grad, weight_norm)

	return grad


def build_updates(grad, params, update_params):
	""" setup optimization algorithm """

	if update_params['optimizer'] == 'sgd':
		update_op = updates.sgd(grad, params, learning_rate=update_params['learning_rate']) 
 
	elif update_params['optimizer'] == 'nesterov_momentum':
		update_op = updates.nesterov_momentum(grad, params, 
									learning_rate=update_params['learning_rate'], 
									momentum=update_params['momentum'])
	
	elif update_params['optimizer'] == 'adagrad':
		if "learning_rate" in update_params:
			update_op = updates.adagrad(grad, params, 
							  learning_rate=update_params['learning_rate'], 
							  epsilon=update_params['epsilon'])
		else:
			update_op = updates.adagrad(grad, params)

	elif update_params['optimizer'] == 'rmsprop':
		if "learning_rate" in update_params:
			update_op = updates.rmsprop(grad, params, 
							  learning_rate=update_params['learning_rate'], 
							  rho=update_params['rho'], 
							  epsilon=update_params['epsilon'])
		else:
			update_op = updates.rmsprop(grad, params)
	
	elif update_params['optimizer'] == 'adam':
		if "learning_rate" in update_params:
			update_op = updates.adam(grad, params, 
							learning_rate=update_params['learning_rate'], 
							beta1=update_params['beta1'], 
							beta2=update_params['beta2'], 
							epsilon=update['epsilon'])
		else:
			update_op = updates.adam(grad, params)
  
	return update_op


