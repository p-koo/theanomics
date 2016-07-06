#!/bin/python
import os
import sys
import numpy as np
from six.moves import cPickle
import time
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates, regularization
from lasagne.regularization import apply_penalty, l1, l2
from scipy import stats

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
		self.objective = optimization["objective"]	
		if 'learning_rate' in optimization:
			self.learning_rate = theano.shared(np.array(optimization['learning_rate'], dtype=theano.config.floatX))
			self.learning_rate = optimization['learning_rate']
		else:
			self.learning_rate = []

		# build model 
		train_fun, test_fun = build_optimizer(network, input_var, target_var, optimization, self.learning_rate)
		self.train_fun = train_fun
		self.test_fun = test_fun

		self.train_monitor = MonitorPerformance(name="train", objective=self.objective, verbose=1)
		self.test_monitor = MonitorPerformance(name="test", objective=self.objective, verbose=1)
		self.valid_monitor = MonitorPerformance(name="cross-validation", objective=self.objective, verbose=1)


	def reinitialize(self):
		network, input_var, target_var, optimization = load_model(self.model_name, self.shape, self.num_labels)
		self.network = network
		self.input_var = input_var
		self.target_var = target_var
		self.optimization = optimization

		train_fun, test_fun = build_optimizer(self.network, self.input_var, self.target_var, self.optimization, self.learning_rate)
		self.train_fun = train_fun
		self.test_fun = test_fun


	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate
		train_fun, test_fun = build_optimizer(self.network, self.input_var, self.target_var, self.optimization, self.learning_rate)
		self.train_fun = train_fun
		self.test_fun = test_fun


	def get_model_parameters(self):
		return layers.get_all_param_values(self.network['output'])


	def set_model_parameters(self, all_param_values):
		self.network['output'] = layers.set_all_param_values(self.network['output'], all_param_values)


	def save_model_parameters(self, filepath, best=False):
		print "saving model parameters to: " + filepath
		all_param_values = layers.get_all_param_values(self.network['output'])
		f = open(filepath, 'wb')
		cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


	def set_parameters_from_file(self, savepath):
		self.reinitialize()

		# load model parameters for a given training epoch
		f = open(savepath, 'rb')
		best_parameters = cPickle.load(f)
		f.close()

		# get test metrics 
		self.set_model_parameters(best_parameters)


	def get_min_loss(self):

		min_loss, min_index = self.valid_monitor.get_min_loss()
		return min_loss, min_index    


	def test_step(self, test, batch_size):

		performance = MonitorPerformance('test',self.objective, verbose=1)

		if self.objective == 'categorical':
			label = np.empty(1)
			prediction = np.empty((1,max(test[1])+1))
		else:
			label = np.empty((1,test[1].shape[1]))
			prediction = np.empty((1,test[1].shape[1]))
		
		num_batches = test[1].shape[0] // batch_size
		batches = batch_generator(test[0], test[1], batch_size, shuffle=False)
		for epoch in range(num_batches):
			X, y = next(batches)
			loss, prediction_minibatch = self.test_fun(X, y)

			performance.add_loss(loss)
			prediction = np.concatenate((prediction, prediction_minibatch), axis=0)
			label = np.concatenate((label, y), axis=0)

		return performance.get_mean_loss(), prediction[1::], label[1::]

	def train_step(self,  train, batch_size, verbose=1):        
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance('train',self.objective, verbose)
		performance.set_start_time(start_time = time.time())

		# train on mini-batch with random shuffling
		num_batches = train[0].shape[0] // batch_size
		batches = batch_generator(train[0], train[1], batch_size)
		value = 0
		for epoch in range(num_batches):
			X, y = next(batches)
			loss, prediction = self.train_fun(X, y)
			value += self.train_metric(prediction, y)
			performance.add_loss(loss)
			performance.progress_bar(epoch+1., num_batches, value/(epoch+1))
		print "" 
		return performance.get_mean_loss()

	def train_metric(self, prediction, y):
		if self.objective == 'categorical':
			return np.mean(np.argmax(prediction, axis=1) == y)
		elif (self.objective == 'binary') | (self.objective == 'multi-binary') | (self.objective == 'hinge'):
			return np.mean(np.round(prediction) == y)
		else:
			R = []
			for i in range(prediction.shape[1]):
				R.append(stats.spearmanr(prediction[:,i], y[:,i])[0])
			return np.mean(R)

	def test_model(self, test, batch_size, name):
		test_loss, test_prediction, test_label = self.test_step(test, batch_size)
		if name == "train":
			self.train_monitor.update(test_loss, test_prediction, test_label)
			self.train_monitor.print_results(name)
		if name == "valid":
			self.valid_monitor.update(test_loss, test_prediction, test_label)
			self.valid_monitor.print_results(name)
		if name == "test":
			self.test_monitor.update(test_loss, test_prediction, test_label)
			self.test_monitor.print_results(name)
		return test_loss
	
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
		

	def print_layers(self):
		all_layers = layers.get_all_layers(self.network['output'])
		print '----------------------------------------------------------------------------'
		print 'Network architecture:'
		print '----------------------------------------------------------------------------'
		counter = 1
		for layer in all_layers:
			output_shape = layer.output_shape
			params = layer.get_params()

			print 'layer'+str(counter) + ': '
			print str(layer)
			print 'shape:' +  str(output_shape)
			if params:
				all_params = ''
				for param in params:
					all_params += str(param) + ', '
				print 'parameters: ' + str(all_params[0:-2])
			counter += 1
		print '----------------------------------------------------------------------------'

#----------------------------------------------------------------------------------------------------
# Monitor performance metrics class
#----------------------------------------------------------------------------------------------------

class MonitorPerformance():
	"""helper class to monitor and store performance metrics during 
	   training. This class uses the metrics for early stopping. """

	def __init__(self, name='', objective='binary', verbose=1):
		self.name = name
		self.objective = objective
		self.verbose = verbose
		if (objective == 'binary') | (objective == 'categorical'):
			self.num_metrics = 3
		else:
			self.num_metrics = 3
		self.loss = []
		self.metric = np.zeros(self.num_metrics)
		self.metric_std = np.zeros(self.num_metrics)	


	def set_verbose(self, verbose):
		self.verbose = verbose


	def add_loss(self, loss):
		self.loss = np.append(self.loss, loss)

	def add_metrics(self, metrics):
		self.metric = np.vstack([self.metric, metrics[0]])
		self.metric_std = np.vstack([self.metric_std, metrics[1]])

	def get_length(self):
		return len(self.loss)

	def update(self, loss, prediction, label):
		metrics = calculate_metrics(label, prediction, self.objective)
		self.add_loss(loss)
		self.add_metrics(metrics)

	def get_mean_loss(self):
		return np.mean(self.loss)

	def get_mean_values(self):
		results = self.metric[-1,:]
		return results

	def get_error_values(self):
		results = self.metric_std[-1,:]
		return results


	def get_min_loss(self):
 		min_loss = min(self.loss)
		min_index = np.argmin(self.loss)
		return min_loss, min_index


	def early_stopping(self, current_loss, current_epoch, patience):
		min_loss, min_epoch = self.get_min_loss()
		status = True

		if min_loss < current_loss:
			if patience - (current_epoch - min_epoch) < 0:
				status = False
				print "Patience ran out... Early stopping."
		return status


	def set_start_time(self, start_time):
		if self.verbose == 1:
			self.start_time = start_time


	def print_results(self, name): 
		if self.verbose == 1:
			print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]/1.))
			if self.metric.any():
				mean_vals = self.get_mean_values()
				error_vals = self.get_error_values()
				
				if (self.objective == "binary") | (self.objective == "categorical") | (self.objective == "multi-binary") | (self.objective == 'hinge'):
					print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
					print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
					print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
				elif (self.objective == 'ols') | (self.objective == 'gls') | (self.objective == 'autoencoder'):
					print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
					print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
					print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
					

	def progress_bar(self, epoch, num_batches, value, bar_length=30):
		if self.verbose == 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-epoch)/epoch
			percent = epoch/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			if (self.objective == "binary") | (self.objective == "categorical") | (self.objective == "multi-binary") | (self.objective == 'hinge'):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f -- accuracy=%.2f%%  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value*100))
			elif (self.objective == 'ols') | (self.objective == 'gls') | (self.objective == 'autoencoder'):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f -- correlation=%.5f  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value))
			sys.stdout.flush()


	def save_metrics(self, filepath):
		savepath = filepath + "_" + self.name +"_performance.pickle"
		print "saving metrics to " + savepath

		f = open(savepath, 'wb')
		cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()


#------------------------------------------------------------------------------------------
# Neural network model building functions
#------------------------------------------------------------------------------------------

def build_optimizer(network, input_var, target_var, optimization, learning_rate):

	if optimization["objective"] == 'autoencoder':
		prediction = layers.get_output(network['output'], deterministic=False)
		loss1 = build_loss(input_var, prediction, optimization)

		prediction2 = layers.get_output(network['encode'], deterministic=False)
		loss2 = build_loss(target_var, prediction2, optimization)

		loss = loss1 + loss2

		# regularize parameters
		loss += regularization(network['output'], optimization)

		# calculate and clip gradients
		params = layers.get_all_params(network['output'], trainable=True)    
		if "weight_norm" in optimization:
			grad = calculate_gradient(loss, params, weight_norm=optimization["weight_norm"])
		else:
			grad = calculate_gradient(loss, params)

		# setup parameter updates
		update_op = build_updates(grad, params, optimization, learning_rate)

		# test/validation set 
		test_prediction = layers.get_output(network['output'], deterministic=True)
		test_loss1 = build_loss(input_var, test_prediction, optimization)
		test_prediction2 = layers.get_output(network['encode'], deterministic=True)
		test_loss2 = build_loss(target_var, test_prediction2, optimization)
		test_loss = test_loss1 + test_loss2


		# create theano function
		train_fun = theano.function([input_var, target_var], [loss, prediction2], updates=update_op)
		test_fun = theano.function([input_var, target_var], [test_loss, test_prediction2])

	else:
		# build loss function
		prediction = layers.get_output(network['output'], deterministic=False)
		loss = build_loss(target_var, prediction, optimization)

		# regularize parameters
		loss += regularization(network['output'], optimization)

		# calculate and clip gradients
		params = layers.get_all_params(network['output'], trainable=True)    
		if "weight_norm" in optimization:
			grad = calculate_gradient(loss, params, weight_norm=optimization["weight_norm"])
		else:
			grad = calculate_gradient(loss, params)
		  
		# setup parameter updates
		update_op = build_updates(grad, params, optimization, learning_rate)

		# test/validation set 
		test_prediction = layers.get_output(network['output'], deterministic=True)
		test_loss = build_loss(target_var, test_prediction, optimization)

		# create theano function
		train_fun = theano.function([input_var, target_var], [loss, prediction], updates=update_op)
		test_fun = theano.function([input_var, target_var], [test_loss, test_prediction])

	return train_fun, test_fun


def build_loss(target_var, prediction, optimization):
	""" setup loss function with weight decay regularization """

	if optimization["objective"] == 'categorical':
		loss = objectives.categorical_crossentropy(prediction, target_var)

	elif optimization["objective"] == 'binary':
		#loss = -(target_var*T.log(prediction) + (1.0-target_var)*T.log(1.0-prediction))
		loss = objectives.binary_crossentropy(prediction, target_var)

	elif optimization["objective"] == 'multi-binary':		
		def interaction(rho, u):
			diag = T.diag(T.dot(u,T.dot(rho, u.T)))
			return diag		
		prediction = T.clip(prediction, 1e-7, 1-1e-7)
		binary_loss = (target_var*T.log(prediction) + (1.0-target_var)*T.log(1.0-prediction)).sum(axis=1)
		u = T.abs_(target_var - prediction)/T.sqrt(prediction*(1-prediction))	
		second_order = interaction(optimization["rho_ij"], u)
		correction = T.log(T.maximum(1 + second_order,1e-7))
		loss = -(binary_loss - correction)/prediction.shape[1] 

	elif optimization["objective"] == 'autoencoder':
		loss = objectives.squared_error(prediction, target_var)

	elif optimization["objective"] == 'ols':
		loss = objectives.squared_error(prediction, target_var)

	elif optimization["objective"] == 'gls':
		error = (target_var - prediction)
		decor_error = T.dot(optimization["Linv"], error.T).T
		loss = decor_error ** 2

	elif optimization["objective"] == 'hinge':
		error = (target_var - prediction)
		loss = T.diag(T.dot(error,T.dot(optimization["rho_ij"], error.T)))
		#prediction = T.dot(optimization['rho_ij'],prediction.T).T
		#loss = T.nnet.relu(1 - prediction*target_var - (1-prediction)*(1-target_var))

	#loss = loss.mean()
	loss = objectives.aggregate(loss, mode='mean')

	return loss


def regularization(network, optimization):
	all_params = layers.get_all_params(network, regularizable=True)    

	# weight-decay regularization
	loss = 0
	if "l1" in optimization:
		l1_penalty = apply_penalty(all_params, l1) * optimization["l1"]
		loss += l1_penalty
	if "l2" in optimization:
		l2_penalty = apply_penalty(all_params, l2)* optimization["l2"]        
		loss += l2_penalty 
	return loss


def calculate_gradient(loss, params, weight_norm=[]):
	""" calculate gradients with option to clip norm """

	grad = T.grad(loss, params)

	# gradient clipping option
	if weight_norm:
		grad = updates.total_norm_constraint(grad, weight_norm)

	return grad


def build_updates(grad, params, update_params, learning_rate):
	""" setup optimization algorithm """

	if update_params['optimizer'] == 'sgd':
		update_op = updates.sgd(grad, params, learning_rate=learning_rate) 
 
	elif update_params['optimizer'] == 'nesterov_momentum':
		update_op = updates.nesterov_momentum(grad, params, 
									learning_rate=learning_rate, 
									momentum=update_params['momentum'])
	
	elif update_params['optimizer'] == 'adagrad':
		if learning_rate:
			update_op = updates.adagrad(grad, params, 
							  learning_rate=learning_rate)
		else:
			update_op = updates.adagrad(grad, params)

	elif update_params['optimizer'] == 'rmsprop':
		if learning_rate:
			update_op = updates.rmsprop(grad, params, 
							  learning_rate=learning_rate, 
							  rho=update_params['rho'])
		else:
			update_op = updates.rmsprop(grad, params)
	
	elif update_params['optimizer'] == 'adam':
		if 'beta1' in update_params:
			update_op = updates.adam(grad, params, 
							learning_rate=learning_rate, 
							beta1=update_params['beta1'], 
							beta2=update_params['beta2'])
		else:
			update_op = updates.adam(grad, params)
  
	return update_op


