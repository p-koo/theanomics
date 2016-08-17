#!/bin/python
import os, sys, time
import numpy as np
from six.moves import cPickle
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates, regularization
from lasagne.regularization import apply_penalty, l1, l2
from scipy import stats
from utils import calculate_metrics, batch_generator

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNet:
	"""Class to build a neural network"""

	def __init__(self, network, input_var, target_var):
		self.network = network
		self.input_var = input_var
		self.target_var = target_var


	def get_model_parameters(self):
		return layers.get_all_param_values(self.network['output'])


	def set_model_parameters(self, all_param_values):
		self.network['output'] = layers.set_all_param_values(self.network['output'], all_param_values)


	def save_model_parameters(self, filepath):
		print "saving model parameters to: " + filepath
		all_param_values = layers.get_all_param_values(self.network['output'])
		with open(filepath, 'wb') as f:
			cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
	

	def load_model_parameters(self, filepath):
		print "loading model parameters from: " + filepath
		all_param_values = layers.get_all_param_values(self.network['output'])
		with open(filepath, 'rb') as f:
			all_param_values = cPickle.load(f)
		self.set_model_parameters(all_param_values)


	def inspect_layers(self):
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
# Train neural networks class
#----------------------------------------------------------------------------------------------------

class NeuralTrainer:
	"""Class to train a feed-forward neural network"""

	def __init__(self, nnmodel, optimization, save='best', filepath='.'):
		self.nnmodel = nnmodel
		self.optimization = optimization	
		self.save = save
		self.filepath = filepath
		self.objective = optimization["objective"]	
		self.learning_rate = theano.shared(np.array(optimization['learning_rate'], dtype=theano.config.floatX))

		# build model 
		print "compiling model"
		train_fun, test_fun = build_optimizer(nnmodel.network, nnmodel.input_var, nnmodel.target_var, 
		                                      optimization, self.learning_rate)
		self.train_fun = train_fun
		self.test_fun = test_fun

		self.train_monitor = MonitorPerformance(name="train", objective=self.objective, verbose=1)
		self.test_monitor = MonitorPerformance(name="test", objective=self.objective, verbose=1)
		self.valid_monitor = MonitorPerformance(name="cross-validation", objective=self.objective, verbose=1)

	def set_learning_rate(self, new_learning_rate):
		self.learning_rate.set_value(new_learning_rate) 
		

	def train_step(self,  train, batch_size, verbose=1):        
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance('train', self.objective, verbose)
		performance.set_start_time(start_time = time.time())

		# train on mini-batch with random shuffling
		num_batches = train[0].shape[0] // batch_size
		batches = batch_generator(train[0], train[1], batch_size, shuffle=True)
		value = 0
		for i in range(num_batches):
			X, y = next(batches)
			loss, prediction = self.train_fun(X, y)
			value += self.train_metric(prediction, y)
			performance.add_loss(loss)
			performance.progress_bar(i+1., num_batches, value/(i+1))
		print "" 
		return performance.get_mean_loss()


	def train_metric(self, prediction, y):
		if self.objective == 'categorical':
			return np.mean(np.argmax(prediction, axis=1) == y)
		elif self.objective == 'binary':
			return np.mean(np.round(prediction) == y)


	def test_step(self, test, batch_size, verbose=1):
		performance = MonitorPerformance('test',self.objective, verbose)
		num_batches = test[1].shape[0] // batch_size
		batches = batch_generator(test[0], test[1], batch_size, shuffle=False)
		label = []
		prediction = []
		for epoch in range(num_batches):
			X, y = next(batches)
			loss, prediction_minibatch = self.test_fun(X, y)
			performance.add_loss(loss)
			prediction.append(prediction_minibatch)
			label.append(y)
		prediction = np.vstack(prediction)
		label = np.vstack(label)

		return performance.get_mean_loss(), prediction, label


	def test_model(self, test, batch_size, name):
		test_loss, test_prediction, test_label = self.test_step(test, batch_size)
		if name == "train":
			self.train_monitor.update(test_loss, test_prediction, test_label)
			self.train_monitor.print_results(name)
		elif name == "valid":
			self.valid_monitor.update(test_loss, test_prediction, test_label)
			self.valid_monitor.print_results(name)
		elif name == "test":
			self.test_monitor.update(test_loss, test_prediction, test_label)
			self.test_monitor.print_results(name)
		return test_loss
	

	def add_loss(self, loss, name):
		if name == "train":
			self.train_monitor.add_loss(loss)
		elif name == "valid":
			self.valid_monitor.add_loss(loss)
		elif name == "test":
			self.test_monitor.add_loss(loss)


	def save_model(self):
		if self.save == 'best':
			min_loss, min_epoch = self.valid_monitor.get_min_loss()
			if self.valid_monitor.loss[-1] <= min_loss:
				filepath = self.filepath + '_best.pickle'
				self.nnmodel.save_model_parameters(filepath)
		elif self.save == 'all':
			epoch = len(self.valid_monitor.loss)
			filepath = self.filepath + '_' + str(epoch) +'.pickle'
			self.nnmodel.save_model_parameters(filepath)


	def save_all_metrics(self, filepath):
		self.train_monitor.save_metrics(filepath)
		self.test_monitor.save_metrics(filepath)
		self.valid_monitor.save_metrics(filepath)


	def early_stopping(self, current_loss, current_epoch, patience):
		min_loss, min_epoch = self.valid_monitor.get_min_loss()
		status = True

		if min_loss < current_loss:
			if patience - (current_epoch - min_epoch) < 0:
				status = False
				print "Patience ran out... Early stopping."
		return status


	def set_best_parameters(self):
		if self.save == 'best':
			filepath = self.filepath + '_best.pickle'
		elif self.save == 'all':
			min_loss, min_epoch = self.valid_monitor.get_min_loss()
			filepath = self.filepath + '_' + str(epoch) +'.pickle'

		f = open(filepath, 'rb')
		all_param_values = cPickle.load(f)
		f.close()
		self.nnmodel.set_model_parameters(all_param_values)


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
		self.loss = []
		self.metric = []
		self.metric_std = []


	def set_verbose(self, verbose):
		self.verbose = verbose


	def add_loss(self, loss):
		self.loss = np.append(self.loss, loss)


	def add_metrics(self, metrics):
		self.metric.append(metrics[0])
		self.metric_std.append(metrics[1])


	def get_length(self):
		return len(self.loss)


	def update(self, loss, prediction, label):
		metrics = calculate_metrics(label, prediction, self.objective)
		self.add_loss(loss)
		self.add_metrics(metrics)


	def get_mean_loss(self):
		return np.mean(self.loss)


	def get_metric_values(self):
		return self.metric[-1], self.metric_std[-1]


	def get_min_loss(self):
 		min_loss = min(self.loss)
		min_index = np.argmin(self.loss)
		return min_loss, min_index


	def set_start_time(self, start_time):
		self.start_time = start_time


	def print_results(self, name): 
		if self.verbose == 1:
			print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]/1.))
			mean_vals, error_vals = self.get_metric_values()
			
			if (self.objective == "binary") | (self.objective == "categorical"):
				print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
			elif (self.objective == 'squared_error'):
				print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
					

	def progress_bar(self, epoch, num_batches, value, bar_length=30):
		if self.verbose == 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-epoch)/epoch
			percent = epoch/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			if (self.objective == "binary") | (self.objective == "categorical"):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f -- accuracy=%.2f%%  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value*100))
			elif (self.objective == 'squared_error'):
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
		prediction = T.clip(prediction, 1e-7, 1-1e-7)
		loss = -(target_var*T.log(prediction) + (1.0-target_var)*T.log(1.0-prediction))
		# loss = objectives.binary_crossentropy(prediction[:,loss_index], target_var[:,loss_index])

	elif optimization["objective"] == 'squared_error':
		loss = objectives.squared_error(prediction, target_var)

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

	# gradient norm option
	if weight_norm:
		grad = updates.total_norm_constraint(grad, weight_norm)

	return grad


def build_updates(grad, params, update_params, learning_rate):
	""" setup optimization algorithm """

	if update_params['optimizer'] == 'sgd':
		update_op = updates.sgd(grad, params, learning_rate=learning_rate) 
 
	elif update_params['optimizer'] == 'nesterov_momentum':
		update_op = updates.nesterov_momentum(grad, params, learning_rate=learning_rate, momentum=update_params['momentum'])
	
	elif update_params['optimizer'] == 'adagrad':
		update_op = updates.adagrad(grad, params, learning_rate=learning_rate)
	
	elif update_params['optimizer'] == 'rmsprop':
		update_op = updates.rmsprop(grad, params, learning_rate=learning_rate)
	
	elif update_params['optimizer'] == 'adam':
		update_op = updates.adam(grad, params, learning_rate=learning_rate)
  
	return update_op


