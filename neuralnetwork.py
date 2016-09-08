#!/bin/python
import os, sys, time
import numpy as np
from six.moves import cPickle
import theano
import theano.tensor as T
from lasagne import layers, objectives, updates, regularization, nonlinearities
from lasagne.regularization import apply_penalty, l1, l2
from scipy import stats
import utils 

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNet:
	"""Class to build a neural network and perform basic functions"""

	def __init__(self, network, input_var, target_var):
		self.network = network
		self.input_var = input_var
		self.target_var = target_var
		self.saliency = T.copy(network)
		self.saliency_fn = []


	def get_model_parameters(self, layer='output'):
		"""return all the parameters of the network"""

		return layers.get_all_param_values(self.network[layer])


	def set_model_parameters(self, all_param_values, layer='output'):
		"""initialize network with all_param_values"""
		layers.set_all_param_values(self.network[layer], all_param_values)


	def save_model_parameters(self, filepath, layer='output'):
		"""save model parameters to a file"""

		print "saving model parameters to: " + filepath
		all_param_values = self.get_model_parameters(layer)
		with open(filepath, 'wb') as f:
			cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
	

	def load_model_parameters(self, filepath, layer='output'):
		"""load model parametes from a file"""

		print "loading model parameters from: " + filepath
		all_param_values = self.get_model_parameters(layer)
		with open(filepath, 'rb') as f:
			all_param_values = cPickle.load(f)
		self.set_model_parameters(all_param_values, layer)


	def inspect_layers(self):
		"""print each layer type and parameters"""

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


	def get_feature_maps(self, layer, X, batch_size=500):
		"""get the feature maps of a given convolutional layer"""

		# setup theano function to get feature map of a given layer
		num_data = len(X)
		feature_maps = theano.function([self.input_var], layers.get_output(self.network[layer], deterministic=True), allow_input_downcast=True)
		map_shape = layers.get_output_shape(self.network[layer])

		# get feature maps in batches for speed (large batches may be too much memory for GPU)
		num_batches = num_data // batch_size
		shape = list(map_shape)
		shape[0] = num_data
		fmaps = np.empty(tuple(shape))
		for i in range(num_batches):
			index = range(i*batch_size, (i+1)*batch_size)    
			fmaps[index] = feature_maps(X[index])

		# get the rest of the feature maps
		index = range(num_batches*batch_size, num_data)    
		fmaps[index] = feature_maps(X[index])
		
		return fmaps


	def get_weights(self, layer, normalize=0):
		"""get the weights of a given layer"""

		W = np.squeeze(self.network[layer].W.get_value())
		if normalize == 1:
			for i in range(len(W)):
				MAX = np.max(W[i])
				W[i] = W[i]/MAX*4
				W[i] = np.exp(W[i])
				norm = np.outer(np.ones(4), np.sum(W[i], axis=0))
				W[i] = W[i]/norm
		return W


	def compile_saliency_reconstruction(self, saliency_layer):
		"""compile a saliency function to perform guided back-propagation through
		a network from the saliency_layer to the inputs"""

		all_param_values = layers.get_all_param_values(self.network['output'])
		layers.set_all_param_values(self.saliency['output'], all_param_values)

		modified_relu = GuidedBackprop(nonlinearities.rectify) 
		relu_layers = [layer for layer in layers.get_all_layers(self.saliency[saliency_layer])
					   if getattr(layer, 'nonlinearity', None) is nonlinearities.rectify]
		for layer in relu_layers:
			layer.nonlinearity = modified_relu

		output = layers.get_output(self.network[saliency_layer], deterministic=True)
		max_output = T.max(output, axis=1)
		saliency = theano.grad(max_output.sum(), wrt=self.input_var)

		self.saliency_fn = theano.function([self.input_var], saliency)


	def get_saliency_reconstruction(self, X, normalize=1, batch_size=500):
		"""get the saliency map to the inputs"""

		if not self.saliency_fn:
			self.compile_saliency_reconstruction()

		if X.shape[0] < batch_size:
			saliency = self.saliency_fn(X)
		else:
			num_data = len(X)
			num_batches = num_data // batch_size
			saliency = []
			for i in range(num_batches):
				index = range(i*batch_size, (i+1)*batch_size)    
				saliency.append(self.saliency_fn(X[index]))

			index = range(num_batches*batch_size, num_data)     
			saliency.append(self.saliency_fn(X[index]))
			saliency = np.vstack(saliency)

		if normalize:
			saliency = np.array(saliency)
			saliency = np.squeeze(saliency)
			for i in range(len(saliency)):
				saliency[i] = utils.normalize_pwm(saliency[i], method=2)

		return saliency


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
		batches = utils.batch_generator(train[0], train[1], batch_size, shuffle=True)
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
		"""metric to monitor performance during training"""

		if self.objective == 'categorical':
			return np.mean(np.argmax(prediction, axis=1) == y)
		elif self.objective == 'binary':
			return np.mean(np.round(prediction) == y)
		elif self.objective == 'squared_error':
			return np.corrcoef(prediction[:,0],y[:,0])[0][1]


	def test_step(self, test, batch_size, verbose=1):
		"""perform a complete forward pass with a test function"""

		performance = MonitorPerformance('test',self.objective, verbose)
		num_batches = test[1].shape[0] // batch_size
		batches = utils.batch_generator(test[0], test[1], batch_size, shuffle=False)
		label = []
		prediction = []
		for batch in range(num_batches):
			X, y = next(batches)
			loss, prediction_minibatch = self.test_fun(X, y)
			performance.add_loss(loss)
			prediction.append(prediction_minibatch)
			label.append(y)
		prediction = np.vstack(prediction)
		label = np.vstack(label)

		return performance.get_mean_loss(), prediction, label


	def test_model(self, test, batch_size, name):
		"""perform a complete forward pass, store and print results"""

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
		"""add loss score to monitor class"""

		if name == "train":
			self.train_monitor.add_loss(loss)
		elif name == "valid":
			self.valid_monitor.add_loss(loss)
		elif name == "test":
			self.test_monitor.add_loss(loss)


	def save_model(self):
		"""save model parameters to file, according to filepath"""

		if self.save == 'best':
			min_loss, min_epoch = self.valid_monitor.get_min_loss()
			if self.valid_monitor.loss[-1] <= min_loss:
				filepath = self.filepath + '_best.pickle'
				self.nnmodel.save_model_parameters(filepath, 'output')
		elif self.save == 'all':
			epoch = len(self.valid_monitor.loss)
			filepath = self.filepath + '_' + str(epoch) +'.pickle'
			self.nnmodel.save_model_parameters(filepath)
			if self.valid_monitor.loss[-1] <= min_loss:
				filepath = self.filepath + '_best.pickle'
				self.nnmodel.save_model_parameters(filepath)


	def save_all_metrics(self, filepath):
		"""save all performance metrics"""

		self.train_monitor.save_metrics(filepath)
		self.test_monitor.save_metrics(filepath)
		self.valid_monitor.save_metrics(filepath)


	def early_stopping(self, current_loss, current_epoch, patience):
		"""check if validation loss is not improving and stop after patience
		runs out"""

		min_loss, min_epoch = self.valid_monitor.get_min_loss()
		status = True

		if min_loss < current_loss:
			if patience - (current_epoch - min_epoch) < 0:
				status = False
				print "Patience ran out... Early stopping."
		return status


	def set_best_parameters(self, filepath=[]):
		""" set the best parameters from file"""

		if not filepath:
			filepath = self.filepath + '_best.pickle'

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
		metrics = utils.calculate_metrics(label, prediction, self.objective)
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
				print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
					

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

	elif (optimization["objective"] == 'squared_error'):
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

#---------------------------------------------------------------------------------------------------------
# saliency and reconstruction



class ModifiedBackprop(object):

	def __init__(self, nonlinearity):
		self.nonlinearity = nonlinearity
		self.ops = {}  # memoizes an OpFromGraph instance per tensor type

	def __call__(self, x):
	   
		cuda_var = theano.sandbox.cuda.as_cuda_ndarray_variable
		x = cuda_var(x)
		tensor_type = x.type
		
		if tensor_type not in self.ops:
			input_var = tensor_type()
			output_var = cuda_var(self.nonlinearity(input_var))
			op = theano.OpFromGraph([input_var], [output_var])
			op.grad = self.grad
			self.ops[tensor_type] = op

		return self.ops[tensor_type](x)

class GuidedBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		dtype = inp.dtype
		return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)
