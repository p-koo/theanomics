#!/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, time
import numpy as np
from six.moves import cPickle
from lasagne import layers, objectives, updates, regularization, nonlinearities
from lasagne.regularization import apply_penalty, l1, l2
from scipy import stats
from .utils import normalize_pwm, batch_generator
from .metrics import calculate_metrics

import theano
import theano.tensor as T



__all__ = [
	"NeuralNet",
	"NeuralTrainer",
	"MonitorPerformance",
	"NeuralGenerator"
]

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNet:
	"""Class to build a neural network and perform basic functions"""

	def __init__(self, network, placeholders):
		self.network = network
		self.placeholders = placeholders
		self.saliency = []
		self.saliency_fn = []

	def get_model_parameters(self, layer='output'):
		"""return all the parameters of the network"""

		return layers.get_all_param_values(self.network[layer])


	def set_model_parameters(self, all_param_values, layer='output'):
		"""initialize network with all_param_values"""
		if layer not in self.network.keys():
			print(layer+ ': not found in network')
		else:
			layers.set_all_param_values(self.network[layer], all_param_values)


	def save_model_parameters(self, file_path, layer='output'):
		"""save model parameters to a file"""
		if layer not in self.network.keys():
			print(layer+ ': not found in network')
		else:
			print("saving model parameters to: " + file_path)
			all_param_values = self.get_model_parameters(layer)
			with open(file_path, 'wb') as f:
				cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
	

	def load_model_parameters(self, file_path, layer='output'):
		"""load model parametes from a file"""
		if layer not in self.network.keys():
			print(layer+ ': not found in network')
		else:
			print("loading model parameters from: " + file_path)
			all_param_values = self.get_model_parameters(layer)
			with open(file_path, 'rb') as f:
				all_param_values = cPickle.load(f)
			self.set_model_parameters(all_param_values, layer)


	def inspect_layers(self):
		"""print each layer type and parameters"""

		all_layers = layers.get_all_layers(self.network[list(self.network)[-1]])
		print('----------------------------------------------------------------------------')
		print('Network architecture:')
		print('----------------------------------------------------------------------------')
		counter = 1
		for layer in all_layers:
			output_shape = layer.output_shape
			params = layer.get_params()

			print('layer'+str(counter) + ': ')
			print(str(layer))
			print('shape:' +  str(output_shape))
			if params:
				all_params = ''
				for param in params:
					all_params += str(param) + ', '
				print('parameters: ' + str(all_params[0:-2]))
			counter += 1
		print('----------------------------------------------------------------------------')


	def get_activations(self, layer, X, batch_size=500):
		"""get the feature maps of a given convolutional layer"""

		# setup theano function to get feature map of a given layer
		num_data = len(X)
		feature_maps = theano.function([self.placeholders['inputs']], layers.get_output(self.network[layer], deterministic=True), allow_input_downcast=True)
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
		if index:
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


	def compile_saliency_reconstruction(self, saliency_layer=None, deterministic=True):
		"""compile a saliency function to perform guided back-propagation through
		a network from the saliency_layer to the inputs"""


		if not saliency_layer:
			saliency_layer = list(self.network.keys())[-2]

		self.saliency = self.network.copy()
		modified_relu = GuidedBackprop(nonlinearities.rectify) 
		relu_layers = [layer for layer in layers.get_all_layers(self.saliency[saliency_layer])
					   if getattr(layer, 'nonlinearity', None) is nonlinearities.rectify]
		for layer in relu_layers:
			layer.nonlinearity = modified_relu

		output = layers.get_output(self.saliency[saliency_layer], deterministic=deterministic)
		max_output = T.max(output, axis=1)
		saliency = theano.grad(max_output.sum(), wrt=self.placeholders['inputs'])
		self.saliency_fn = theano.function([self.placeholders['inputs']], saliency)


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
				saliency[i] = normalize_pwm(saliency[i], method=2)

		return saliency


#----------------------------------------------------------------------------------------------------
# Train neural networks class
#----------------------------------------------------------------------------------------------------

class NeuralTrainer:
	"""Class to train a feed-forward neural network"""

	def __init__(self, nnmodel, optimization, save='best', file_path='.', verbose=1):
		self.nnmodel = nnmodel
		self.optimization = optimization    
		self.save = save
		self.file_path = file_path
		self.objective = optimization["objective"]  
		self.learning_rate = theano.shared(np.array(optimization['learning_rate'], dtype=theano.config.floatX))
		if 'momentum' in optimization:
			self.momentum = theano.shared(np.array(optimization['momentum'], dtype=theano.config.floatX))
		else:
			self.momentum = []

		# build model 
		#print("compiling model")
		train_fun, test_fun = build_optimizer(nnmodel.network, nnmodel.placeholders, 
											  optimization, self.learning_rate)
		self.train_fun = train_fun
		self.test_fun = test_fun

		self.train_monitor = MonitorPerformance(name="train", objective=self.objective, verbose=verbose)
		self.test_monitor = MonitorPerformance(name="test", objective=self.objective, verbose=verbose)
		self.valid_monitor = MonitorPerformance(name="cross-validation", objective=self.objective, verbose=verbose)


	def set_learning_rate(self, new_learning_rate):
		self.learning_rate.set_value(new_learning_rate) 

		
	def set_momentum(self, new_momentum):
		if self.momentum:
			self.momenum.set_value(momenum) 
		

	def train_step(self, train, batch_size, verbose=1, shuffle=True):        
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance('train', self.objective, verbose)
		performance.set_start_time(start_time = time.time())

		# train on mini-batch with random shuffling
		if isinstance(train, (list, tuple)):
			num_data = train[0].shape[0]
		else:
			num_data = train.shape[0]
		num_batches = np.floor(train[0].shape[0] / batch_size).astype(int)
		batches = batch_generator(train, batch_size, shuffle=shuffle)
		value = 0
		for i in range(num_batches):
			X = next(batches)
			loss, prediction = self.train_fun(*X)
			value += self.train_metric(prediction, X[-1])
			performance.add_loss(loss)
			performance.progress_bar(i+1., num_batches, value/(i+1))
		self.add_loss(loss, 'train')
		if verbose:
			print("")
		return performance.get_mean_loss()


	def train_metric(self, prediction, y):
		"""metric to monitor performance during training"""

		if self.objective == 'categorical':
			return np.mean(np.argmax(prediction, axis=1) == np.argmax(y, axis=1))
		elif self.objective == 'binary':
			return np.mean(np.round(prediction) == y)
		elif self.objective == 'squared_error':
			return np.corrcoef(prediction[:,0],y[:,0])[0][1]
		elif self.objective == 'lower_bound':
			return np.mean((prediction - y)**2)


	def test_step(self, test, batch_size):
		"""perform a complete forward pass with a test function"""

		performance = MonitorPerformance('test', self.objective, verbose=0)

		if isinstance(test, (list, tuple)):
			num_data = test[0].shape[0]
		else:
			num_data = test.shape[0]
		num_batches = np.floor(num_data / batch_size).astype(int)
		batches = batch_generator(test, batch_size, shuffle=False)
		label = []
		prediction = []
		for i in range(num_batches):
			X = next(batches)
			loss, prediction_minibatch = self.test_fun(*X)
			performance.add_loss(loss)
			prediction.append(prediction_minibatch)
			label.append(X[-1])
		prediction = np.vstack(prediction)
		label = np.vstack(label)

		return performance.get_mean_loss(), prediction, label


	def test_model(self, test, name, batch_size=100, verbose=1):
		"""perform a complete forward pass, store and print results"""

		test_loss, test_prediction, test_label = self.test_step(test, batch_size)
		if name == "train":
			self.train_monitor.update(test_loss, test_prediction, test_label)
			if verbose >= 1:
				self.train_monitor.print_results(name)
		elif name == "valid":
			self.valid_monitor.update(test_loss, test_prediction, test_label)
			if verbose >= 1:
				self.valid_monitor.print_results(name)
		elif name == "test":
			self.test_monitor.update(test_loss, test_prediction, test_label)
			if verbose >= 1:
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
		"""save model parameters to file, according to file_path"""

		if self.save == 'best':
			min_loss, min_epoch, num_epochs = self.valid_monitor.get_min_loss()
			if self.valid_monitor.loss[-1] <= min_loss:
				file_path = self.file_path + '_best.pickle'
				self.nnmodel.save_model_parameters(file_path, 'output')
		elif self.save == 'all':
			epoch = len(self.valid_monitor.loss)
			file_path = self.file_path + '_' + str(epoch) +'.pickle'
			self.nnmodel.save_model_parameters(file_path)
			if self.valid_monitor.loss[-1] <= min_loss:
				file_path = self.file_path + '_best.pickle'
				self.nnmodel.save_model_parameters(file_path)


	def save_all_metrics(self, file_path):
		"""save all performance metrics"""

		self.train_monitor.save_metrics(file_path)
		self.test_monitor.save_metrics(file_path)
		self.valid_monitor.save_metrics(file_path)


	def early_stopping(self, current_loss, patience):
		"""check if validation loss is not improving and stop after patience
		runs out"""

		min_loss, min_epoch, num_epochs = self.valid_monitor.get_min_loss()
		status = True

		if min_loss < current_loss:
			if patience - (num_epochs - min_epoch) < 0:
				status = False
				print("Patience ran out... Early stopping.")
		return status


	def set_best_parameters(self, file_path=[]):
		""" set the best parameters from file"""

		if not file_path:
			file_path = self.file_path + '_best.pickle'

		f = open(file_path, 'rb')
		all_param_values = cPickle.load(f)
		f.close()
		self.nnmodel.set_model_parameters(all_param_values)


#----------------------------------------------------------------------------------------------------
# Neural generator class
#----------------------------------------------------------------------------------------------------

class NeuralGenerator():
	def __init__(self, network, binary=True):
		self.network = network
		self.binary = binary
		self.z_var = T.dmatrix()
		

		layer_name = list(network.keys())[-2]
		generated_x = layers.get_output(self.network[layer_name], {network['Z']: self.z_var}, deterministic=True)
		self.gen_fn = theano.function([z_var], generated_x)


	def generate_samples(self, z, batch_size=128):
		def sigmoid(x):
		    return 1 / (1 + np.exp(-x))

		num_data = len(z)
		if num_data > batch_size:
			num_batches = num_data // batch_size
			generated_samples = [np.empty(tuple(num_data, z.shape[1]))]
			for i in range(num_batches):
				index = range(i*batch_size, (i+1)*batch_size)    
				generated_samples[index] = self.gen_fn(z[index])

			# get the rest of the feature maps
			index = range(num_batches*batch_size, num_data)    
			if index:
				generated_samples[index] = self.gen_fn(z[index])
		else:
			generated_samples = self.gen_fn(z)

		if self.binary:
			generated_samples = 1 / (1 + np.exp(-generated_samples))

		return generated_samples


	def generate_manifold(self, shape, num_grid, limits=[-2, 2]):

		pos = np.linspace(-limits[0], limits[1], num_grid)

		z = []
		for i in range(num_grid):
		    for j in range(num_grid):
		        z.append(np.asarray([pos[i], pos[j]], dtype=theano.config.floatX))
		z = np.vstack(z)
		generated_samples = self.generate_samples(z)
		
		return generated_samples


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


	def add_metrics(self, scores):
		self.metric.append(scores[0])
		self.metric_std.append(scores[1])


	def get_length(self):
		return len(self.loss)


	def update(self, loss, prediction, label):
		scores = calculate_metrics(label, prediction, self.objective)
		self.add_loss(loss)
		self.add_metrics(scores)


	def get_mean_loss(self):
		return np.mean(self.loss)


	def get_metric_values(self):
		return self.metric[-1], self.metric_std[-1]


	def get_min_loss(self):
		min_loss = min(self.loss)
		min_index = np.argmin(self.loss)
		return min_loss, min_index, len(self.loss)


	def set_start_time(self, start_time):
		self.start_time = start_time


	def print_results(self, name):
		if self.verbose >= 1:
			if name == 'test':
				name += ' '

			print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]))
			mean_vals, error_vals = self.get_metric_values()

			if (self.objective == "binary") | (self.objective == "categorical"):
				print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
			elif (self.objective == 'squared_error'):
				print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
			elif (self.objective == 'lower_bound'):
				print("  " + name + " squared loss:\t{:.5f}".format(mean_vals))
				

	def progress_bar(self, epoch, num_batches, value, bar_length=30):
		if self.verbose >= 2:
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
			elif (self.objective == 'lower_bound'):
				sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss()))

			sys.stdout.flush()


	def save_metrics(self, file_path):
		savepath = file_path + "_" + self.name +"_performance.pickle"
		print("saving metrics to " + savepath)

		f = open(savepath, 'wb')
		cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
		cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()





#------------------------------------------------------------------------------------------
# Neural network model building functions
#------------------------------------------------------------------------------------------

def build_optimizer(network, placeholders, optimization, learning_rate):

	# build loss function 
	
	if optimization['objective'] == 'lower_bound':
		if 'binary' in optimization:
			binary = optimization['binary']
		else:
			binary = False

		loss, prediction = variational_lower_bound(network, placeholders['inputs'], 
													deterministic=False, binary=binary)

		# regularize parameters
		loss += regularization(network['X'], optimization)	

		params = layers.get_all_params(network['X'], trainable=True)

	else:
		prediction = layers.get_output(network['output'], deterministic=False)
		loss = build_loss(placeholders['targets'], prediction, optimization)

		# regularize parameters
		loss += regularization(network['output'], optimization)

		params = layers.get_all_params(network['output'], trainable=True)    


	# calculate and clip gradients
	if "weight_norm" in optimization:
		weight_norm = optimization['weight_norm']
	else:
		weight_norm = None
	grad = calculate_gradient(loss, params, weight_norm=weight_norm)
	  
	# setup parameter updates
	update_op = build_updates(grad, params, optimization, learning_rate)

	# test/validation set 
	if optimization['objective'] == 'lower_bound':
		test_loss, test_prediction = variational_lower_bound(network, placeholders['inputs'], deterministic=False, binary=binary)	
	else:
		test_prediction = layers.get_output(network['output'], deterministic=True)
		test_loss = build_loss(placeholders['targets'], test_prediction, optimization)
			
	# create theano function
	train_fun = theano.function(list(placeholders.values()), [loss, prediction], updates=update_op)
	test_fun = theano.function(list(placeholders.values()), [test_loss, test_prediction])

	return train_fun, test_fun

def variational_lower_bound(network, targets, deterministic=False, binary=True):

	z_mu = layers.get_output(network['encode_mu'], deterministic=deterministic)
	z_logsigma = layers.get_output(network['encode_logsigma'], deterministic=deterministic)
	kl_divergence = 0.5*T.sum(1 + 2*z_logsigma - T.sqr(z_mu) - T.exp(2*z_logsigma), axis=1)

	x_mu = layers.get_output(network['X'], deterministic=deterministic)
	if binary:
		x_mu = T.clip(x_mu, 1e-7, 1-1e-7)
		log_likelihood = T.sum(targets*T.log(1e-10+x_mu) + (1.0-targets)*T.log(1e-10+1.0-x_mu), axis=1)
	else:
		x_logsigma = T.log(T.sqrt(x_mu*(1-x_mu))) #layers.get_output(network['decode_logsigma'], deterministic=deterministic)
		log_likelihood = T.sum(-0.5*T.log(2*np.float32(np.pi))- x_logsigma - 0.5*T.sqr(targets-x_mu)/T.exp(2*x_logsigma),axis=1)

	variational_lower_bound = -log_likelihood - kl_divergence
	prediction = x_mu
	return variational_lower_bound.mean(), prediction


def build_loss(targets, prediction, optimization):
	""" setup loss function with weight decay regularization """

	if optimization["objective"] == 'categorical':
		loss = objectives.categorical_crossentropy(prediction, targets)

	elif optimization["objective"] == 'binary':
		prediction = T.clip(prediction, 1e-7, 1-1e-7)
		loss = -(targets*T.log(prediction) + (1.0-targets)*T.log(1.0-prediction))
		# loss = objectives.binary_crossentropy(prediction[:,loss_index], targets[:,loss_index])

	elif (optimization["objective"] == 'squared_error'):
		loss = objectives.squared_error(prediction, targets)

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


def build_updates(grad, params, optimization, learning_rate):
	""" setup optimization algorithm """

	if optimization['optimizer'] == 'sgd':
		update_op = updates.sgd(grad, params, learning_rate=learning_rate) 
 
	elif optimization['optimizer'] == 'nesterov_momentum':
		if momenum in optimization:
			momentum = optimization['momentum']
		else:
			momentum = 0.9
		update_op = updates.nesterov_momentum(grad, params, learning_rate=learning_rate, momentum=momentum)
	
	elif optimization['optimizer'] == 'adagrad':
		update_op = updates.adagrad(grad, params, learning_rate=learning_rate)
	
	elif optimization['optimizer'] == 'rmsprop':
		if 'rho' in optimization:
			rho = optimization['rho']
		else:
			rho = 0.9
		update_op = updates.rmsprop(grad, params, learning_rate=learning_rate, rho=rho)
	
	elif optimization['optimizer'] == 'adam':
		if 'beta1' in optimization:
			beta1 = optimization['beta1']
		else:
			beta1 = 0.9
		if 'beta2' in optimization:
			beta2 = optimization['beta2']
		else:
			beta2 = 0.999
		update_op = updates.adam(grad, params, learning_rate=learning_rate, beta1=beta1, beta2=beta2)
  
	return update_op



#---------------------------------------------------------------------------------------------------------
# saliency and reconstruction

class ModifiedBackprop(object):
	# https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
	def __init__(self, nonlinearity):
		self.nonlinearity = nonlinearity
		self.ops = {}  

	def __call__(self, x):
	   
		cuda_var = theano.sandbox.cuda.as_cuda_ndarray_variable
		x = cuda_var(x)
		tensor_type = x.type
		
		if tensor_type not in self.ops:
			inputs = tensor_type()
			output_var = cuda_var(self.nonlinearity(inputs))
			op = theano.OpFromGraph([inputs], [output_var])
			op.grad = self.grad
			self.ops[tensor_type] = op

		return self.ops[tensor_type](x)


class GuidedBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		dtype = inp.dtype
		return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)





