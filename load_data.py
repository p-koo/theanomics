#/bin/python
#!/bin/python
import os, sys, h5py
import numpy as np
import scipy.io
from six.moves import cPickle

"""
Data sets:
	'load_DeepSea',
	'load_MotifSimulation_categorical'
	'load_MotifSimulation_categorical'
"""

def simulation_pickle(filepath, class_type='binary'):
	# setup paths for file handling

	# load training set
	print "loading data from: " + filepath
	f = open(filepath, 'rb')
	print "loading train data"
	train = cPickle.load(f)
	print "loading cross-validation data"
	cross_validation = cPickle.load(f)
	print "loading test data"
	test = cPickle.load(f)
	f.close()

	X_train = train[0].transpose((0,1,2)).astype(np.float32)
	y_train = train[1].astype(np.int32)
	X_val = cross_validation[0].transpose((0,1,2)).astype(np.float32)
	y_val = cross_validation[1].astype(np.int32)
	X_test = test[0].transpose((0,1,2)).astype(np.float32)
	y_test = test[1].astype(np.int32)

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	if class_type == 'categorical':
		y_train = np.argmax(y_train, axis=1)
		y_val = np.argmax(y_val, axis=1)
		y_test = np.argmax(y_test, axis=1)

	train = (X_train, y_train)
	valid = (X_val, y_val)
	test = (X_test, y_test)

	return train, valid, test


def simulation_hdf5(filepath, class_type='binary'):
	# setup paths for file handling

	trainmat = h5py.File(filepath, 'r')
	X_train = np.array(trainmat['trainx']).astype(np.float32)
	y_train = np.array(trainmat['trainy']).astype(np.float32)
	model_train = np.array(trainmat['trainmodel']).astype(np.float32)

	X_val = np.array(trainmat['validx']).astype(np.float32)
	y_val = np.array(trainmat['validy']).astype(np.int32)
	model_val = np.array(trainmat['validmodel']).astype(np.float32)

	X_test = np.array(trainmat['testx']).astype(np.float32)
	y_test = np.array(trainmat['testy']).astype(np.int32)
	model_test = np.array(trainmat['testmodel']).astype(np.float32)

	X_train = np.expand_dims(X_train, axis=3)
	X_val = np.expand_dims(X_val, axis=3)
	X_test = np.expand_dims(X_test, axis=3)

	if class_type == 'categorical':
		y_train = np.argmax(y_train, axis=1)
		y_val = np.argmax(y_val, axis=1)
		y_test = np.argmax(y_test, axis=1)

	train = (X_train, y_train, model_train)
	valid = (X_val, y_val, model_val)
	test = (X_test, y_test, model_test)

	return train, valid, test


def Encode_TF(filepath, tf_index):
	name = ['H1-hESC','HepG2', 'K562', 'combine', 'CTCF', 'all']
	#name = ['HepG2', 'K562', 'GM12878', 'HeLa', 'H1-hESC', 'HepG2-low', 'K562-low', 'GM12878-low', 'HeLa-low']

	dataset = h5py.File(filepath,'r')

	print "loading training data"
	X_train = np.transpose(dataset['/'+name[tf_index]+'/X_train']).transpose([3,2,1,0])
	y_train = np.transpose(dataset['/'+name[tf_index]+'/Y_train']).transpose([1,0]) 
	print X_train.shape

	print "loading validation data"  
	X_valid = np.transpose(dataset['/'+name[tf_index]+'/X_valid']).transpose([3,2,1,0])
	y_valid = np.transpose(dataset['/'+name[tf_index]+'/Y_valid']) .transpose([1,0])
	print X_valid.shape

	print "loading test data"
	X_test = np.transpose(dataset['/'+name[tf_index]+'/X_test']).transpose([3,2,1,0])
	y_test = np.transpose(dataset['/'+name[tf_index]+'/Y_test']) .transpose([1,0])
	print X_test.shape

	train = (X_train.astype(np.float32), y_train.astype(np.int32))
	valid = (X_valid.astype(np.float32), y_valid.astype(np.int32))
	test = (X_test.astype(np.float32), y_test.astype(np.int32))

	return train, valid, test 


def DeepSea_all(filepath, class_range=range(918), num_include=[]):

	def data_subset(y, class_range, negative=True):
		" gets a subset of data in the class_range"
		data_index = []
		for i in class_range:
			index = np.where(y[:, i] == 1)[0]
			data_index = np.concatenate((data_index, index), axis=0)
		unique_index = np.unique(data_index)
		num_data = y.shape[0]
		non_index = np.array(list(set(range(num_data)) - set(unique_index)))
		if negative:
			index = [unique_index.astype(int), non_index.astype(int)]
		else:
			index = unique_index.astype(int)
		return index


	print "loading training data"
	trainmat = h5py.File(os.path.join(filepath,'train.mat'), 'r')
	y_train = np.transpose(trainmat['traindata'], axes=(1,0))
	index = data_subset(y_train, class_range, negative=False)
	if num_include :
		shuffle = np.sort(np.random.permutation(len(index))[:num_include])
		index = index[shuffle]
  	y_train = y_train[:,class_range]
	y_train = y_train[index,:]
	X_train = np.transpose(trainmat['trainxdata'],axes=(2,1,0)) 
	X_train = X_train[index,:,:]
	X_train = np.expand_dims(X_train, axis=3)
	#train = (X_train.astype(np.float32), y_train.astype(np.int32))
	train = (X_train, y_train)
	
	print X_train.shape

	print "loading validation data"  
	validmat = scipy.io.loadmat(os.path.join(filepath,'valid.mat'))
	y_valid = np.array(validmat['validdata'])
	index = data_subset(y_valid,class_range, negative=False)
	y_valid = y_valid[:, class_range]
	y_valid = y_valid[index,:]
	X_valid = np.transpose(validmat['validxdata'],axes=(0,1,2))  
	X_valid = X_valid[index,:,:]
	X_valid = np.expand_dims(X_valid, axis=3)
	#test = (X_valid.astype(np.float32), y_valid.astype(np.int32))
	print X_valid.shape
	test = (X_valid, y_valid)

	print "loading test data"
	testmat = scipy.io.loadmat(os.path.join(filepath,'test.mat'))
	y_test = np.array(testmat['testdata'])
	index = data_subset(y_test,class_range, negative=False)
	y_test = y_test[:, class_range]
	y_test = y_test[index,:]
	X_test = np.transpose(testmat['testxdata'],axes=(0,1,2)) 
	X_test = X_test[index,:,:]
	X_test = np.expand_dims(X_test, axis=3)
	#valid = (X_test.astype(np.float32), y_test.astype(np.int32))
	valid = (X_test, y_test)
	print X_test.shape

	return train, valid, test 
	

def Basset(filepath, class_range=range(164), num_include=[]):
	"""Loads Basset dataset"""

	def data_subset(y, class_range, negative=True):
		" gets a subset of data in the class_range"
		data_index = []
		for i in class_range:
			index = np.where(y[:, i] == 1)[0]
			data_index = np.concatenate((data_index, index), axis=0)
		unique_index = np.unique(data_index)
		num_data = y.shape[0]
		non_index = np.array(list(set(range(num_data)) - set(unique_index)))
		if negative:
			index = [unique_index.astype(int), non_index.astype(int)]
		else:
			index = unique_index.astype(int)
		return index

	print "loading training data"
	trainmat = h5py.File(os.path.join(filepath,'er.h5'), 'r')
	y_train = np.array(trainmat['train_out'])
	index = data_subset(y_train, class_range, negative=True)
	if num_include:
		N = len(index[0])
		shuffle = np.sort(np.random.permutation(len(index[1]))[:N])
		index = np.hstack([index[0], index[1][shuffle[:N]]])
		shuffle = np.sort(np.random.permutation(len(index))[:num_include])
		index = index[shuffle]
  	y_train = y_train[:,class_range]
	y_train = y_train[index,:]
	X_train = np.transpose(np.array(trainmat['train_in']), axes=(0,1,3,2))
	X_train = X_train[index,:,:,:]
	train = (X_train.astype(np.float32), y_train.astype(np.int32))

	print "loading validation data"
	y_valid = np.array(trainmat['valid_out'])
	index = data_subset(y_valid,class_range, negative=False)
	y_valid = y_valid[:, class_range]
	y_valid = y_valid[index,:]
	X_valid = np.transpose(np.array(trainmat['valid_in']), axes=(0,1,3,2))
	X_valid = X_valid[index,:,:,:]
	valid = (X_valid.astype(np.float32), y_valid.astype(np.int32))

	print "loading testing data"
	y_test = np.array(trainmat['test_out'])
	index = data_subset(y_test,class_range, negative=False)
	y_test = y_test[:, class_range]
	y_test = y_test[index,:]
	X_test = np.transpose(np.array(trainmat['test_in']), axes=(0,1,3,2))
	X_test = X_test[index,:,:,:]
	test = (X_test.astype(np.float32), y_test.astype(np.int32))
	
	return train, valid, test