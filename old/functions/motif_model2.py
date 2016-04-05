#!/bin/python

def motif_model():
	# create model
	layer1 = {'layer': 'input', 
			  'input_var': 'input_var'
	          'shape': (None, 1000, 4)}
	layer2 = {'layer': 'convolution', 
	          'num_filters': 300, 
	          'filter_size': 19,
	          'W': 'glorotuniform',
	          'b': None,
	          'default': True,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'maxpool': 3}
	layer3 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': 8,
	          'default': True,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'maxpool': 3}
	layer4 = {'layer': 'convolution', 
	          'num_filters': 200, 
	          'filter_size': 8,
	          'default': True,
	          'W': 'glorotuniform',
	          'b': None,
	          'norm': 'batch', 
	          'activation': 'prelu',
	          'maxpool': 3}
	layer5 = {'layer': 'dense', 
	          'num_units': 1000, 
	          'default': True,
	          'W': 'glorotuniform',
	          'b': 'const', 
	          'dropout': .5,
	          'norm': 'batch',
	          'activation': 'prelu'}
	layer6 = {'layer': 'dense', 
	          'num_units': 919, 
	          'default': True,
	          'W': 'glorot-uniform',
	          'b': 'const',
	          'noise': 'dropout',
	          'activation': 'sigmoid'}

	layers = [layer1, layer2, layer3, layer4, layer5, layer6]
	return layers