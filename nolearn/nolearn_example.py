#!/bin/python

import cPickle as pickle
import os
import sys
import time

from matplotlib import pyplot
import numpy as np
import theano
import thano.tensor as T
from lasagne.layers import DropoutLayer, InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, BatchNormLayer
from lasagne import updates
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, TrainSplit, 

sys.setrecursionlimit(10000)  
np.random.seed(42)




batch_size = 128
num_data = len(X_train[0])
num_batches = (num_data + batch_size - 1) // batch_size # np.floor(num_data/batch_size)


X_trian, X_valid, y_train, y_valid = get_train_data(train, valid)

monitor = MonitorPerformance()

net = NeuralNet(
    layers=[
            (InputLayer, {'input_shape': (None, X.shape[1], X.shape[2], X.shape[3])}, 'name': 'input'),
            (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'activation': None, 
                            'W': GlorotUniform(), 'b': None}, 'name': 'conv1'),
            (BatchNormLayer, {'name': 'batch1'}),
            (ParametricRectifyLayer, {'name': 'prelu1'}),        
            (MaxPool2DLayer, {'pool_size': (2, 2), 'name': 'max_pool1'}),
            (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'activation': None, 
                            'W': GlorotUniform(), 'b': None, 'name': 'conv2'}),
            (BatchNormLayer, {'name': 'batch2'}),
            (ParametricRectifyLayer, {'name': 'prelu2'}),        
            (MaxPool2DLayer, {'pool_size': (2, 2), 'name': 'max_pool2'}),
            (DropoutLayer, {'p': 0.3, 'name': 'dropout2'}),
            (DenseLayer, {'num_units': 500, 'nonlinearity': None, 'name': 'fc3'}),
            (ParametricRectifyLayer, {'name': 'prelu3'}),        
            (DropoutLayer, {'p': 0.3, 'name': 'dropout3'}),
            (DenseLayer, {'num_units': num_labels, 'nonlinearity': softmax, 'name': 'output'}),
            ],
        
    # loss function
    objective_loss_function=binary_cross_entropy, # categorical_cross_entropy, squared_error
    objective_l1=0.0,   # changes default parameters
    objective_l2=0.0001,   # changes default parameters

    # optimizer
    update=updates_grad_clip,
    update_weight_norm = 10, 
    # update_learning_rate=theano.shared(float32(0.03)),
    # update_momentum=theano.shared(float32(0.9)),

    #  on_training_started, on_batch_finished, on_epoch_finished, on_training_finished
    on_training_started=[monitor.start_time()], 
    on_batch_finished=[monitor.progress_bar(num_batches)],
    on_training_finished=[monitor.print_results()]
    on_epoch_finished=[
        #AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200),
        ],

    # data handlers
    train_split=TrainSplit(eval_size=0.15),
    batch_iterator_train=BatchIterator(batch_size=batch_size, shuffle=True)
    batch_iterator_test=BatchIterator(batch_size=batch_size)

    # custom scores --> put in AUC-ROC and AUC-PR
    custom_scores=None,
    # custom_scores=[('first output', lambda y1, y2: abs(y1[0,0]-y2[0,0])), ('second output', lambda y1,y2: abs(y1[0,1]-y2[0,1]))]")

    max_epochs=3000,
    verbose=1,          # higher values yield more outputs
    regression=False,   # not a regression problem
    )


net.initialize()
layer_info = PrintLayerInfo()
layer_info(net1)


X = X.astype(np.float32)
y = y.astype(np.float32)

# train with custom training
train_minibatch(train, valid)
with open(filepath, 'wb') as f:
    pickle.dump(params, f, -1)
net = load_weights_from(filepath)

# train with nolearn 
net.fit(X, y)
with open('net.pickle', 'wb') as f:
    pickle.dump(net, f, -1)

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

plot_loss(net)

# Visualizing layer weights
plot_conv_weights(net.layers_[1], figsize=(4, 4))


# Visualizing the layers' activities
x = X[0:1]
plot_conv_activity(net.layers_[1], x)


# Plot occlusion images
plot_occlusion(net, X[:5], y[:5])







