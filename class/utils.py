#!/bin/python

import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer
from lasagne.layers import BatchNormLayer, ParametricRectifierLayer, NonlinearityLayer
from lasagne.layers import DropoutLayer, get_output, get_all_params
from lasagne.nonlinearities import softmax, sigmoid, tanh, rectify, LeakyRectify
from lasagne.init import GlorotUniform Constant
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.layers import get_output, get_all_params


from lasagne.objectives import binary_crossentropy, binary_crossentropy, squared_error
from lasagne.updates import sgd, nesterov_momentum, adagrad,
from lasagne.updates import rmsprop, adam, norm_constraint


def base_layer(layer, network=[]):

  # Input layer
    default = 1;
    if 'default' in layer:
        if layer['default'] == False:
            default = 0;

    # input layer
    if layer['layer'] == 'input':
        network = InputLayer(layer['shape'], input_var=layer['input_var'])

    # dense layer
    elif layer['layer'] == 'dense'
        if default:
            network = DenseLayer(network,
                                num_units=layer['num_units'])
        else:
            network = DenseLayer(network,
                                num_units=layer['num_units'],
                                W=layer['W'],
                                b=layer['b'])

    # convolution layer
    elif layer['layer'] == 'convolution':
        if default:
            network = Conv1DLayer(network,
                                  num_filters=layer['num_filters'],
                                  filter_size=layer['filter_size'])
        else:
            network = Conv1DLayer(network,
                                  num_filters = layer['num_filters'],
                                  filter_size = layer['filter_size'],
                                  W=layer['W'],
                                  b=layer['b'])

    return network


def build_model(layers, input_var):

    # loop to build each layer of network
    network = []
    for layer in layers:

        # create base layer
        network = base_layer(layer, network)
                
        # add Batch normalization layer
        if 'norm' in layer:
            if 'norm' == 'batch':
                network = BatchNormLayer(network)

        # add activation layer
        if 'activation' in layer:
            if layer['activation'] == 'prelu':
                network = ParametricRectifierLayer(network,
                                                  alpha=Constant(0.25),
                                                  shared_axes='auto')
            else:
                network = NonlinearityLayer(network, nonlinearity=layer['activation'])

        # add dropout layer
        if 'dropout' in layer:
            DropoutLayer(network, p=layer['dropout'])

        # add max-pooling layer
        if layer['layer'] == 'convolution':            
            network = MaxPool1DLayer(network, pool_size=layer['pool_size'],stride=None)

    return network



def build_loss(network, input_var, target_var, objective, deterministic=False):

  # setup training loss
  predict = get_output(network, input_var, deterministic = deterministic)
  
  # create loss function
  return eval("T.mean(" + objective + "(predict, target_var))"), predict



def calculate_gradient(network, loss, weight_norm):

    # calculate gradients
    params = get_all_params(network, trainable=True)
    grad = T.grad(loss, params)

    # gradient clipping option
    if weight_norm > 0:
        grad = norm_constraint(grad, weight_norm)

    return grad, params



def build_fun(network, params, update_params):

    if update_params['update'] == 'sgd':
        updates = sgd(scaled_grad, params, learning_rate=update_params['learning_rate']) 
 
    elif update_params['update'] == 'nersterov_momentum':
        updates = nersterov_momentum(scaled_grad, params, 
                                    learning_rate=update_params['learning_rate'], 
                                    momentum=update_params['momentum'])
    
    elif update_params['update'] == 'adagrad':
        if len(objective) < 2:
            updates = adagrad(scaled_grad, params)
        else:
            updates = adagrad(scaled_grad, params, 
                              learning_rate=update_params['learning_rate'], 
                              epsilon=update_params['epsilon'])

    elif update_params['update'] == 'rmsprop':
        if len(objective) < 2:
            updates = rmsprop(scaled_grad, params)
        else:
            updates = rmsprop(scaled_grad, params, 
                              learning_rate=update_params['learning_rate'], 
                              rho=update_params['rho'], 
                              epsilon=update_params['epsilon'])
    
    elif update_params['update'] == 'adam':
        if len(objective) < 2:
            updates = adam(scaled_grad, params)
        else:
            updates = adam(scaled_grad, params, 
                            learning_rate=update_params['learning_rate'], 
                            beta1=update_params['beta1'], 
                            beta2=update_params['beta2'], 
                            epsilon=update['epsilon'])
  
    return updates



def class_accuracy(prediction, target_var):
    accuracy = T.mean(T.eq(prediction, target_var))
    return accuracy



# For training, we want to sample examples at random in small batches
def batch_gen(X, y, N):
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


