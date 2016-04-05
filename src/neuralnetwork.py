
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
from lasagne.layers import DropoutLayer, get_output, get_all_params, get_output_shape
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.init import Constant
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params, regularize_network_params
from lasagne.updates import sgd, nesterov_momentum, rmsprop, adagrad, adam, total_norm_constraint
sys.path.append('/home/peter/Code/GenomeMotifs/models')


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
        self.monitor = MonitorTraining()
        self.best_parameters = []

    def get_model_name(self):
        return self.model_name

    def get_network(self):
        return self.network


    def get_train_function(self):
        return self.train_fun


    def get_test_function(self):
        return self.train_fun


    def get_model_parameters(self):
        return get_all_param_values(self.network)


    def set_model_parameters(self, all_param_values):
        self.network = set_all_param_values(self.network, all_param_values)


    def save_model(self, filepath):
        print "saving model parameters to: " + filepath
        all_param_values = get_all_param_values(self.network)
        f = open(filepath, 'wb')
        cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def update_best_model_parameters(self):
        min_cost, min_index = self.monitor.get_min_cost("valid")
        if min_index+1 == self.monitor.get_len("valid"):
            print ("better model saved...")
            self.best_parameters = self.get_model_parameters()


    def test_results(self, test):
        test_cost, test_accuracy = self.test_fun(test[0].astype(np.float32), test[1].astype(np.int32)) 
        return test_cost, test_accuracy


    def prediction_accuracy(self, prediction, y):

        objective = self.optimization["objective"]
        if objective == "categorical":
            accuracy = np.mean((np.argmax(prediction, axis=1) == y).astype(np.float))
        elif objective == "binary":
            accuracy = T.mean(T.eq(prediction, y))
        elif objective == "meansquare":
            print "work in progress"
        return accuracy


    def epoch_train(self,  mini_batches, num_batches):        
        
        # set timer for epoch run
        self.monitor.set_start_time(start_time = time.time())

        # train on mini-batch with random shuffling
        epoch_cost = 0
        epoch_accuracy = 0
        for index in range(num_batches):
            X, y = next(mini_batches)
            cost, prediction = self.train_fun(X, y)
            epoch_cost += cost

            accuracy = self.prediction_accuracy(prediction, y)
            epoch_accuracy += accuracy
            
            # plot progress bar
            self.monitor.progress_bar(index, num_batches, epoch_cost, epoch_accuracy)
        print "" 
        epoch_cost /= num_batches
        epoch_accuracy /= num_batches
        return epoch_cost, epoch_accuracy


    def train(self, train, valid, test, batch_size=128, num_epochs=500, 
                    patience=10, save='all', filepath='.', verbose=1):

        self.monitor.set_verbose(verbose)

        # setup generator for mini-batches
        num_train_batches = len(train[0]) // batch_size
        train_batches = batch_generator(train[0], train[1], batch_size)

        # train model
        for epoch in range(num_epochs):
            if verbose == 1:
                sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))

            # training set
            train_cost, train_accuracy = self.epoch_train( train_batches, num_train_batches)

            # validation set
            valid_cost, valid_prediction = self.test_results(valid)
            valid_accuracy = self.prediction_accuracy(valid_prediction, valid[1])

            # store training performance info
            self.monitor.append_values(train_cost, train_accuracy, 'train')
            self.monitor.append_values(valid_cost, valid_accuracy, 'valid')
            self.monitor.print_results("valid", epoch, num_epochs) 
            self.update_best_model_parameters()

            # check for early stopping
            status = self.monitor.early_stopping(patience)
            if not status:
                break

            # save model
            if save == 'all':
                savepath = filepath + "_" + str(epoch) + ".pickle"
                self.save_model(savepath)
                    
        # get test cost and accuracy
        self.set_model_parameters(self.best_parameters)
        savepath = filepath + "_best.pickle"
        self.save_model(savepath)

        test_cost, test_prediction = self.test_results(test)
        test_accuracy = self.prediction_accuracy(test_prediction, test[1])
        self.monitor.append_values(test_cost, test_accuracy, 'test')
        self.monitor.print_results("test")   
        savepath =  filepath + "_performance.pickle"
        self.monitor.save(savepath)


#----------------------------------------------------------------------------------------------------
# Monitor traning class
#----------------------------------------------------------------------------------------------------

class MonitorTraining():
    def __init__(self):
        self.train_cost = []
        self.valid_cost = []
        self.test_cost = []
        self.train_accuracy = []
        self.valid_accuracy = []
        self.test_accuracy = []
        self.verbose = 1

    def append_values(self, cost, accuracy, name):
        if name == "train":
            self.train_cost.append(cost)
            self.train_accuracy.append(accuracy)
        if name == "test":
            self.test_cost.append(cost)
            self.test_accuracy.append(accuracy)
        if name == "valid":
            self.valid_cost.append(cost)
            self.valid_accuracy.append(accuracy)


    def set_verbose(self, verbose):
        self.verbose = verbose


    def set_start_time(self, start_time):
        self.start_time = start_time


    def get_values(self, name):
        if name == "train":
            return self.train_cost, self.train_accuracy
        if name == "test":
            return self.test_cost, self.test_accuracy
        if name == "valid":
            return self.valid_cost, self.valid_accuracy


    def get_last_value(self, name):
        if name == "train":
            return self.train_cost[-1], self.train_accuracy[-1]
        if name == "test":
            return self.test_cost[-1], self.test_accuracy[-1]
        if name == "valid":
            return self.valid_cost[-1], self.valid_accuracy[-1]


    def get_len(self, name):
        if name == "train":
            return len(self.train_cost)
        if name == "test":
            return len(self.test_cost)
        if name == "valid":
            return len(self.valid_cost)


    def get_min_cost(self, name):
        if name == "train":
            min_cost = min(self.train_cost)
            min_index = np.argmin(self.train_cost)
            return min_cost, min_index
        if name == "test":
            min_cost = min(self.test_cost)
            min_index = np.argmin(self.test_cost)
            return min_cost, min_index
        if name == "valid":
            min_cost = min(self.valid_cost)
            min_index = np.argmin(self.valid_cost)
        return min_cost, min_index

    def save(self, filepath):

        performance = [self.train_cost, self.train_accuracy, 
                       self.valid_cost, self.valid_accuracy, 
                       self.test_cost, self.test_accuracy]

        print "saving training performance to:" + filepath
        f = open(filepath, 'wb')
        for data in performance:
            cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    def print_results(self, name, epoch=0, num_epochs=0): 
        if self.verbose == 1:
            #if num_epochs != 0:   
            #    sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
            cost, accuracy = self.get_last_value(name)
            print("  " + name + " cost:\t{:.6f}".format(float(cost)))
            print("  " + name + " accuracy:\t{:.2f} %".format(float(accuracy)*100))


    def progress_bar(self, index, num_batches, cost, accuracy, bar_length=20):
        if self.verbose == 1:
            remaining_time = (time.time()-self.start_time)*(num_batches-index-1)/(index+1)
            percent = (index+1.)/num_batches
            progress = '='*int(round(percent*bar_length))
            spaces = ' '*int(bar_length-round(percent*bar_length))
            sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- cost=%.3f -- accuracy=%.2f%%" \
            %(progress+spaces, percent*100, remaining_time, cost/(index+1), accuracy/(index+1)*100))
            sys.stdout.flush()


    def early_stopping(self, patience):
        min_cost, min_epoch = self.get_min_cost('valid')
        current_cost = self.valid_cost[-1]
        current_epoch = len(self.valid_cost)
        status = True
        if min_cost < current_cost:
            if patience - (current_epoch - min_epoch) < 0:
                status = False
                if self.verbose == 1:
                    print "Patience ran out... Early stopping."
        return status


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
            network = MaxPool2DLayer(network, pool_size=layer['pool_size'])

    return network


def build_cost(network, target_var, objective, deterministic=False):

    prediction = get_output(network, deterministic=deterministic)
    if objective == 'categorical':
        cost = categorical_crossentropy(prediction, target_var)
    elif objective == 'binary':
        cost = binary_crossentropy(prediction, target_var)
        
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