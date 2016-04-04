
import sys
import os
import numpy as np
import theano
import theano.tensor as T

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_MotifSimulation
from train_utils import batch_generator, early_stopping, epoch_train, print_progress, prediction_accuracy
from file_utils import make_directory, save_model, load_model, save_train_performance
from model_utils import build_cost, calculate_gradient, optimizer, build_model
from lasagne.layers import get_all_params
sys.path.append('/home/peter/Code/GenomeMotifs/models')
from models import genome_motif_simple_model

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=1)
num_data, dim, sequence_length,_ = train[0].shape
num_labels = max(train[1])+1   # number of labels (output units)

#-------------------------------------------------------------------------------------
# build model

optimization = {"objective": "categorical",
                "optimizer": "sgd", 
                "learning_rate": 0.1}


print "building theano model function"

layers, input_var, target_var = genome_motif_simple_model(train[0], train[1])
network = build_model(layers, input_var)

cost, prediction = build_cost(network, target_var, objective=optimization["objective"])

params = get_all_params(network, trainable=True)    
grad = calculate_gradient(network, cost, params, weight_norm=10)
updates = optimizer(grad, params, optimization)

objective = optimization["objective"]
test_cost, test_prediction = build_cost(network, target_var, objective, deterministic=True)
test_accuracy = prediction_accuracy(test_prediction, target_var, objective)

train_fun = theano.function([input_var, target_var], [cost, test_accuracy], updates=updates)
test_fun = theano.function([input_var, target_var], [test_cost, test_accuracy])


#-------------------------------------------------------------------------------------
# train model

batch_size = 500
num_epochs = 100
patience = 0


print("Starting training...")

num_train_batches = len(train[0]) // batch_size
num_valid_batches = len(valid[0]) // batch_size
train_batches = batch_generator(train[0], train[1], batch_size)
valid_batches = batch_generator(valid[0], valid[1], batch_size)


save = 'all' # final
filename = 'test'
dirpath = '/home/peter/Code/GenomeMotifs/Results'
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
for epoch in range(num_epochs):
    
    train_cost, train_accuracy = epoch_train(train_fun, train_batches, num_train_batches, 1)

    valid_cost, valid_accuracy = epoch_train(test_fun, valid_batches, num_train_batches)
    print_progress(valid_cost, valid_accuracy, "cross-validation", epoch, num_epochs)    

    # store training performance info
    train_loss.append(train_cost)
    train_acc.append(train_accuracy)
    valid_loss.append(valid_cost)
    valid_acc.append(valid_accuracy)

    # save model
    if save == 'all':
        filepath = os.path.join(dirpath, filename+"_"+str(epoch)+".pickle")
        save_model(network, filepath)

    # check for early stopping
    status = early_stopping(valid_loss, patience)
    if not status:
        print "Patience ran out... Early stopping."
        break
            
# get test loss and accuracy
num_test_batches = len(test[0]) // batch_size
test_batches = batch_generator(test[0], test[1], batch_size)
test_cost, test_accuracy = epoch_train(test_fun, test_batches, num_test_batches)
print_progress(test_cost, test_accuracy, "test")    

# save performance
filepath =  os.path.join(dirpath, filename+"_performance.pickle")
performance = [train_loss, train_acc, valid_loss, valid_acc, test_cost, test_accuracy]
save_train_performance(filepath, performance)

