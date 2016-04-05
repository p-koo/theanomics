import sys
import os
import numpy as np
import theano
import theano.tensor as T

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_MotifSimulation
from train_utils import batch_generator, early_stopping, epoch_train, print_progress
from file_utils import make_directory, save_model, set_model_parameters, save_train_performance
from model_utils import load_model, prediction_accuracy
from lasagne.layers import get_all_params
sys.path.append('/home/peter/Code/GenomeMotifs/models')
from supervised_models import genome_motif_simple_model

np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

filename = 'N=100000_S=200_M=10_G=20_data.pickle'
dirpath = '/home/peter/Data/SequenceMotif'
train, valid, test = load_MotifSimulation(filename, dirpath, categorical=1)
"""
batch_size = 500
objective = "categorical"
layers, input_var, target_var = genome_motif_simple_model(train[0], train[1])
network = build_model(layers, input_var)
test_cost, test_prediction = build_cost(network, target_var, objective, deterministic=True)
test_accuracy = prediction_accuracy(test_prediction, target_var, objective)
test_fun = theano.function([input_var, target_var], [test_cost, test_accuracy])

epoch = 3
savename = "mytest"
filepath = os.path.join(dirpath, "Results", savename+"_"+str(epoch)+".pickle")
network = set_model_parameters(network, filepath)
test_cost, test_accuracy = test_fun(test[0].astype(np.float32), test[1].astype(np.int32))
test_accuracy
test_cost

"""

optimization = {"objective": "categorical",
                "optimizer": "sgd", 
                "learning_rate": 0.1,
                "weight_norm": 10}
savename = "mytest"

accuracy = []
cost = []
for epoch in range(4):
	filepath = os.path.join(dirpath, "Results", savename+"_"+str(epoch)+".pickle")

	layers, input_var, target_var = genome_motif_simple_model(train[0], train[1])
	network, train_fun, test_fun = load_model(layers, input_var, target_var, optimization)
	network = set_model_parameters(network, filepath)
	
	test_cost, test_accuracy = test_fun(test[0].astype(np.float32), test[1].astype(np.int32))
	print str(epoch)
	print str(float(test_accuracy))
	print str(float(test_cost))


#-------------------------------------------------------------------------------------
# build model

"""

cost, prediction = build_cost(network, target_var, objective=optimization["objective"])

params = get_all_params(network, trainable=True)    
grad = calculate_gradient(network, cost, params, weight_norm=10)
updates = optimizer(grad, params, optimization)

train_fun = theano.function([input_var, target_var], [cost, test_accuracy], updates=updates)

#-------------------------------------------------------------------------------------
# train model



filename="simulation"
savepath = os.path.join(dirpath, 'Results', filename+"_"+str(epoch)+".pickle")
"""