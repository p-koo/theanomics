import theano.tensor as T
import lasagne
from motif_model2 import motif_model


# setup variables
input_var = T.matrix()
target_var = T.ivector()

# build network
layers = motif_model()
network = build_model(layers, input_var)

# setup loss 
objective = 'binary_crossentropy'
deterministic = False
loss, prediction = build_loss(network, input_var, target_var, objective, deterministic)

# calculate gradient with clipping
weight_norm = 10
grad = calculate_gradient(network, loss, weight_norm)

# setup updates
update_params = {'update': 'rmsprop'}
updates = build_fun(network, params, update_params)

# test loss
deterministic = True
loss_pass = build_loss(network, input_var, target_var, objective, deterministic)

# build theano function
accuracy = class_accuracy(prediction, target_var)
train_fun = theano.function([input_var, target_var], [loss, accuracy], updates=updates)
valid_fun = theano.function([input_var, target_var], [loss_pass, accuracy])
