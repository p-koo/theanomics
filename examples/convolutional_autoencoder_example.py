import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNet
from src import train as fit
from src import make_directory 
from models import load_model
from data import load_data

import lasagne
from lasagne import layers
from lasagne import init, nonlinearities
import theano.tensor as T
import theano
from lasagne import updates, objectives
from lasagne.layers import get_output, get_output_shape, get_all_params, get_all_layers
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'MotifSimulation_binary'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'N=100000_S=200_M=10_G=20_data.pickle')
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

#-------------------------------------------------------------------------------------


# build model
input_var = T.tensor4('input')
target_var = T.dmatrix('codes')

l_in = layers.InputLayer(shape=shape, input_var=input_var, name='input')
l_conv1 = layers.Conv2DLayer(l_in, num_filters=100, filter_size=(9,1), W=init.GlorotUniform(), 
							  nonlinearity=None, b=None, pad='same', name='conv1')
l_bias1 = layers.BiasLayer(l_conv1, b=init.Constant(0.05))
l_nonlin1 = layers.NonlinearityLayer(l_bias1, nonlinearity=nonlinearities.rectify)
l_dense2 = layers.DenseLayer(l_nonlin1, num_units=20, W=init.GlorotUniform(), name='inter')
l_bias2 = layers.BiasLayer(l_dense2, init.Constant(0.05))
l_nonlin2 = layers.NonlinearityLayer(l_bias2, nonlinearity=nonlinearities.tanh)
l_enc = layers.NonlinearityLayer(l_nonlin2, nonlinearity=None)
l_dec = layers.InverseLayer(l_enc, l_dense2, name='decode')
l_bias3 = layers.BiasLayer(l_dec, init.Constant(0.05))
l_nonlin3 = layers.NonlinearityLayer(l_bias3, nonlinearity=nonlinearities.tanh)
l_deconv4 = layers.InverseLayer(l_nonlin3, l_conv1, name='deconv')
l_bias4 = layers.BiasLayer(l_deconv4, init.Constant(0.05))
l_out = layers.NonlinearityLayer(l_bias4, nonlinearity=nonlinearities.rectify)

#l_out = layers.ReshapeLayer(l_nonlin4, shape=(([0],-1)), name='out')


prediction = get_output(l_out)
train_loss = objectives.squared_error(prediction, input_var)
train_loss = train_loss.mean()

valid_prediction = get_output(l_out, deterministic=True)
valid_loss = objectives.squared_error(valid_prediction, input_var)
valid_loss = valid_loss.mean()

params = get_all_params(l_out, trainable=True)
update_op = updates.adam(train_loss, params, learning_rate=1E-4)

train_function = theano.function([input_var], train_loss, updates=update_op)
valid_function = theano.function([input_var], valid_loss)


def batch_generator(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('float32')

batch_size = 64
num_train_batches = train[0].shape[0] // batch_size
train_batches = batch_generator(train[0], train[1], batch_size)

n_epochs = 20
for e in range(n_epochs):
    for index in range(num_train_batches):
        X, y = next(train_batches)
        train_loss = train_function(X)
    print("train: %f" % train_loss)



full = theano.function([input_var], layers.get_output(l_nonlin4), allow_input_downcast=True)
encode = theano.function([input_var], layers.get_output(l_enc), allow_input_downcast=True)



out_expr = layers.get_output(l_out, {l_enc:target_var})
fn = theano.function([target_var, l_in.input_var], out_expr, allow_input_downcast=True)


orig_image = np.random.randn(1,1,5,5)
zero_image=np.zeros((1,1,5,5))
code = encode(orig_image)

print("Reconstruction with Image = Zeros")
print fn(code,zero_image)
print("Reconstruction with Image = Original Image")
print fn(code,orig_image)
print("Full pass through Autoencoder")
print full(orig_image)
