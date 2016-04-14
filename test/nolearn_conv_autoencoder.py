
from lasagne import layers
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T

import sys
import os
import urllib
import gzip
import cPickle
from PIL import Image

from nnbase.layers import Unpool2DLayer
### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
from nnbase.shape import ReshapeLayer

from nnbase.utils import FlipBatchIterator


assert len(sys.argv)==2, "single argument is basefilename of model output"

model_name = sys.argv[1]

fname = 'mnist.pkl.gz'
if not os.path.isfile(fname):
    testfile = urllib.URLopener()
    testfile.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", fname)
f = gzip.open(fname, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
X, y = train_set
X = X.reshape((-1, 1, 28, 28))
mu, sigma = np.mean(X.flatten()), np.std(X.flatten())

print "mu, sigma:", mu, sigma

X_normalized = (X - mu) / sigma

# we need our target to be 1 dimensional
X_out = X_normalized.reshape((X_normalized.shape[0], -1))

conv_filters = 32
deconv_filters = 32
filter_sizes = 7
epochs = 20
encode_size = 40
ae = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv', layers.Conv2DLayer),
        ('pool', layers.MaxPool2DLayer),
        ('flatten', ReshapeLayer),  # output_dense
        ('encode_layer', layers.DenseLayer),
        ('hidden', layers.DenseLayer),  # output_dense
        ('unflatten', ReshapeLayer),
        ('unpool', Unpool2DLayer),
        ('deconv', layers.Conv2DLayer),
        ('output_layer', ReshapeLayer),
        ],
    y_tensor_type=T.dmatrix, # That screwed me over. It took 1 hour of debugging to realize this is needed when float64.
    input_shape=(None, 1, 28, 28),
    conv_num_filters=conv_filters, conv_filter_size = (filter_sizes, filter_sizes),
    conv_border_mode="valid",
    conv_nonlinearity=None,
    pool_pool_size=(2, 2),
    flatten_shape=(([0], -1)), # not sure if necessary?
    encode_layer_num_units = encode_size,
    hidden_num_units= deconv_filters * (28 + filter_sizes - 1) ** 2 / 4,
    unflatten_shape=(([0], deconv_filters, (28 + filter_sizes - 1) / 2, (28 + filter_sizes - 1) / 2 )),
    unpool_ds=(2, 2),
    deconv_num_filters=1, deconv_filter_size = (filter_sizes, filter_sizes),
    deconv_border_mode="valid",
    deconv_nonlinearity=None,
    output_layer_shape = (([0], -1)),
    update_learning_rate = 0.01,
    update_momentum = 0.975,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    regression=True,
    max_epochs= epochs,
    verbose=1
)

print "starting training"
sys.stdout.flush()
# Note: smartass nolearn catches KeyboardInterrupt
# https://github.com/dnouri/nolearn/commit/5caf66c83eaac814a4b91ef1f3352bcfc52e62e2
ae.fit(X_normalized, X_out)

print "finished training, hopefully not with a caught KeyboardInterrupt, because I'm saving now"
sys.stdout.flush()
sys.setrecursionlimit(10000)
cPickle.dump(ae, open(model_name+'.pkl','w'))
ae.save_weights_to(model_name+'.np')

