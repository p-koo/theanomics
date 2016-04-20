import sys
import os
import numpy as np
import theano
import theano.tensor as T
import lasagne as nn
import time
import pickle
from PIL import Image
from lasagne.layers import MergeLayer

# ################## Define custom layer for middle of VCAE ##################
# This layer takes the mu and sigma (both DenseLayers) and combines them with
# a random vector epsilon to sample values for the code Z
class VAE_Z_Layer(MergeLayer):
    def __init__(self, epsilon, mu, logsigma, **kwargs):
        self.epsilon = epsilon
        super(VAE_Z_Layer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        mu, logsigma = inputs
        return mu + T.exp(logsigma) * self.epsilon

# ################## Download and prepare the MNIST dataset ##################
def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(255)

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    return X_train, X_val, X_test

# ############################# Output images ################################
def get_picture_array(X, index, shp=(28,28), channels=1):
    ret = (X[index] * 255.).reshape(channels,shp[0],shp[1]) \
            .transpose(2,1,0).clip(0,255).astype(np.uint8)
    if channels == 1:
        ret = ret.reshape(shp[1], shp[0])
    return ret

def get_image_pair(X, Xpr, channels=1, idx=-1):
    mode = 'RGB' if channels == 3 else 'L'
    shp=X[0][0].shape
    i = np.random.randint(X.shape[0]) if idx == -1 else idx
    orig = Image.fromarray(get_picture_array(X, i, shp, channels), mode=mode)
    new_size = (orig.size[0], orig.size[1]*2)
    new_im = Image.new(mode, new_size)
    new_im.paste(orig, (0,0))
    rec = Image.fromarray(get_picture_array(Xpr, i, shp, channels), mode=mode)
    new_im.paste(rec, (0, orig.size[1]))
    return new_im

# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Build Model #################################

def build_vcae(inputvar=None, specstr='', imgshape=(28,28), channels=1, epsilon=None):
    l_input = nn.layers.InputLayer(shape=(None,channels,imgshape[-1], imgshape[1]),
            input_var=inputvar, name='input')
    l_last = l_input
    to_invert=[]
    specs=map(lambda s: s.split('-'), specstr.split(','))
    layerIdx = 1
    for spec in specs:
        if len(spec) == 2 and spec[0] == 'd':
            # do not append, because we don't do the inverse of a dropout
            # layer on the way up
            l_last = nn.layers.DropoutLayer(l_last, p=float(spec[1]), rescale=True,
                    name='dropout{}'.format(layerIdx))
        elif len(spec) == 2:
            nfilt = int(spec[0])
            fsize = int(spec[1])
            l_last = nn.layers.Conv2DLayer(l_last, num_filters=nfilt,
                    filter_size=(fsize,fsize),
                    W=nn.init.GlorotUniform(),
                    nonlinearity=None,
                    b=None,
                    name='conv{}'.format(layerIdx))
            to_invert.append(l_last)
            l_last = nn.layers.BiasLayer(l_last,
                    name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        elif len(spec) == 1:
            l_last = nn.layers.DenseLayer(l_last, num_units=int(spec[0]),
                    W = nn.init.GlorotUniform(),
                    nonlinearity=None,
                    b=None,
                    name='dense{}'.format(layerIdx))
            to_invert.append(l_last)
            l_last = nn.layers.BiasLayer(l_last,
                    name='bias{}'.format(layerIdx))
            l_last = nn.layers.NonlinearityLayer(l_last,
                    nonlinearity=nn.nonlinearities.tanh,
                    name='nl{}'.format(layerIdx))
        layerIdx += 1
    l_mu = nn.layers.DenseLayer(l_last,
            num_units=to_invert[-1].num_units,
            nonlinearity = None,
            W = nn.init.GlorotUniform(),
            name='mu')
    l_logsigma = nn.layers.DenseLayer(l_last,
            num_units=to_invert[-1].num_units,
            nonlinearity = None,
            W = nn.init.GlorotUniform(),
            name='logsigma')
    l_last = VAE_Z_Layer(epsilon, l_mu, l_logsigma,
            name='Z')
    for lay in to_invert[::-1]:
        l_last=nn.layers.InverseLayer(l_last, lay, name='inv_{}'.format(lay.name))
        l_last = nn.layers.BiasLayer(l_last,
                name='inv_bias_{}'.format(lay.name))
        l_last = nn.layers.NonlinearityLayer(l_last,
                nonlinearity=nn.nonlinearities.tanh,
                name='inv_nl_{}'.format(lay.name))
    l_output = nn.layers.ReshapeLayer(l_last, shape=(([0], -1)), name='output')
    return l_output

# ############################## Main program ################################

def log_prior(mu, logsigma):
    return 0.5 * T.sum(1 + 2*logsigma - mu ** 2 - T.exp(2 * logsigma))

def main(num_epochs=500):
    print("Loading data...")
    X_train, X_val, X_test = load_dataset()
    X_train_tgt = X_train.reshape(X_train.shape[0],-1)
    X_val_tgt = X_val.reshape(X_val.shape[0],-1)
    X_test_tgt = X_test.reshape(X_test.shape[0],-1)
    input_var = T.tensor4('inputs')
    target_var = T.dmatrix('targets')
    epsilon = T.dmatrix('epsilon')

    # Create neural network model (layers specified by specstr)
    print("Building model and compiling functions...")
    z_dim = 128
    specstr = '16-3,16-2,{}'.format(z_dim)
    network = build_vcae(input_var, epsilon=epsilon, specstr=specstr)

    # Loss expression has two parts: reconstruction error (here MSE) and
    # the prior loss, as specified in [1]
    prediction = nn.layers.get_output(network)
    l_list = nn.layers.get_all_layers(network)
    l_mu = l_list[next(i for i in xrange(len(l_list)) if l_list[i].name=='mu')]
    l_logsigma = l_list[next(i for i in xrange(len(l_list)) if l_list[i].name=='logsigma')]
    mu = nn.layers.get_output(l_mu)
    logsigma = nn.layers.get_output(l_logsigma)
    kl_div = log_prior(mu, logsigma)
    loss = nn.objectives.squared_error(prediction, target_var).mean() - kl_div
    test_prediction = nn.layers.get_output(network, deterministic=True)
    test_loss = nn.objectives.squared_error(test_prediction,
            target_var).mean() - kl_div

    # ADAM updates
    params = nn.layers.get_all_params(network, trainable=True)
    updates = nn.updates.adam(loss, params)
    train_fn = theano.function([input_var, target_var, epsilon], loss, updates=updates)
    val_fn = theano.function([input_var, target_var, epsilon], test_loss)

    print("Starting training...")
    batch_size = 100
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, X_train_tgt, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets, np.random.randn(batch_size, z_dim))
            train_batches += 1
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, X_val_tgt, batch_size, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets, np.zeros((batch_size, z_dim)))
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, X_test_tgt, batch_size, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets, np.zeros((batch_size, z_dim)))
        test_err += err
        test_batches += 1
    test_err /= test_batches
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err))
    print("Saving")
    fn = 'p_{:.6f}.params'.format(test_err)
    X_comp = X_test[:20]
    X_pred = test_prediction.eval({input_var: X_comp, 
        epsilon: np.zeros((X_comp.shape[0], z_dim))}) \
            .reshape(-1, 1, 28, 28)
    for i in range(20):
        get_image_pair(X_comp, X_pred, idx=i, channels=1).save('output_{}.jpg'.format(i))
    pickle.dump(nn.layers.get_all_param_values(network), open(fn, 'w'))
    

if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
    main(**kwargs)