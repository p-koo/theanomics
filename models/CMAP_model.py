import sys
sys.path.append('..')
import numpy as np
from six.moves import cPickle
import theano
import theano.tensor as T
from lasagne.layers.base import Layer
from lasagne import layers, init, nonlinearities, utils
from src.build_network import build_network


class BatchNormLayer(Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = utils.floatX(epsilon)
        self.alpha = utils.floatX(alpha)

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=False)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

        self.beta = T.cast(self.beta, dtype='floatX')
        self.gamma = T.cast(self.gamma, dtype='floatX')
        self.mean = T.cast(self.mean, dtype='floatX')
        self.inv_std = T.cast(self.inv_std, dtype='floatX')
        
    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized

#---------------------------------------------------------------------------------------------------------------
# residual network

#---------------------------------------------------------------------------------------------------------------
# BIG denoising autoencoder with dropout

"""

def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 1
    net['corrupt1'] = layers.GaussianNoiseLayer(net['input'], sigma=0.1)
    net['encode1'] = layers.DenseLayer(net['corrupt1'], num_units=2000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.01), nonlinearity=None)
    net['encode1_norm'] = BatchNormLayer(net['encode1'])
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1_norm'], nonlinearity=nonlinearities.rectify)
    #net['encode1_active'] = layers.ParametricRectifierLayer(net['encode1_norm'])
    net['encode1_dropout'] = layers.DropoutLayer(net['encode1_active'], p=.5)

    # encode layer 2
    net['corrupt2'] = layers.GaussianNoiseLayer(net['encode1_dropout'], sigma=0.1)
    net['encode2'] = layers.DenseLayer(net['corrupt2'], num_units=4000, W=init.GlorotUniform(), 
                                      b=init.Constant(.01), nonlinearity=None)
    net['encode2_norm'] = BatchNormLayer(net['encode2'])
    net['encode2_active'] = layers.NonlinearityLayer(net['encode2_norm'], nonlinearity=nonlinearities.rectify)
    #net['encode2_active'] = layers.ParametricRectifierLayer(net['encode2_norm'])
    net['encode2_dropout'] = layers.DropoutLayer(net['encode2_active'], p=.5)

    # encode layer 3
    net['corrupt3'] = layers.GaussianNoiseLayer(net['encode2_dropout'], sigma=0.1)
    net['encode3'] = layers.DenseLayer(net['corrupt3'], num_units=6000, W=init.GlorotUniform(), 
                                      b=init.Constant(.01), nonlinearity=None)
    net['encode3_norm'] = BatchNormLayer(net['encode3'])
    net['encode3_active'] = layers.NonlinearityLayer(net['encode3_norm'], nonlinearity=nonlinearities.rectify)
    #net['encode3_active'] = layers.ParametricRectifierLayer(net['encode3_norm'])
    net['encode3_dropout'] = layers.DropoutLayer(net['encode3_active'], p=.5)

    # encode layer
    net['encode'] = layers.DenseLayer(net['encode3_dropout'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(.01), nonlinearity=None)
    net['encode_active'] = layers.NonlinearityLayer(net['encode'], nonlinearity=nonlinearities.linear)

    # decode layer
    net['decode'] = layers.DenseLayer(net['encode_active'], num_units=6000, W=net['encode'].W.T, 
                                      b=init.Constant(.01), nonlinearity=None)
    net['decode_norm'] = BatchNormLayer(net['decode'])
    net['decode_active'] = layers.NonlinearityLayer(net['decode_norm'], nonlinearity=nonlinearities.rectify)
    #net['decode_active'] = layers.ParametricRectifierLayer(net['decode_norm'])

    # decode layer 1
    net['decode3'] = layers.DenseLayer(net['decode_active'], num_units=4000, W=net['encode3'].W.T, 
                                      b=init.Constant(.01), nonlinearity=None)
    net['decode3_norm'] = BatchNormLayer(net['decode3'])
    net['decode3_active'] = layers.NonlinearityLayer(net['decode3_norm'], nonlinearity=nonlinearities.rectify)
    #net['decode3_active'] = layers.ParametricRectifierLayer(net['decode3_norm'])

    # decode layer 1
    net['decode2'] = layers.DenseLayer(net['decode3_active'], num_units=2000, W=net['encode2'].W.T, 
                                      b=init.Constant(.01), nonlinearity=None)
    net['decode2_norm'] = BatchNormLayer(net['decode2'])
    net['decode2_active'] = layers.NonlinearityLayer(net['decode2_norm'], nonlinearity=nonlinearities.rectify)
    #net['decode2_active'] = layers.ParametricRectifierLayer(net['decode2_norm'])

    # decode layer 2
    net['decode1'] = layers.DenseLayer(net['decode2_active'], num_units=970, W=net['encode1'].W.T, 
                                      b=init.Constant(.01), nonlinearity=None)
    net['decode1_norm'] = BatchNormLayer(net['decode1'])
    net['decode1_active'] = layers.NonlinearityLayer(net['decode1_norm'], nonlinearity=nonlinearities.linear)

    net['output'] = net['decode1_active']


    # optimization parameters
    optimization = {"objective": "autoencoder",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization

"""

#---------------------------------------------------------------------------------------------------------------
# denoising autoencoder with dropout


def model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 1
    net['encode1'] = layers.DenseLayer(net['input'], num_units=2000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['encode1_norm'] = BatchNormLayer(net['encode1'])
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1_norm'], nonlinearity=nonlinearities.rectify)

    # encode layer 2
    net['encode2'] = layers.DenseLayer(net['encode1_active'], num_units=4000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['encode2_norm'] = BatchNormLayer(net['encode2'])
    net['encode2_active'] = layers.NonlinearityLayer(net['encode2_norm'], nonlinearity=nonlinearities.rectify)

    # encode layer
    net['encode'] = layers.DenseLayer(net['encode2_active'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['output'] = layers.NonlinearityLayer(net['encode'], nonlinearity=nonlinearities.linear)
    
    """
    
    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 1
    net['corrupt1'] = layers.GaussianNoiseLayer(net['input'], sigma=0.1)
    net['encode1'] = layers.DenseLayer(net['corrupt1'], num_units=2000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['encode1_norm'] = BatchNormLayer(net['encode1'])
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1_norm'], nonlinearity=nonlinearities.rectify)

    # encode layer 2
    net['corrupt2'] = layers.GaussianNoiseLayer(net['encode1_active'], sigma=0.1)
    net['encode2'] = layers.DenseLayer(net['corrupt2'], num_units=4000, W=init.GlorotUniform(), 
                                      b=init.Constant(.05), nonlinearity=None)
    net['encode2_norm'] = BatchNormLayer(net['encode2'])
    net['encode2_active'] = layers.NonlinearityLayer(net['encode2'], nonlinearity=nonlinearities.rectify)

    # encode layer
    net['encode'] = layers.DenseLayer(net['encode2_active'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(.05), nonlinearity=None)
    net['encode_active'] = layers.NonlinearityLayer(net['encode'], nonlinearity=nonlinearities.linear)

    # decode layer
    net['decode'] = layers.DenseLayer(net['encode_active'], num_units=4000, W=net['encode'].W.T, 
                                      b=init.Constant(.01), nonlinearity=None)
    #net['decode_norm'] = BatchNormLayer(net['decode'])
    net['decode_active'] = layers.NonlinearityLayer(net['decode'], nonlinearity=nonlinearities.rectify)

    # decode layer 1
    net['decode2'] = layers.DenseLayer(net['decode_active'], num_units=2000, W=net['encode2'].W.T, 
                                      b=init.Constant(.05), nonlinearity=None)
    #net['decode2_norm'] = BatchNormLayer(net['decode2'])
    net['decode2_active'] = layers.NonlinearityLayer(net['decode2'], nonlinearity=nonlinearities.rectify)

    # decode layer 2
    net['decode1'] = layers.DenseLayer(net['decode2_active'], num_units=970, W=net['encode1'].W.T, 
                                      b=init.Constant(.05), nonlinearity=None)
    net['decode1_active'] = layers.NonlinearityLayer(net['decode1'], nonlinearity=nonlinearities.linear)

    net['output'] = net['decode1_active']

    """
    # optimization parameters
    optimization = {"objective": "autoencoder",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization



#---------------------------------------------------------------------------------------------------------------
# feed forward neural network
"""

def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 
    net['encode1'] = layers.DenseLayer(net['input'], num_units=3000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.01), nonlinearity=None)
    net['encode1_norm'] = BatchNormLayer(net['encode1'])
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1_norm'], nonlinearity=nonlinearities.rectify)
    
    # encode layer 2
    net['encode2'] = layers.DenseLayer(net['encode1_active'], num_units=6000, W=init.GlorotUniform(), 
                                      b=init.Constant(.01), nonlinearity=None)
    net['encode2_norm'] = BatchNormLayer(net['encode2'])
    net['encode2_active'] = layers.NonlinearityLayer(net['encode2_norm'], nonlinearity=nonlinearities.rectify)
    
    # encode layer
    net['encode'] = layers.DenseLayer(net['encode2_active'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(.01), nonlinearity=None)
    net['encode_active'] = layers.NonlinearityLayer(net['encode'], nonlinearity=nonlinearities.linear)    
    net['output'] = net['encode_active']


    # optimization parameters
    optimization = {"objective": "ols",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization
"""

#---------------------------------------------------------------------------------------------------------------

# Mean RIS: 1188564.27

"""
def CMAP_model(shape, num_labels):

    target_var = T.dmatrix('targets')
    input_var = T.dmatrix('inputs')

    net = {}
    net['input'] = InputLayer(shape=(None, 970), input_var=input_var)
    net['dense1'] = DenseLayer(net['input'], num_units=11350, W=GlorotUniform(), 
                                      b=Constant(.0), nonlinearity=None)
    net['dense1_active'] = NonlinearityLayer(net['dense1'], nonlinearity=linear)
    net['output'] = net['dense1_active']

    # optimization parameters
    optimization = {"objective": "ols",
                    "optimizer": "adam",
                    "learning_rate": 0.001,                 
                    "beta1": .9,
                    "beta2": .999,
                    "epsilon": 1e-8,
#                   "weight_norm": 7, 
#                   "momentum": 0.9
                    "l1": 1e-5,
                    "l2": 1e-6
                    }

    return net, input_var, target_var, optimization

"""

