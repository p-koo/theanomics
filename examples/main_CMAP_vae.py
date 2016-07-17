import sys, os, time, h5py
import numpy as np
from six.moves import cPickle
sys.setrecursionlimit(10000)

from scipy import stats
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne import layers, init, nonlinearities, utils, regularization, objectives, updates
from lasagne.layers.base import Layer


np.random.seed(727) # for reproducibility


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

class GaussianSampleLayer(layers.MergeLayer):
    def __init__(self, incoming_mu, incoming_logsigma, **kwargs):
        super(GaussianSampleLayer, self).__init__(incomings=[incoming_mu, incoming_logsigma], **kwargs)
        self.srng = RandomStreams(seed=234)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.srng.normal(shape, avg=0.0, std=1).astype(theano.config.floatX)


def vae_model(input_var):
    net = {}
    net['input'] = layers.InputLayer(shape=(None, 970), input_var=input_var)

    # encode layer 1
    net['encode1'] = layers.DenseLayer(net['input'], num_units=3000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['encode1_norm'] = BatchNormLayer(net['encode1'])
    net['encode1_active'] = layers.NonlinearityLayer(net['encode1_norm'], nonlinearity=nonlinearities.rectify)
    net['encode1_dropout'] = layers.DropoutLayer(net['encode1_active'],p=0.5)

    # encode layer
    net['Z_mu'] = layers.DenseLayer(net['encode1_dropout'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=nonlinearities.linear)
    
    net['Z_logsigma'] = layers.DenseLayer(net['encode1_dropout'], num_units=11350, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=nonlinearities.linear)
    net['Z'] = GaussianSampleLayer(net['Z_mu'], net['Z_logsigma'])

    # encode layer 1
    net['decode1'] = layers.DenseLayer(net['Z'], num_units=3000, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=None)
    net['decode1_norm'] = BatchNormLayer(net['decode1'])
    net['decode1_active'] = layers.NonlinearityLayer(net['decode1_norm'], nonlinearity=nonlinearities.rectify)
    net['decode1_dropout'] = layers.DropoutLayer(net['decode1_active'],p=0.5)


    # encode layer
    net['X_mu'] = layers.DenseLayer(net['decode1_dropout'], num_units=970, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=nonlinearities.linear)
    net['X_logsigma'] = layers.DenseLayer(net['decode1_dropout'], num_units=970, W=init.GlorotUniform(), 
                                      b=init.Constant(0.05), nonlinearity=nonlinearities.linear)
    net['X'] = GaussianSampleLayer(net['X_mu'], net['X_logsigma'])
    return net


def batch_generator(X, y, batch_size=128, shuffle=True):
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X)-batch_size+1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batch_size]
        else:
            excerpt = slice(start_idx, start_idx+batch_size)
        yield X[excerpt], y[excerpt]



# data file and output files
outputname = 'vae'
datapath='/home/peter/Data/CMAP'
filepath = os.path.join(datapath, 'Results', outputname)

trainmat = h5py.File(os.path.join(datapath, 'data_set.hd5f'), 'r')
filepath = os.path.join(datapath, 'Results', outputname)
landmark= np.array(trainmat['landmark']).astype(np.float32)
nonlandmark = np.array(trainmat['nonlandmark']).astype(np.float32)

def normalize_data(landmark, mean_landmark, std_landmark, num_samples):
    landmark = (landmark - np.outer(mean_landmark,np.ones(num_samples)))/np.outer(std_landmark,np.ones(num_samples))
    landmark = landmark.transpose([1,0])
    return landmark.astype(np.float32)
mean_nonlandmark = np.mean(nonlandmark, axis=1)
std_nonlandmark = np.std(nonlandmark, axis=1)
nonlandmark = normalize_data(nonlandmark, mean_nonlandmark, std_nonlandmark, nonlandmark.shape[1])

mean_landmark = np.mean(landmark, axis=1)
std_landmark = np.std(landmark, axis=1)
landmark = normalize_data(landmark, mean_landmark, std_landmark, landmark.shape[1])

split=10000
test_landmark = landmark[:split]
test_nonlandmark = nonlandmark[:split]
landmark = landmark[split:]
nonlandmark = nonlandmark[split:]


# setup model
input_var = T.dmatrix('landmark')
target_var = T.dmatrix('nonlandmark')
network = vae_model(input_var)

z_mu = layers.get_output(network['Z_mu'], deterministic=False)
z_logsigma = layers.get_output(network['Z_logsigma'], deterministic=False)
z_logsigma = T.abs_(z_logsigma)
x_mu = layers.get_output(network['X_mu'], deterministic=False)
x_logsigma = layers.get_output(network['X_logsigma'], deterministic=False)
x_logsigma = T.abs_(x_logsigma)
kl_divergence = 0.5*T.sum(1 + 2*z_logsigma - T.sqr(z_mu-target_var) - T.exp(2*z_logsigma), axis=1)
log_likelihood = T.sum(-0.5*T.log(2*np.float32(np.pi))- x_logsigma - 0.5*T.sqr(input_var-x_mu)/T.exp(2*x_logsigma),axis=1)
variational_lower_bound = -.1*log_likelihood - kl_divergence
loss = variational_lower_bound.mean()

# weight-decay regularization
#all_params = layers.get_all_params(network['output'], regularizable=True)
#l1_penalty = regularization.apply_penalty(all_params, regularization.l1) * 1e-5
#l2_penalty = regularization.apply_penalty(all_params, regularization.l2) * 1e-6        
#loss = loss + l1_penalty + l2_penalty 


# setup updates
learning_rate_schedule = {
0: 0.001
#2: 0.01,
#5: 0.001,
#15: 0.0001
}
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

all_params = layers.get_all_params(network['X'], trainable=True, deterministic=False)
#updates = updates.nesterov_momentum(loss, all_params, learning_rate=learning_rate, momentum=0.9)
updates = updates.adam(loss, all_params, learning_rate=learning_rate)

# setup cross-validation
test_prediction = layers.get_output(network['Z'], deterministic=True)
test_loss = objectives.squared_error(test_prediction, target_var)
test_loss = test_loss.mean()

# compile theano functions
train_fn = theano.function([input_var, target_var], loss, updates=updates)
valid_fn = theano.function([input_var, target_var], [test_loss, test_prediction])


# train model
batch_size = 50     
bar_length = 30     
num_epochs = 500   
verbose = 1
train_performance = []
valid_performance = []
for epoch in range(num_epochs):
    sys.stdout.write("\rEpoch %d \n"%(epoch+1))

    # change learning rate if on schedule
    if epoch in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[epoch])
        print(" setting learning rate to %.5f" % lr)
        learning_rate.set_value(lr)

    
    train_loss = 0
    start_time = time.time()
    num_batches = landmark.shape[0] // batch_size
    batches = batch_generator(landmark, nonlandmark, batch_size)
    for j in range(num_batches):
        X, y = next(batches)
        loss = train_fn(X,y)
        train_loss += loss
        train_performance.append(loss)

        percent = float(j+1)/num_batches
        remaining_time = (time.time()-start_time)*(num_batches-j-1)/(j+1)
        progress = '='*int(round(percent*bar_length))
        spaces = ' '*(bar_length-int(round(percent*bar_length)))
        sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- loss=%.5f " \
            %(progress+spaces, percent*100, remaining_time, train_loss/(j+1)))
        sys.stdout.flush()
    print "" 
    
    # test current model with cross-validation data and store results
    num_batches = test_landmark.shape[0] // batch_size
    batches = batch_generator(test_landmark, test_nonlandmark, batch_size, shuffle=False)
    test_prediction = np.zeros(test_nonlandmark.shape)
    value = 0
    for j in range(num_batches):
        X, y = next(batches)
        loss = valid_fn(X, y)
        value += loss[0]
        valid_performance.append(loss[0])
        test_prediction[range(j*batch_size, (j+1)*batch_size),:] = loss[1]

    spearman = np.zeros(test_nonlandmark.shape[1])
    pearson = np.zeros(test_nonlandmark.shape[1])
    for i in range(test_nonlandmark.shape[1]):   
        spearman[i] = stats.spearmanr(test_nonlandmark[:,i], test_prediction[:,i])[0]
        pearson[i] = stats.pearsonr(test_nonlandmark[:,i], test_prediction[:,i])[0]
    print("  valid loss:\t\t{:.5f}".format(value/num_batches))
    print("  valid Pearson corr:\t{:.5f}+/-{:.5f}".format(np.mean(pearson), np.std(pearson)))
    print("  valid Spearman corrr:\t{:.5f}+/-{:.5f}".format(np.mean(spearman), np.std(spearman)))
    
    # save model
    savepath = filepath + "_epoch_" + str(epoch+1) + ".pickle"
    all_param_values = layers.get_all_param_values(network['X'])
    f = open(savepath, 'wb')
    cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

