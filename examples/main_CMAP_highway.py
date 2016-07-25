
import lasagne as nn
import sys, os
import numpy as np
from lasagne import layers, init, nonlinearities, utils, regularization, objectives, updates
from six.moves import cPickle
sys.setrecursionlimit(10000)
import theano
import theano.tensor as T
from lasagne.layers.base import Layer
from scipy import stats
import time
import h5py
np.random.seed(727) # for reproducibility


class MultiplicativeGatingLayer(nn.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]


def highway_dense(incoming, Wh=init.Orthogonal(), bh=init.Constant(0.0),
                  Wt=init.Orthogonal(), bt=init.Constant(-4.0),
                  nonlinearity=nonlinearities.rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    # regular layer
    l_h = layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh,
                               nonlinearity=nonlinearity)
    # gate layer
    l_t = layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                               nonlinearity=T.nnet.sigmoid)
    
    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)

def build_model(input_var, batch_size=100,
                num_hidden_units=970, num_hidden_layers=25):

    l_in = layers.InputLayer(shape=(batch_size, 970), input_var=input_var)

    # first, project it down to the desired number of units per layer
    l_hidden1 = layers.DenseLayer(l_in, num_units=num_hidden_units)
    #l_hidden1 = layers.DropoutLayer(l_hidden1, p=0.3)
    
    # then stack highway layers on top of this
    l_current = l_hidden1
    for k in range(num_hidden_layers - 1):
        l_current = highway_dense(l_current)
    
    l_hidden2 = layers.DenseLayer(l_current, num_units=2000)
    #l_hidden2 = layers.DropoutLayer(l_hidden2, p=0.5)
    
    # finally add an output layer
    l_out = layers.DenseLayer( l_hidden2, num_units=11350, nonlinearity=nonlinearities.linear)
    
    return l_out


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


#def main(trainmat, filepath):

# data file and output files
outputname = 'highway_corr2'
datapath='/home/peter/Data/CMAP'
filepath = os.path.join(datapath, 'Results', outputname)

trainmat = h5py.File(os.path.join(datapath, 'data_set.hd5f'), 'r')
filepath = os.path.join(datapath, 'Results', outputname)
landmark= np.array(trainmat['landmark']).astype(np.float32)
nonlandmark = np.array(trainmat['nonlandmark']).astype(np.float32)
shuffle_index = np.random.permutation(100000)
landmark = landmark[:,shuffle_index]
nonlandmark = nonlandmark[:,shuffle_index]


def normalize_data(landmark, mean_landmark, std_landmark, num_samples):
    landmark = (landmark - mean_landmark)/std_landmark
    landmark = landmark.transpose([1,0])
    return landmark.astype(np.float32)


#landmark = landmark.transpose([1,0]).astype(np.float32)
#nonlandmark = nonlandmark.transpose([1,0]).astype(np.float32)


split=10000
test_landmark = landmark[:split]
test_nonlandmark = nonlandmark[:split]
landmark = landmark[split:]
nonlandmark = nonlandmark[split:]


mean_landmark = np.mean(landmark)
std_landmark = np.std(landmark)
landmark = normalize_data(landmark, mean_landmark, std_landmark, landmark.shape[1])
nonlandmark = normalize_data(nonlandmark, mean_landmark, std_landmark, nonlandmark.shape[1])


mean_landmark = np.mean(test_landmark)
std_landmark = np.std(test_landmark)
test_landmark = normalize_data(test_landmark, mean_landmark, std_landmark, test_landmark.shape[1])
test_nonlandmark = normalize_data(test_nonlandmark, mean_landmark, std_landmark, nonlandmark.shape[1])


# setup model
input_var = T.dmatrix('landmark')
network = build_model(input_var, batch_size=100, num_hidden_units=500, num_hidden_layers=50)


target_var = T.dmatrix('nonlandmark')
prediction = layers.get_output(network, deterministic=False)
loss_corr = T.sum(target_var*prediction)/T.sqrt(T.sum(prediction**2)*T.sum(target_var**2))
loss = -loss_corr.mean()

#prediction = layers.get_output(network, deterministic=False)
loss_landmark = objectives.squared_error(prediction, target_var)
loss += loss_landmark.mean()

#loss_nonlandmark = objectives.squared_error(prediction, target_var)
#loss += loss_nonlandmark.mean()

# weight-decay regularization
#all_params = layers.get_all_params(network, regularizable=True)
#l1_penalty = regularization.apply_penalty(all_params, regularization.l1) * 1e-6
#l2_penalty = regularization.apply_penalty(all_params, regularization.l2) * 1e-6        
#loss = loss + l2_penalty 


# setup updates
learning_rate_schedule = {
0: 0.001
#2: 0.01,
#5: 0.001,
#15: 0.0001
}
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))

all_params = layers.get_all_params(network, trainable=True, deterministic=False)
#updates = updates.nesterov_momentum(loss, all_params, learning_rate=learning_rate, momentum=0.9)
updates = updates.adam(loss, all_params, learning_rate=learning_rate)

# setup cross-validation
test_prediction = layers.get_output(network, deterministic=True)
test_loss = objectives.squared_error(test_prediction, target_var)
test_loss = test_loss.mean()

# compile theano functions
train_fn = theano.function([input_var, target_var], loss, updates=updates)
valid_fn = theano.function([input_var, target_var], [test_loss, test_prediction])


# train model
batch_size = 100     
bar_length = 30     
num_epochs = 100   
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
    savepath = filepath + "epoch_" + str(epoch+1) + ".pickle"
    all_param_values = layers.get_all_param_values(network)
    f = open(savepath, 'wb')
    cPickle.dump(all_param_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

