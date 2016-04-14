#/bin/python
import sys
import os
import numpy as np
sys.path.append('..')
from src import NeuralNet
from src import train as fit
from src import make_directory 
from models import load_model
from data import load_data
from visualize import plot_loss, plot_roc_all, plot_pr_all
from utils import get_performance, get_layer_activity
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'MotifSimulation_binary'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'N=100000_S=200_M=10_G=20_data.pickle')
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

# load model parameters
model_name = "binary_genome_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)

#-------------------------------------------------------------------------------------
# load performance results

outputname = 'binary'
filepath = os.path.join(datapath, 'Results', outputname)
savepath = filepath + "_best.pickle"
nnmodel.set_best_parameters(savepath)

savepath = filepath + "_train_performance.pickle"
train_cost, train_metric, train_metric_std, train_roc, trian_pr = get_performance(savepath)

savepath = filepath + "_cross-validation_performance.pickle"
valid_cost, valid_metric, valid_metric_std, valid_roc, valid_pr = get_performance(savepath)

savepath = filepath + "_test_all_performance.pickle"
test_cost, test_metric, test_metric_std, test_roc, test_pr = get_performance(savepath)

#-------------------------------------------------------------------------------------
# plot optimization metrics during training

# plot loss
fig, plt = plot_loss([train_cost, valid_cost, test_cost])
plt.show()
fig.savefig('test.pdf', format='pdf')

#-------------------------------------------------------------------------------------
# plot performance metrics

savepath = filepath + "_test_performance.pickle"
final_cost, final_metric, final_metric_std, final_roc, final_pr = get_performance(savepath)

fig, plt = plot_roc_all(final_roc)
plt.show()
#fig.savefig('test.pdf', format='pdf')

fig, plt = plot_roc_all(final_pr)
plt.show()
#fig.savefig('test.pdf', format='pdf')

#-------------------------------------------------------------------------------------
# plot wieghts for a given convolutional layer

# find convolution layers
network = nnmodel.network
keys = network.keys()
filter_layers = []
for key in keys:
    if 'conv' in key:
        if hasattr(network[key], 'W'):
            filter_layers.append(key) 
filter_layers

for layer in filter_layers:
    
    # plot weights
    layer = network[layer]
    figs, axes = plot_conv_weights(layer)
    figs.tight_layout()
    plt.subplots_adjust(wspace=.001)

    # plot activity at a given convolutional layer
    layer = network[layer]
    x = np.expand_dims(test[0][0].astype(np.float32), axis=0)
    activity = getlayer_activity(layer, x)
    plot_conv_activity(activity)

    # plot activity in a single map
    activity = np.squeeze(activity)
    fig = plt.figure()
    plt.imshow(activity, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


layer = network['dense']
weights = layer.W.get_value()
plot_weights(weights)

#----------------------------------------------------------------------------------------
# print sequence logos

layer = network['conv1']
W =  np.squeeze(layer.W.get_value())
weights = W[0]

weights = weights/sum()

np.savetxt('test.table', W[0], delimiter='\t')

"""
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

X, _ = load(test=True)
y_pred = net1.predict(X)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()
"""
#----------------------------------------------------------------------------------------------------



















