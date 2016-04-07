


#/bin/python
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle as pickle

sys.path.append('..')
from src import NeuralNets
from models import load_model
from data import load_data

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

# load model
model_name = "binary_genome_motif_model"
outputname = 'binary'
filepath = os.path.join(datapath, 'Results', outputname)
savepath = filepath + "_best.pickle"

def load_best_model(model_name, savepath):
	model_layers, input_var, target_var, optimization = load_model(model_name, shape, num_labels)
	nnmodel = NeuralNets(model_layers, input_var, target_var, optimization)
	nnmodel.set_best_model(filepath)
	return nnmodel

# load performance metrics
savepath = os.path.join(datapath, name, "test_performance.pickle")
f = open(savepath, 'rb')
name = cPickle.load(f)
test_cost = cPickle.load(f)
test_metric = cPickle.load(f)
test_metric_std = cPickle.load(f)
test_roc = cPickle.load(f)
test_pr = cPickle.load(f)
f.close()

savepath = os.path.join(datapath, name, "train_performance.pickle")
f = open(savepath, 'rb')
name = cPickle.load(f)
train_cost = cPickle.load(f)
train_metric = cPickle.load(f)
train_metric_std = cPickle.load(f)
train_roc = cPickle.load(f)
train_pr = cPickle.load(f)
f.close()

savepath = os.path.join(datapath, name, "valid_performance.pickle")
f = open(savepath, 'rb')
name = cPickle.load(f)
valid_cost = cPickle.load(f)
valid_metric = cPickle.load(f)
valid_metric_std = cPickle.load(f)
valid_roc = cPickle.load(f)
valid_pr = cPickle.load(f)
f.close()


#----------------------------------------------------------------------------------------
# plot training loss, validation loss, and test loss

plt.figure()
plt.plot(train_cost, linewidth=3, label="train")
plt.plot(valid_cost, linewidth=3, label="valid")
plt.plot(test_cost, linewidth=3, label="valid")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-entropy Loss")
plt.show()
plt.legend(fontsize=12, loc="upper right")


#----------------------------------------------------------------------------------------------------
roc curve
#----------------------------------------------------------------------------------------------------

# plt.figure(figsize=(6,6))
fpr = test_roc[0]
tpr = test_roc[1]

plt.figure()
for i in range(num_labels):
	plt.plot(fpr[i], tpr[i], linewidths=1)
plt.plot([0, 1],[0, 1],'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
ax.grid(True, linestyle=':')
ax = plt.gca()
ax.xaxis.label.set_fontsize(17)
ax.yaxis.label.set_fontsize(17)
map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
plt.tight_layout()
out_pdf = 'my.pdf'
plt.savefig(out_pdf)
plt.close()


#-------------------------------------------------------------------------------------------

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




net1 = lasagne.layers.get_all_layers(network)
    feat_layer = lasagne.layers.get_output(net1[len(net1)-3])
    input_variable = net1[0].input_var
    f_output = theano.function([input_variable],feat_layer)
    num_test = X_test.shape[0]
    X = X_test[0][None,:,:]
    instance = f_output(X)
    for i in range(num_test):
        instance = np.vstack((instance,f_output(X_test[i][None,:,:]).flatten()))


#----------------------------------------------------------------------------------------------------
weblogo
#----------------------------------------------------------------------------------------------------

 weblogo_cmd = 'weblogo weblogo_opts < out_prefix.fa >out_prefix.eps'
 weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % (weblogo_opts, out_prefix, out_prefix)
    subprocess.call(weblogo_cmd, shell=True)

# colors
    color_str = '-c classic'
    if color_mode == 'classic':
        pass
    elif color_mode == 'meme':
        color_str = '--color red A "A"'
        color_str += ' --color blue C "C"'
        color_str += ' --color orange G "G"'
        color_str += ' --color green T "T"'
    else:
        print >> sys.stderr, 'Unrecognized color_mode %s' % color_mode

    # print figure to a temp eps file
    eps_fd, eps_file = tempfile.mkstemp()
    weblogo_cmd = 'weblogo --errorbars NO --show-xaxis NO --show-yaxis NO --fineprint "" %s -n %d %s < %s > %s' % (color_str, len(seq), weblogo_args, fasta_file, eps_file)
    subprocess.call(weblogo_cmd, shell=True)











---------------------------------------------------------------------------
training loss vs validation loss vs test loss
---------------------------------------------------------------------------

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
pyplot.plot(train_loss, linewidth=3, label="train")
pyplot.plot(valid_loss, linewidth=3, label="valid")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()


---------------------------------------------------------------------------
sequence logos
---------------------------------------------------------------------------
	extract filters
	plot sequence logos
	plot sequence logos of original data

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


---------------------------------------------------------------------------
performance
---------------------------------------------------------------------------
	plot roc curves
	plot pr curves

---------------------------------------------------------------------------
feature maps for each layer
---------------------------------------------------------------------------
	plot feature maps for each layer for a given input
layers = net.get_all_layers()
layercounter = 0
for l in layers:

if('Conv2DLayer' in str(type(l))):
    f = open('layer' + str(layercounter) + '.weights','wb')
    weights = l.W.get_value()
    weights = weights.reshape(weights.shape[0]*weights.shape[1],weights.shape[2],weights.shape[3])
    #weights[0]
    for i in range(weights.shape[0]):
        wmin = float(weights[i].min())
        wmax = float(weights[i].max())
        weights[i] *= (255.0/float(wmax-wmin))
        weights[i] += abs(wmin)*(255.0/float(wmax-wmin))
    np.save(f, weights)
    f.close()
    layercounter += 1
with open('layer0.weights', 'rb') as f:
layer0 = np.load(f)

fig, ax = plt.subplots(nrows=3, ncols=32, sharex=True, sharey=False)
#sorg = fig.add_subplot(3,32,1)
for i in xrange(1,97):
#s = fig.add_subplot(3,32,i)
#s.set_adjustable('box-forced')
#s.autoscale(False)
ax[(i-1)/32][(i-1)%32].imshow(layer0[i-1])#,cmap = cm.BLUE,interpolation='bilinear')
ax[(i-1)/32][(i-1)%32].autoscale(False)
ax[(i-1)/32][(i-1)%32].set_ylim([0,12])
plt.show()



# assuming W shape is (num_filters, 3, height, width)
for f in W:
    rgb = f.transpose( (1,2,0) ) # to (height, width, channel) order
    # ... normalize to (0,1) if using float or (0, 255) if using uint8 here
    imshow( rgb )


Nolearn:

clf_layers = clf.get_all_layers()
layer = clf_layers[1]
W = layer.W.get_value()
b = layer.b.get_value()
f = [w + bb for w, bb in zip(W, b)]

gs = gridspec.GridSpec(6, 6)
for i in range(layer.num_filters):
    g = gs[i]
    ax = plt.subplot(g)
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(f[i][0])

output = layer.get_output_for(X_test[:1]).eval()[0]
gs = gridspec.GridSpec(6, 6)
for i in range(layer.num_filters):
    g = gs[i]
    ax = plt.subplot(g)
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(output[i])


layer = clf_layers[4]
W = layer.W.get_value()
b = layer.b.get_value()
f = [w + bb for w, bb in zip(W, b)]

gs = gridspec.GridSpec(6, 6)
for i in range(layer.num_filters):
    g = gs[i]
    ax = plt.subplot(g)
    ax.grid()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(f[i][0])




convolutional auto-encoder for unsupervised pretraining
DanQ version --> LSTM --> bidirectional LSTM
DeepBind --> RNA-compete model


______________________________________________________________________
nolearn visualize

from itertools import product

from lasagne.layers import get_output
from lasagne.layers import get_output_shape
from lasagne.objectives import binary_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    return plt


def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    for feature_map in range(shape[1]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='nearest')
    return plt


def plot_conv_activity(layer, x, figsize=(6, 8)):
    """Plot the acitivities of a specific layer.
    Only really makes sense with layers that work 2D data (2D
    convolutional layers, 2D pooling layers ...).
    Parameters
    ----------
    layer : lasagne.layers.Layer
    x : numpy.ndarray
      Only takes one sample at a time, i.e. x.shape[0] == 1.
    """
    if x.shape[0] != 1:
        raise ValueError("Only one sample can be plotted at a time.")

    # compile theano function
    xs = T.tensor4('xs').astype(theano.config.floatX)
    get_activity = theano.function([xs], get_output(layer, xs))

    activity = get_activity(x)
    shape = activity.shape
    nrows = np.ceil(np.sqrt(shape[1])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows + 1, ncols, figsize=figsize)
    axes[0, ncols // 2].imshow(1 - x[0][0], cmap='gray',
                               interpolation='nearest')
    axes[0, ncols // 2].set_title('original')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[1]:
            break
        ndim = activity[0][i].ndim
        if ndim != 2:
            raise ValueError("Wrong number of dimensions, image data should "
                             "have 2, instead got {}".format(ndim))
        axes[r + 1, c].imshow(-activity[0][i], cmap='gray',
                              interpolation='nearest')
    return plt


def occlusion_heatmap(net, x, target, square_length=7):
    """An occlusion test that checks an image for its critical parts.
    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.
    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.
    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.
    target : int
      The true value of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).
    """
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    num_classes = get_output_shape(net.layers_[-1])[1]
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    # generate occluded images
    for i in range(s0):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]
        y_proba = net.predict_proba(x_occluded)
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array


def _plot_heat_map(net, X, figsize, get_heat_image):
    if (X.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(X.shape))

    num_images = X.shape[0]
    if figsize[1] is None:
        figsize = (figsize[0], num_images * figsize[0] / 3)
    figs, axes = plt.subplots(num_images, 3, figsize=figsize)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for n in range(num_images):
        heat_img = get_heat_image(net, X[n:n + 1, :, :, :], n)

        ax = axes if num_images == 1 else axes[n]
        img = X[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds',
                     alpha=0.6)
        ax[2].set_title('super-imposed')
    return plt


def plot_occlusion(net, X, target, square_length=7, figsize=(9, None)):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    X : numpy.array
      The input data, should be of shape (b, c, 0, 1). Only makes
      sense with image data.
    target : list or numpy.array of ints
      The true values of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at. If more than one sample is passed to X, each of them needs
      its own target.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    figsize : tuple (int, int)
      Size of the figure.
    Plots
    -----
    Figure with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.
    """
    return _plot_heat_map(net, X, figsize, lambda net, X, n: occlusion_heatmap(net, X, target[n], square_length))


def saliency_map(input, output, pred, X):
    score = -binary_crossentropy(output[:, pred], np.array([1])).sum()
    return np.abs(T.grad(score, input).eval({input: X}))


def saliency_map_net(net, X):
    input = net.layers_[0].input_var
    output = get_output(net.layers_[-1])
    pred = output.eval({input: X}).argmax(axis=1)
    return saliency_map(input, output, pred, X)[0].transpose(1, 2, 0).squeeze()


def plot_saliency(net, X, figsize=(9, None)):
    return _plot_heat_map(net, X, figsize, lambda net, X, n: -saliency_map_net(net, X))

































