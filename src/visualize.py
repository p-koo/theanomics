#!/bin/python
import matplotlib.image as img
from matplotlib import pyplot as plt
import theano.tensor as T
import theano
from lasagne.layers import get_output


def plot_loss(loss):
    """Plot trainig/validation/test loss during training"""

    fig = plt.figure()
    num_data_types = len(loss)
    if num_data_types == 2:
        plt.plot(loss[0], label='train loss', linewidth=2)
        plt.plot(loss[1], label='valid loss', linewidth=2)
    elif num_data_types == 3:
        plt.plot(loss[0], label='train loss', linewidth=2)
        plt.plot(loss[1], label='valid loss', linewidth=2)
        plt.plot(loss[2], label='test loss', linewidth=2)

    plt.xlabel('epoch', fontsize=22)
    plt.ylabel('loss', fontsize=22)
    plt.legend(loc='best', frameon=False, fontsize=18)
	map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
	map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
	plt.tight_layout()
    return fig, plt


def plot_roc_all(final_roc):
    """Plot ROC curve for each class"""

    fig = plt.figure()
    for i in range(len(final_roc)):
        plt.plot(final_roc[i][0],final_roc[i][1])
    plt.xlabel('False positive rate', fontsize=22)
    plt.ylabel('True positive rate', fontsize=22)
    plt.plot([0, 1],[0, 1],'k--')
    ax = plt.gca()
	ax.xaxis.label.set_fontsize(17)
	ax.yaxis.label.set_fontsize(17)
	map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
	map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
	plt.tight_layout()
    #plt.legend(loc='best', frameon=False, fontsize=14)
    return fig, plt


def plot_pr_all(final_pr):
    """Plot PR curve for each class"""

    fig = plt.figure()
    for i in range(len(final_roc)):
        plt.plot(final_pr[i][0],final_pr[i][1])
    plt.xlabel('Recall', fontsize=22)
    plt.ylabel('Product', fontsize=22)
    plt.plot([0, 1],[0, 1],'k--')
    ax = plt.gca()
	ax.xaxis.label.set_fontsize(17)
	ax.yaxis.label.set_fontsize(17)
	map(lambda xl: xl.set_fontsize(13), ax.get_xticklabels())
	map(lambda yl: yl.set_fontsize(13), ax.get_yticklabels())
	plt.tight_layout()
    #plt.legend(loc='best', frameon=False, fontsize=14)
    return fig, plt


def plot_conv_weights(layer, figsize=(6, 6)):
    """nolearn's plot the weights of a specific layer"""

    W =  np.squeeze(layer.W.get_value())
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, figsize=figsize,frameon=False)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, ax in enumerate(axes.ravel()):
        if i >= shape[0]:
            break
        im = ax.imshow(W[i], cmap='gray', interpolation='nearest')

    return figs, axes


def plot_weights(weights):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(weights.T, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return plt


def plot_conv_activity(activity, figsize=(6, 8)):
    """nolearn's plot the acitivities of a specific layer.
        x : numpy.ndarray (1 data point) """

    fig = plt.figure()
    shape = activity.shape
    nrows = np.ceil(np.sqrt(shape[1])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows + 1, ncols, figsize=figsize)
    axes[0, ncols // 2].imshow(1 - x[0][0], cmap='gray', interpolation='nearest')
    axes[0, ncols // 2].set_title('original')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, ax in enumerate(axes.ravel()):    
        if i >= shape[1]:
            break
        ndim = activity[0][i].ndim
        if ndim != 2:
            raise ValueError("Wrong number of dimensions, image data should "
                             "have 2, instead got {}".format(ndim))
        ax.imshow(-activity[0][i], cmap='gray', interpolation='nearest')

    plt.show()
    return plt









"""

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


"""











