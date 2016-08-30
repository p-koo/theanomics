#!/bin/python
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.misc import imresize
import pandas as pd
from lasagne.layers import get_output, get_output_shape, get_all_params
import theano.tensor as T
import theano
"""
class NeuralVisualize():
    def __init__(nnmodel, options=[]):
        self.network = nnmodel.network
        self.num_labels = nnmodel.num_labels
        self.input_var = nnmodel.input_var
        self.options = options
"""

def plot_filter_logos(W, figsize=(2,10), height=25, nt_width=10, norm=0, rna=0):
    W =  np.squeeze(W)
    num_filters = W.shape[0]
    num_rows = int(np.ceil(np.sqrt(num_filters)))    
    grid = mpl.gridspec.GridSpec(num_rows, num_rows)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 
    fig = plt.figure(figsize=figsize);
    for i in range(num_filters):
        logo = seq_logo(W[i], height=height, nt_width=nt_width, norm=norm, rna=rna)
        plt.subplot(grid[i]);
        plt.imshow(logo);
        plt.axis('off');
    return fig, plt


def plot_mean_activations(fmaps, y, options):
    """plot mean activations for a given layer"""
    def get_mean_activation(fmaps, y, batch_size=512):
        fmaps = np.squeeze(fmaps)
        mean_activation = []
        std_activation = []
        for i in range(max(y)+1):
            index = np.where(y == i)[0]
            mean_activation.append(np.mean(fmaps[index], axis=0))
            std_activation.append(np.std(fmaps[index], axis=0))
        return np.array(mean_activation), np.array(std_activation)

    mean_activation, std_activation = get_mean_activation(fmaps, y)
    num_labels = len(mean_activation)
    nrows = np.ceil(np.sqrt(num_labels)).astype(int)
    ncols = nrows

    fig = plt.figure()
    grid = subplot_grid(nrows, ncols)
    for i in range(num_labels):
        plt.subplot(grid[i])
        plt.plot(mean_activation[i].T)
        fig_options(plt, options)
    return fig, plt


def tSNE_plot(data, labels, figsize):
    """scatter plot of tSNE 2D projections, with a color corresponding to labels"""
    num_labels = max(labels)+1
    x = data[:,0]
    y = data[:,1]
    plt.figure(figsize = figsize)
    plt.scatter(x, y, c=labels, cmap=plt.cm.get_cmap("jet", num_labels),  edgecolor = 'none')
    plt.axis('off')
    return plt


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


#------------------------------------------------------------------------------------------------
# helper functions

def fig_options(plt, options):
    if 'figsize' in options:
        fig = plt.gcf()
        fig.set_size_inches(options['figsize'][0], options['figsize'][1], forward=True)
    if 'ylim' in options:
        plt.ylim(options['ylim'][0],options['ylim'][1])
    if 'yticks' in options:
        plt.yticks(options['yticks'])
    if 'xticks' in options:
        plt.xticks(options['xticks'])
    if 'labelsize' in options:        
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=options['labelsize'])
        ax.tick_params(axis='y', labelsize=options['labelsize'])
    if 'axis' in options:
        plt.axis(options['axis'])
    if 'xlabel' in options:
        plt.xlabel(options['xlabel'], fontsize=options['fontsize'])
    if 'ylabel' in options:
        plt.ylabel(options['ylabel'], fontsize=options['fontsize'])
    if 'linewidth' in options:
        plt.rc('axes', linewidth=options['linewidth'])

def subplot_grid(nrows, ncols):
    grid= mpl.gridspec.GridSpec(nrows, ncols)
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2) 
    return grid


def seq_logo(pwm, height=30, nt_width=10, norm=0, rna=1, filepath='nt'):
    """generate a sequence logo from a pwm"""
    
    def load_alphabet(filepath, rna):
        """load images of nucleotide alphabet """
        df = pd.read_table(os.path.join(filepath, 'A.txt'), header=None);
        A_img = df.as_matrix()
        A_img = np.reshape(A_img, [72, 65, 3], order="F").astype(np.uint8)

        df = pd.read_table(os.path.join(filepath, 'C.txt'), header=None);
        C_img = df.as_matrix()
        C_img = np.reshape(C_img, [76, 64, 3], order="F").astype(np.uint8)

        df = pd.read_table(os.path.join(filepath, 'G.txt'), header=None);
        G_img = df.as_matrix()
        G_img = np.reshape(G_img, [76, 67, 3], order="F").astype(np.uint8)

        if rna == 1:
            df = pd.read_table(os.path.join(filepath, 'U.txt'), header=None);
            T_img = df.as_matrix()
            T_img = np.reshape(T_img, [74, 57, 3], order="F").astype(np.uint8)
        else:
            df = pd.read_table(os.path.join(filepath, 'T.txt'), header=None);
            T_img = df.as_matrix()
            T_img = np.reshape(T_img, [72, 59, 3], order="F").astype(np.uint8)

        return A_img, C_img, G_img, T_img


    def get_nt_height(pwm, height, norm):
        """get the heights of each nucleotide"""

        def entropy(p):
            """calculate entropy of each nucleotide"""
            s = 0
            for i in range(4):
                if p[i] > 0:
                    s -= p[i]*np.log2(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (np.log2(4) - entropy(pwm[:, i]))*height;
            heights[:,i] = np.floor(pwm[:,i]*total_height);
        return heights.astype(int)

    
    # get the alphabet images of each nucleotide
    A_img, C_img, G_img, T_img = load_alphabet(filepath, rna)
    
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)
    
    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    total_height = np.sum(heights,axis=0)
    max_height = np.max(total_height)
    logo = np.ones((height*2, width, 3)).astype(int)*255;
    for i in range(num_seq):
        remaining_height = total_height[i];
        offset = max_height-remaining_height
        nt_height = np.sort(heights[:,i]);
        index = np.argsort(heights[:,i])

        for j in range(num_nt):
            if nt_height[j] > 0:
                # resized dimensions of image
                resize = (nt_height[j], nt_width)
                if index[j] == 0:
                    nt_img = imresize(A_img, resize)
                elif index[j] == 1:
                    nt_img = imresize(C_img, resize)
                elif index[j] == 2:
                    nt_img = imresize(G_img, resize)
                elif index[j] == 3:
                    nt_img = imresize(T_img, resize)

                # determine location of image
                height_range = range(remaining_height-nt_height[j], remaining_height)
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                if height_range:
                    for k in range(3):
                        for m in range(len(width_range)):
                            logo[height_range+offset, width_range[m],k] = nt_img[:,m,k];

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


