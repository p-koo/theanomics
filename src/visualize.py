#!/bin/python
import matplotlib.image as img
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from scipy.misc import imresize

from lasagne.layers import get_output, get_output_shape, get_all_params
import theano.tensor as T
import theano

class Visualize():
    def __init__(nnmodel, options=[]):
        self.network = nnmodel.network
        self.num_labels = nnmodel.num_labels
        self.input_var = nnmodel.input_var
        self.options = options

    def get_feature_maps(layer, X, batch_size=500):
        """get the feature maps of a given convolutional layer"""
        
        num_data = len(X)
        feature_maps = theano.function([self.input_var], get_output(layer), allow_input_downcast=True)
        map_shape = get_output_shape(layer)

        # get feature maps in batches for speed (large batches may be too much memory for GPU)
        num_batches = num_data // batch_size
        shape = list(map_shape)
        shape[0] = num_data
        fmaps = np.empty(tuple(shape))
        for i in range(num_batches):
            index = range(i*batch_size, (i+1)*batch_size)    
            fmaps[index] = feature_maps(X[index])

        # get the rest of the feature maps
        excess = num_data-num_batches*batch_size
        if excess:
            index = range(num_data-excess, num_data)  
            fmaps[index] = feature_maps(X[index])

        return fmaps

    def get_weights(layer, convert_pwm=0):
        W =  np.squeeze(layer.W.get_value())
        if convert_pwm == 1:
            for i in range(len(W)):
                #weights = np.exp(W[i])
                MIN = np.min(W[i])
                weights = W[i] - MIN
                Z = np.sum(weights, axis=0)
                weights /= np.tile(Z, (W[i].shape[0],1))
                W_norm[i] = weights
        else:
            W_norm = W
        return W_norm




def plot_conv_filter(plt, pwm, height=200, bp_width=100, norm=0, rna=1, adjust=-1, filepath='.', showbar=0):
    num_seq = pwm.shape[1]
    width = bp_width*num_seq

    logo = seq_logo(pwm, height, width, norm, rna, filepath)
    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].imshow(logo, extent=[bp_width*2, width+bp_width, 0, height])
    axes[0].set_axis_off()
    im = axes[1].imshow(pwm, cmap='jet', vmin=0, vmax=1, interpolation='none') 
    axes[1].set_axis_off()
    fig.subplots_adjust(bottom=adjust)
    if showbar == 1:
        cbar_ax = fig.add_axes([.85, 0.05, 0.05, 0.45])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.5, 1])
        cb.ax.tick_params(labelsize=16)
    return fig


def plot_conv_weights(W, options):
    num_filters = W.shape[0]
    nrows = np.ceil(np.sqrt(num_filters)).astype(int)
    ncols = nrows
    plt.figure()
    grid = subplot_grid(nrows, ncols)
    for i in range(num_filters):
        plt.subplot(grid[i])
        plt.imshow(W[i], cmap='hot_r', interpolation='nearest')
        fig_options(plt, options)
    return plt


def seq_logo(pwm, height=100, width=200, norm=0, rna=1, filepath='.'):
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
                    s -= p[i]*np.log(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq));
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (2 - entropy(pwm[:, i]))*height/2;
            heights[:,i] = np.floor(pwm[:,i]*total_height);

        return heights.astype(int)

    
    # get the alphabet images of each nucleotide
    A_img, C_img, G_img, T_img = load_alphabet(filepath='.', rna=1)
    
    
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    nt_width = np.floor(width/num_seq).astype(int)
    logo = np.ones((height, width, 3)).astype(int)*255;
    for i in range(num_seq):
        remaining_height = height;
        nt_height = np.sort(heights[:,i]);
        index = np.argsort(heights[:,i])

        
        for j in range(num_nt):
            # resized dimensions of image
            if nt_height[j] > 0:
                resize = (nt_height[j],nt_width)
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
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range, width_range[m],k] = nt_img[:,m,k];

                remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


def seq_logo2(pwm, height=100, width=200, norm=0, rna=1, filepath='.'):
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
                    s -= p[i]*np.log(p[i])
            return s

        num_nt, num_seq = pwm.shape
        heights = np.zeros((num_nt,num_seq))
        for i in range(num_seq):
            if norm == 1:
                total_height = height
            else:
                total_height = (2 - entropy(np.abs(pwm[:, i])))*height
            heights[:,i] = np.floor(np.abs(pwm[:,i])*total_height);
            

        return heights.astype(int)

    
    # get the alphabet images of each nucleotide
    A_img, C_img, G_img, T_img = load_alphabet(filepath='.', rna=1)
    
    
    # get the heights of each nucleotide
    heights = get_nt_height(pwm, height, norm)

    # resize nucleotide images for each base of sequence and stack
    num_nt, num_seq = pwm.shape
    nt_width = np.floor(width/num_seq).astype(int)
    logo = np.ones((2*height, width, 3)).astype(int)*255;
    sign = np.sign(pwm)
    for i in range(num_seq):
        pos_index = np.where(sign[:,i]> 0)[0]
        nt_height = np.sort(heights[pos_index,i]);
        index = pos_index[np.argsort(heights[pos_index,i])]
        remaining_height = height;
        
        for j in range(len(pos_index)):
            # resized dimensions of image
        
            if nt_height[j] > 0:
                resize = (nt_height[j],nt_width)
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
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range, width_range[m],k] = nt_img[:,m,k];
                remaining_height -= nt_height[j]
        
        pos_index = np.where(sign[:,i]< 0)[0]
        nt_height = np.sort(heights[pos_index,i]);
        nt_height = nt_height[::-1]
        index = pos_index[np.argsort(heights[pos_index,i])]
        index = index[::-1]
        remaining_height = np.sum(nt_height)-height;
        
        for j in range(len(pos_index)):
            # resized dimensions of image
            if nt_height[j] > 0:
                resize = (nt_height[j],nt_width)
                if index[j] == 0:
                    nt_img = imresize(A_img[::-1,:,:], resize)
                elif index[j] == 1:
                    nt_img = imresize(C_img[::-1,:,:], resize)
                elif index[j] == 2:
                    nt_img = imresize(G_img[::-1,:,:], resize)
                elif index[j] == 3:
                    nt_img = imresize(T_img[::-1,:,:], resize)

                # determine location of image
                height_range = range(remaining_height-nt_height[j],remaining_height)
                width_range = range(i*nt_width, i*nt_width+nt_width)

                # 'annoying' way to broadcast resized nucleotide image
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range, width_range[m],k] = nt_img[:,m,k];
                remaining_height -= nt_height[j]
        
    return logo.astype(np.uint8)




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
