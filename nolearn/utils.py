#!/bin/python
import sys, time
import numpy as np
from lasagne import updates
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo, TrainSplit, 


def get_all_params_values(self):
        return_value = OrderedDict()
        for name, layer in self.layers_.items():
            return_value[name] = [p.get_value() for p in layer.get_params()]
        return return_value


        
class AdjustVariable(object):
    """ Adjust the variable linearly during training,
        such as the learning rate or momentum. """

    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start = start
        self.stop = stop
        self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    """ check validation loss for early stopping. """
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping...")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class MonitorPerformance():
    """ Monitor performance during training with accuracy, 
        AUC-ROC, and AUC-PR.  Also allows a progress bar"""
    self.start_time = 0

    def set_start_time(start_time = time.time()):
        self.start_time = time()

    def print_results(name): 
        print("  " + name + " cost:\t\t{:.4f}".format(self.cost[-1]/1.))
        if self.metric.any():
            accuracy, auc_roc, auc_pr = self.get_mean_values()
            accuracy_std, auc_roc_std, auc_pr_std = self.get_error_values()
            print("  " + name + " accuracy:\t{:.4f}+/-{:.4f}".format(accuracy, accuracy_std))
            print("  " + name + " auc-roc:\t{:.4f}+/-{:.4f}".format(auc_roc, auc_roc_std))
            print("  " + name + " auc-pr:\t\t{:.4f}+/-{:.4f}".format(auc_pr, auc_pr_std))

    def progress_bar(self, num_batches, bar_length=30):
        train_loss = train_history[-1]['train_loss']    
        valid_loss = train_history[-1]['valid_loss']
        index = train_history[-1]['epoch']

        remaining_time = (time.time()-self.start_time)*(num_batches-index-1)/(index+1)
        percent = (index+1.)/num_batches
        progress = '='*int(round(percent*bar_length))
        spaces = ' '*int(bar_length-round(percent*bar_length))
        sys.stdout.write("\r[%s] %.1f%% -- time=%ds -- train loss=%.5f -- valid loss=%.5f    " \
        %(progress+spaces, percent*100, remaining_time, train_loss, valid_loss))
        sys.stdout.flush()


class 

    def set_best_model(self, filepath):
        min_cost, min_index = self.valid_monitor.get_min_cost()    
        savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
        f = open(savepath, 'rb')
        best_parameters = cPickle.load(f)
        f.close()
        self.set_model_parameters(best_parameters)



    def test_step_batch(self, test):
        test_cost, test_prediction = self.test_fun(test[0].astype(np.float32), test[1].astype(np.int32)) 
        return test_cost, test_prediction


    def test_step_minibatch(self, mini_batches, num_batches):
        performance = MonitorPerformance()
        for index in range(num_batches):
            X, y = next(mini_batches)
            cost, prediction = self.test_fun(X, y)
            performance.add_cost(cost)
        return performance.get_mean_cost()


#--------------------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------------------

def updates_grad_clip(grad, all_params, **update_params):
    """ modified optimization algorithm with gradient clipping"""

    # gradient clipping option
    if 'weight_norm' in update_params:
        grad = updates.total_norm_constraint(grad, update_params['weight_norm'])
        del update_params['weight_norm']

    if 'optimizer' in update_params:
        optimizer = update_params['optimizer']
        del update_params['optimizer']

    if optimizer:
        if update_params['optimizer'] == 'sgd':
            update_op = updates.sgd(grad, all_params, **update_params) 
     
        elif update_params['optimizer'] == 'nesterov_momentum':
            update_op = updates.nesterov_momentum(grad, all_params, **update_params)    

        elif update_params['optimizer'] == 'adagrad':
            update_op = updates.adagrad(grad, all_params, **update_params)

        elif update_params['optimizer'] == 'rmsprop':
            update_op = updates.rmsprop(grad, all_params, **update_params)
        
        elif update_params['optimizer'] == 'adam':
            update_op = updates.adam(grad, all_params, **update_params) 
    else:
        update_op = updates.nesterov_momentum(grad, params, **update_params)

    return update_op


def make_directory(path, foldername, verbose=1):
    """make a directory"""
    if not os.path.isdir(path):
        os.mkdir(path)
        print "making directory: " + path

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print "making directory: " + outdir
    return outdir


def batch_generator(X, y, N):
    """python generator to get a randomized minibatch"""
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')


def one_hot_labels(label):
    """convert categorical labels to one hot"""
    num_data = label.shape[0]
    num_labels = max(label)+1
    label_expand = np.zeros((num_data, num_labels))
    for i in range(num_data):
        label_expand[i, label[i]] = 1
    return label_expand


def calculate_metrics(label, prediction):
    """calculate metrics for classification"""

    def accuracy_metrics(label, prediction):
        num_labels = label.shape[1]
        accuracy = np.zeros((num_labels))
        for i in range(num_labels):
            score = accuracy_score(label[:,i], np.round(prediction[:,i]))
            accuracy[i] = score
        return accuracy

    def roc_metrics(label, prediction):
        num_labels = label.shape[1]
        roc = []
        auc_roc = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
            score = auc(fpr, tpr)
            auc_roc[i]= score
            roc.append((fpr, tpr))
        return auc_roc, roc

    def pr_metrics(label, prediction):
        num_labels = label.shape[1]
        pr = []
        auc_pr = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
            score = auc(recall, precision)
            auc_pr[i] = score
            pr.append((precision, recall))
        return auc_pr, pr

    num_samples = len(prediction)
    ndim = np.ndim(label)
    if ndim == 1:
        label = one_hot_labels(label)

    accuracy = accuracy_metrics(label, prediction)
    auc_roc, roc = roc_metrics(label, prediction)
    auc_pr, pr = pr_metrics(label, prediction)
    mean = [np.nanmean(accuracy), np.nanmean(auc_roc), np.nanmean(auc_pr)]
    std = [np.std(accuracy), np.std(auc_roc), np.std(auc_pr)]
    
    return mean, std, roc, pr













