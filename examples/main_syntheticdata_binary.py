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
np.random.seed(247) # for reproducibility

#------------------------------------------------------------------------------
# load data

name = 'MotifSimulation_correlated'
datapath = '/home/peter/Data/SequenceMotif'
filepath = os.path.join(datapath, 'synthetic_correlated_motifs_300000.hdf5')
train, valid, test = load_data(name, filepath)
shape = (None, train[0].shape[1], train[0].shape[2], train[0].shape[3])
num_labels = np.round(train[1].shape[1])

"""
random
  test loss:		0.19582
  test accuracy:	0.93177+/-0.02346
  test auc-roc:	0.90736+/-0.03641
  test auc-pr:		0.72602+/-0.09486

  test loss:		0.18212
  test accuracy:	0.93527+/-0.02120
  test auc-roc:	0.92287+/-0.03157
  test auc-pr:		0.75277+/-0.08848


correlated
  test loss:		0.19108
  test accuracy:	0.93210+/-0.02161
  test auc-roc:	0.91711+/-0.03147
  test auc-pr:		0.73934+/-0.08664

  test loss:		0.18065
  test accuracy:	0.93428+/-0.02234
  test auc-roc:	0.92796+/-0.02906
  test auc-pr:		0.76041+/-0.08327

random 300000
  test loss:		0.16663
  test accuracy:	0.94084+/-0.01945
  test auc-roc:	0.93659+/-0.02787
  test auc-pr:		0.79017+/-0.08711

correlated
  test loss:		0.15933
  test accuracy:	0.94169+/-0.02506
  test auc-roc:	0.94500+/-0.02649
  test auc-pr:		0.80914+/-0.07878

  test loss:		0.15302
  test accuracy:	0.94342+/-0.02494
  test auc-roc:	0.95012+/-0.02519
  test auc-pr:		0.82071+/-0.07227

"""

#-------------------------------------------------------------------------------------

# build model
model_name = "binary_genome_motif_model"
nnmodel = NeuralNet(model_name, shape, num_labels)
nnmodel.print_layers()

# set output file paths
outputname = 'binary'
datapath = make_directory(datapath, 'Results')
filepath = os.path.join(datapath, outputname)

# train model
batch_size = 100
#nnmodel = fit.anneal_train_valid_minibatch(nnmodel, train, valid, batch_size, num_epochs=500, patience=5, verbose=1, filepath=filepath)
nnmodel = fit.train_minibatch(nnmodel, train, valid, batch_size=batch_size, num_epochs=500, 
			patience=10, verbose=1, filepath=filepath)

# save best model --> lowest cross-validation error
min_loss, min_index = nnmodel.get_min_loss()
savepath = filepath + "_epoch_" + str(min_index) + ".pickle"
nnmodel.set_parameters_from_file(savepath)
savepath = filepath + "_best.pickle"
nnmodel.save_model_parameters(savepath)

# test model
nnmodel.test_model(test, batch_size, "test")

# save all performance metrics (train, valid, test)
nnmodel.save_all_metrics(filepath)

# monitor/save test performance with parameters for each training epoch
num_train_epochs = nnmodel.get_num_epochs()
performance = fit.test_model_all(nnmodel, test, batch_size, num_train_epochs, filepath)

# save test performance
performance.save_metrics(filepath)















