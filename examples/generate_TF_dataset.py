#/bin/python
import os, sys, h5py
import numpy as np
from six.moves import cPickle
sys.path.append('..')
from models import load_model
from data import load_data
np.random.seed(247) # for reproducibility


# H1-hESC   CTCF  GABP SP1  SRF  TAF1 YY1                   
range1 = [262, 265, 278, 281, 282, 287]        
#
# HepG2     CTCF  GABP SP1  SRF  TAF1 YY1                     
range2 = [299, 305, 321, 323, 324, 328]                 
#
# K562      CTCF  GABP SP1  SRF  TAF1 YY1                     
range3 = [338, 345, 359, 361, 363, 369]                  
#
# all
range4 = [262, 265, 278, 281, 282, 287, 299, 305, 321, 333, 324, 328, 338, 345, 359, 361, 363, 369] 
#
# just CTCF
# K562 H1-hESC HepG2                                       
range5 = [415, 299, 338]                                              

range6 = range(129,330) 

name = ['H1-hESC','HepG2', 'K562', 'combine', 'CTCF', 'all']
class_range = [range1, range2, range3, range4, range5, range6]


#------------------------------------------------------------------------------
# load data

datapath = '/media/peter/storage/'
output_file = os.path.join(datapath,'DeepSea', 'TF_dataset.hdf5')
with h5py.File(output_file,'w') as f:
    for i in range(len(name)):
        options = {"class_range": class_range[i]}
        train, valid, test = load_data('DeepSea', os.path.join(datapath,'DeepSea'), options)
        grp = f.create_group(name[i])
        X_train = grp.create_dataset('X_train', data=train[0], dtype='int8')
        Y_train = grp.create_dataset('Y_train', data=train[1], dtype='int8')
        X_valid = grp.create_dataset('X_valid', data=valid[0], dtype='int8')
        Y_valid = grp.create_dataset('Y_valid', data=valid[1], dtype='int8')
        X_test = grp.create_dataset('X_test', data=test[0], dtype='int8')
        Y_test = grp.create_dataset('Y_test', data=test[1], dtype='int8')

