import numpy as np
import h5py
import scipy.io
from six.moves import cPickle
import sys
import os.path

trainmat = h5py.File('train.mat')
validmat = scipy.io.loadmat('valid.mat')
testmat = scipy.io.loadmat('test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T
f = open('train.pickle', 'wb')
cPickle.dump(X_train, f)
cPickle.dump(y_train, f)
f.close()

"""

X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
y_test = testmat['testdata']
f = open('test.pickle', 'wb')
cPickle.dump(X_test, f)
cPickle.dump(y_test, f)
f.close()

X_valid = np.transpose(validmat['validxdata'],axes=(0,2,1))
y_valid = validmat['validdata']
f = open('valid.pickle', 'wb')
cPickle.dump(X_valid, f)
cPickle.dump(y_valid, f)
f.close()

"""
