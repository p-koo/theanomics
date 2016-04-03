#!/bin/python
from six.moves import cPickle
import numpy as np
import sys
 
# convert to 1 hot representation sequence
def convert_one_hot_rna(seq):
    """convert a sequence into a 1-hot representation"""
    nucleotide = 'ACGU'
    N = len(seq)
    one_hot_seq = np.zeros((4,N))
    for i in xrange(200):         
        #for j in range(4):
        #    if seq[i] == nucleotide[j]:
        #        one_hot_seq[j,i] = 1
        index = [j for j in range(4) if seq[i] == nucleotide[j]]
        one_hot_seq[index,i] = 1
    return one_hot_seq

def main():

    # load data
    filename = sys.argv[1]
    # filename = 'data_10000_200_10_20.pickle'
    f = open(filename, 'rb')
    data = cPickle.load(f)
    label = cPickle.load(f)
    f.close()

    # percentage for each dataset
    train_size = 0.7
    cross_validation_size = 0.15
    test_size = 0.15

    # get indices for each dataset
    N = len(data)
    cum_index = np.cumsum([0, train_size*N, cross_validation_size*N, test_size*N]).astype(int) 
    
    # shuffle data
    shuffle = np.random.permutation(N)

    # training dataset
    indices = np.arange(cum_index[0],cum_index[1])
    train_set = [] 
    train_set_label = []
    for i in range(len(indices)):
        index = shuffle[indices[i]]
        train_set.append(convert_one_hot_rna(data[index]))
        train_set_label.append(label[index])

    # cross-validation set
    indices = np.arange(cum_index[1],cum_index[2])
    cross_validation_set = [] 
    cross_validation_set_label = []
    for i in range(len(indices)):
        index = shuffle[indices[i]]
        cross_validation_set.append(convert_one_hot_rna(data[index]))
        cross_validation_set_label.append(label[index])

    # test set
    indices = np.arange(cum_index[2],cum_index[3])
    test_set = [] 
    test_set_label = []
    for i in range(len(indices)):
        index = shuffle[indices[i]]
        test_set.append(convert_one_hot_rna(data[index]))
        test_set_label.append(label[index])

    # save training dataset in one-hot representation
    trainname = 'train_' + filename 
    f = open(trainname, 'wb')
    cPickle.dump(train_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(train_set_label, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # save cross-validation dataset in one-hot representation
    crossname = 'cross_validation_' + filename 
    f = open(crossname, 'wb')
    cPickle.dump(cross_validation_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(cross_validation_set_label, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    # save training dataset in one-hot representation
    testname = 'test_' + filename 
    f = open(testname, 'wb')
    cPickle.dump(test_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(test_set_label, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == "__main__":
   main()


