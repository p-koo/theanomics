#!/bin/python
from six.moves import cPickle
import numpy as np
import os

def generate_sequence_pwm(num_seq, seq_length, motif):
    """generate a regulatory grammar model: various numbers of motifs with 
    distinct separations."""
    
    # generate G regulatory grammars (combinations of motifs + distance between motifs)
    start_index = np.random.randint(1,seq_length-20,num_seq)
    sequence_pwm = []
    for index in start_index:
        sequence_pwm = np.ones((4, index))/4
        sequence_pwm = np.hstack((sequence_pwm, motif))
        sequence_pwm = np.hstack((sequence_pwm, np.ones((4, seq_length-sequence_pwm.shape[1]))/4))
        sequence_pwm.append(sequence_pwm) 
        
    return sequence_pwm


def simulate_sequence(sequence_pwm):
    """simulate a sequence given a sequence model"""
    
    nucleotide = 'ACGU'

    # sequence length
    seq_length = sequence_pwm.shape[1]

    # generate uniform random number for each nucleotide in sequence
    Z = np.random.uniform(0,1,seq_length)
    
    # calculate cumulative sum of the probabilities
    cum_prob = sequence_pwm.cumsum(axis=0)

    # go through sequence and find bin where random number falls in cumulative 
    # probabilities for each nucleotide
    sequence = ''
    for i in xrange(seq_length):
        index=[j for j in xrange(4) if Z[i] < cum_prob[j,i]][0]
        sequence += nucleotide[index]

    return sequence


def simulate_data(motif_set, num_seq, uniform):
    """simulates N sequences with random population fractions for each sequence 
    model (PWM) of each regulatory grammar """

    # simulate random population fractions and scale to N sequences
    if uniform == 1:
        w = 1./len(motif_set)
    elif:
        w = np.random.uniform(0, 1, size=len(motif_set))
        w = np.round(w/sum(w)*num_seq)
        popFrac = w.astype(int)

    # create a popFrac weighted number of simulation for each regulatory grammar
    label = []
    data = []
    for i in xrange(len(popFrac)):
        sequence_pwm = generate_sequence_pwm(popFrac[i], seq_length, motif_set[i])

        for j in xrange():

            sequence = simulate_sequence(seq_model[i])
            data.append(sequence)
            label.append(i)
            
    return data, label


def convert_one_hot(seq):
    """convert a sequence into a 1-hot representation"""
    
    nucleotide = 'ACGU'
    N = len(seq)
    one_hot_seq = np.zeros((4,N))
    for i in xrange(200):         
        #for j in range(4):
        #    if seq[i] == nucleotide[j]:
        #        one_hot_seq[j,i] = 1
        index = [j for j in xrange(4) if seq[i] == nucleotide[j]]
        one_hot_seq[index,i] = 1
        
    return one_hot_seq


def subset_data(data, label, sub_index):
    """returns a subset of the data and labels based on sub_index"""
    
    num_labels = len(np.unique(label))
    num_sub = len(sub_index)
    
    sub_set = np.zeros((num_sub, 4, len(data[0])))
    sub_set_label = np.zeros((num_sub, num_labels))
    
    k = 0;
    for index in sub_index:
        sub_set[k] = convert_one_hot(data[index])
        sub_set_label[k,label[index]] = 1
        k += 1

    sub_set_label = sub_set_label.astype(np.uint8)
    
    return (sub_set, sub_set_label)


def split_data(data, label, split_size):
    """split data into train set, cross-validation set, and test set"""
    
    # number of labels
    num_labels = len(np.unique(label))

    # determine indices of each dataset
    N = len(data)
    cum_index = np.cumsum(np.multiply([0, split_size[0], split_size[1], split_size[2]],N)).astype(int) 

    # shuffle data
    shuffle = np.random.permutation(N)

    # training dataset
    train_index = shuffle[range(cum_index[0], cum_index[1])]
    cross_validation_index = shuffle[range(cum_index[1], cum_index[2])]
    test_index = shuffle[range(cum_index[2], cum_index[3])]

    # create subsets of data based on indices 
    train = subset_data(data, label, train_index)
    cross_validation = subset_data(data, label, cross_validation_index)
    test = subset_data(data, label, test_index)
    
    return train, cross_validation, test


def main():

    # dataset parameters
    outdir = 'data'     
    num_seq = 500000       # number of sequences
    seq_length = 100       # length of sequence
    num_motif = 100        # number of motifs
    filename =  'simple_' + str(num_motif) + '_' + str(num_seq) # output filename
                
    # percentage for each dataset
    train_size = 0.7
    cross_validation_size = 0.15
    test_size = 0.15

    # load motif list from file
    motiflist = 'motif.pickle'
    f = open(motiflist, 'rb')
    motif_set = cPickle.load(f)
    f.close()

    # select num_motif random motifs from the complete list of motif_set
    motifIndex = np.random.permutation(len(motif_set))[0:num_motif]

    # convert this to a sequence position weight matrix for each model
    seq_model = generate_sequence_model(num_seq, seq_length, motif_set)


    # simulate N sequences based on the position weight matrices
    print "Generating synthetic data"
    data, label = simulate_data(seq_model, num_seq)

    # get indices for each dataset
    print "Splitting dataset into train, cross-validation, and test"
    split_size = [train_size, cross_validation_size, test_size]
    train, cross_validation, test = split_data(data, label, split_size)

    # save training dataset in one-hot representation
    print "Saving dataset"
    f = open(os.path.join(outdir, filename+'_data.pickle'), 'wb')
    cPickle.dump(train, f)
    cPickle.dump(cross_validation, f)
    cPickle.dump(test, f)
    f.close()

    # save training dataset in one-hot representation
    print "Saving model"
    f = open(os.path.join(outdir, filename+'_model.pickle'), 'wb')
    cPickle.dump(options, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(seq_model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == "__main__":
   main()
