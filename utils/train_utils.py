#!/bin/python

import numpy as np
import time
import sys


def batch_generator(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')

def progress_bar(start_time, index, num_batches, loss, accuracy, bar_length=20):
    remaining_time = (time.time()-start_time)*(num_batches-index-1)/(index+1)
    percent = (index+1.)/num_batches
    progress = '='*int(round(percent*bar_length))
    spaces = ' '*int(bar_length-round(percent*bar_length))
    sys.stdout.write("\r[%s] %.1f%% -- est.time=%ds -- loss=%.3f -- accuracy=%.2f%%" \
    %(progress+spaces, percent*100, remaining_time, loss/(index+1), accuracy/(index+1)*100))
    sys.stdout.flush()

def epoch_train(train_fun, mini_batches, num_batches, verbose=0):        
    if verbose == 1:
        start_time = time.time()

    epoch_loss = 0
    epoch_accuracy = 0
    for index in range(num_batches):
        X, y = next(mini_batches)
        loss, accuracy = train_fun(X, y)
        epoch_loss += loss
        epoch_accuracy += accuracy
        
        if verbose == 1:
            # progress bar
            progress_bar(start_time, index, num_batches, epoch_loss, epoch_accuracy)
    print "" 
    epoch_loss /= num_batches
    epoch_accuracy /= num_batches
    return epoch_loss, epoch_accuracy

def print_progress(valid_loss, valid_accuracy, name, epoch=0, num_epochs=0): 
    if num_epochs != 0:   
        sys.stdout.write("\rEpoch %d out of %d \n"%(epoch+1, num_epochs))
    else:
        print('Final Results:')
    print("  " + name + " loss:\t{:.6f}".format(valid_loss))
    print("  " + name + " accuracy:\t{:.2f} %".format(valid_accuracy*100))

def early_stopping(valid_memory, patience):
    min_loss = min(valid_memory)
    min_epoch = valid_memory.index(min_loss)
    current_loss = valid_memory[-1]
    current_epoch = len(valid_memory)
    status = True
    if min_loss < current_loss:
        if patience - (current_epoch - min_epoch) < 0:
            status = False
    return status