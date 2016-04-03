import numpy as np
import h5py
import scipy.io
import os
import sys

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1l2, activity_l1l2
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm

sys.path.append('/home/peter/Code/GenomeMotifs/utils')
from data_utils import load_DeepSea
from train_utils import batch_generator
from file_utils import make_directory

np.random.seed(247) # for reproducibility


foldername = 'DeepSea'
path = '/home/peter/Code/GenomeMotifs/Results/'
outdir = make_directory(foldername, path)

# load data and munge 
num_labels = 100
num_include = 500000
class_range = range(num_labels)
data_path = "/home/peter/Data/DeepSea"
train, valid, test = load_DeepSea(data_path, num_include, class_range)
num_data, dim, sequence_length = train[0].shape



#------------------------------------------------------------------------------
# Model hyper-parameters

# convolutional layer parameters
conv_filters = [300, 200, 200]     # number of convolution filters for each layer
filter_length = [19, 11, 7]    # filter length for each layer
pool_length = [3, 4, 4]         # max pool length for each layer
conv_activation="relu"          # convolution activation units
W_l1_conv = 0#.00000001                # l1 weight decay convolutional layer
W_l2_conv = 0#.0000005                # l2 weight decay convolutional layer
b_l1_conv = 0                # l1 weight decay convolutional layer
b_l2_conv = 0                # l2 weight decay convolutional layer
dropout_conv = 0.2                   # dropout rate

# fully connected layer parameters
fully_connected = [1000]     # number of fully connected units
fc_activation = 'relu'          # fully connected activation units
W_l1_fc = 0#.00000001                  # l1 weight decay dense layer
W_l2_fc = 0#.0000005                  # l2 weight decay dense layer
b_l1_fc = 0                  # l1 weight decay dense layer
b_l2_fc = 0                  # l2 weight decay dense layer
dropout_fc = 0.3                   # dropout rate
weight_norm = 10

# output layer parameters
num_labels = train[1].shape[1]   # number of labels (output units)
output_activation = 'sigmoid'   # activation for output unit (prediction)

# optimization parameters
optimizer = 'rmsprop'           # optimizer (rmsprop, adam, sgd, adagrad)
loss = 'binary_crossentropy'    # loss function to minimize (mse, binary_crossentropy,)
batch_size = 1000                # mini-batch size 
nb_epoch = 100                  # number of epochs to train

# figure out fan-in for each layer 
num_data, dim, seq_length = train[0].shape
input_length = [seq_length, seq_length/pool_length[0], seq_length/pool_length[0]/pool_length[1], seq_length/np.sum(pool_length[0])]
input_length = np.round(input_length).astype(int)

#------------------------------------------------------------------------------

# Deep learning model
model = Sequential()

# convolutional layer 1 
model.add(Convolution1D(input_dim=dim,
                        input_length=input_length[0],
                        nb_filter=conv_filters[0],
                        filter_length=filter_length[0],
                        init='glorot_uniform', 
                        border_mode="same",
                        subsample_length=1
                        #activation=conv_activation
#                        W_constraint = maxnorm(weight_norm), 
#                        W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv)
                        ))            
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling1D(pool_length=pool_length[0], stride=pool_length[0]))


# convolutional layer 2
model.add(Convolution1D(input_dim=conv_filters[0],
                        input_length=input_length[1],
                        nb_filter=conv_filters[1],
                        filter_length=filter_length[1],
                        init='glorot_uniform', 
                        border_mode="same",
                        subsample_length=1
                        #activation=conv_activation
                        #W_constraint = maxnorm(weight_norm), 
                        #W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv)
                        ))                    
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling1D(pool_length=pool_length[1], stride=pool_length[1]))

# convolutional layer 3
model.add(Convolution1D(input_dim=conv_filters[1],
                        input_length=input_length[2],
                        nb_filter=conv_filters[2],
                        filter_length=filter_length[2],
                        init='glorot_uniform', 
                        border_mode="same",
                        subsample_length=1
                        # activation=conv_activation
                        #W_constraint = maxnorm(weight_norm), 
                        #W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv)
                        ))                         
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling1D(pool_length=pool_length[2], stride=pool_length[2]))
model.add(Dropout(dropout_conv))


# flatten feature maps for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(input_dim=input_length[2]*conv_filters[2], 
                output_dim=fully_connected[0], 
                init='glorot_uniform'
                #activation=fc_activation
                #W_constraint = maxnorm(weight_norm), 
                #W_regularizer=l1l2(l1=W_l1_fc, l2=W_l2_fc)
                )) 
model.add(BatchNormalization())
model.add(PReLU())
model.add(Dropout(dropout_fc))

# sigmoid output layer
model.add(Dense(input_dim=fully_connected[0], 
                output_dim=num_labels,
                init='glorot_uniform', 
                #W_constraint = maxnorm(weight_norm), 
                activation=output_activation
                #W_regularizer=l1l2(l1=W_l1_fc, l2=W_l2_fc)
                )) 

# loss function and optimization method
# model.compile(loss=loss, optimizer=SGD(lr=.002, momentum=0.98, nersterov=True))
model.compile(loss=loss, 
              optimizer=optimizer)
# save models during training
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir,"bestmodel.hdf5"), # 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                               verbose=1, 
                               save_best_only=True,
                               monitor='val_loss')

# early stopping with validation loss
earlystopper = EarlyStopping(monitor='val_loss', 
                             patience=10, 
                             verbose=1)

# train model
model.fit(train[0], train[1], batch_size=batch_size, 
                            nb_epoch=nb_epoch, 
                            shuffle=True, 
                            show_accuracy=True, 
                            validation_data=(valid[0], valid[1]), 
                            callbacks=[checkpointer, earlystopper])

# run trained model on test set
results = model.evaluate(test[0], test[1], show_accuracy=True)

print results

