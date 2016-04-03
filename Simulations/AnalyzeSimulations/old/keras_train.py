import os
import numpy as np
from six.moves import cPickle
import os.path

from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1l2, activity_l1l2

#------------------------------------------------------------------------------
filename = 'N=500000_S=200_M=10_G=20_data.pickle'

# setup paths for file handling
filepath = os.path.join('data',filename)
name, ext = os.path.splitext(filename)
outdir = os.path.join('data',name)
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    print "making directory: " + outdir
outpath = os.path.join(outdir,'model_log.hdf5')

# load training set
print "loading data from: " + filepath
f = open(filepath, 'rb')
print "loading train data"
train = cPickle.load(f)
print "loading cross-validation data"
cross_validation = cPickle.load(f)
print "loading test data"
test = cPickle.load(f)
f.close()

X_train = train[0].transpose((0,2,1))
y_train = train[1]
X_valid = cross_validation[0].transpose((0,2,1))
y_valid = cross_validation[1]
X_test = test[0].transpose((0,2,1))
y_test = test[1]
num_data, seq_length, dim = X_train.shape


#------------------------------------------------------------------------------
# Model hyper-parameters

# convolutional layer parameters
conv_filters = [100, 200, 200]     # number of convolution filters for each layer
filter_length = [16, 4, 4]    # filter length for each layer
pool_length = [4, 4]         # max pool length for each layer
conv_activation="relu"          # convolution activation units
W_l1_conv = 0.00000000                # l1 weight decay convolutional layer
W_l2_conv = 0.0000000                # l2 weight decay convolutional layer
b_l1_conv = 0                # l1 weight decay convolutional layer
b_l2_conv = 0                # l2 weight decay convolutional layer
dropout_conv = 0.2                   # dropout rate

# fully connected layer parameters
fully_connected = [200]     # number of fully connected units
fc_activation = 'relu'          # fully connected activation units
W_l1_fc = 0.00000000                  # l1 weight decay dense layer
W_l2_fc = 0.0000000                  # l2 weight decay dense layer
b_l1_fc = 0                  # l1 weight decay dense layer
b_l2_fc = 0                  # l2 weight decay dense layer
dropout_fc = 0.5                   # dropout rate

# output layer parameters
num_labels = y_train.shape[1]   # number of labels (output units)
output_activation = 'sigmoid'   # activation for output unit (prediction)

# optimization parameters
optimizer = 'rmsprop'           # optimizer (rmsprop, adam, sgd, adagrad)
loss = 'categorical_crossentropy'#'binary_crossentropy'    # loss function to minimize (mse, binary_crossentropy,)
batch_size = 128                # mini-batch size 
nb_epoch = 100                  # number of epochs to train

# figure out fan-in for each layer 
input_length = [seq_length, seq_length/pool_length[0], seq_length/pool_length[0]/pool_length[1]]
input_length = np.round(input_length).astype(int)

#------------------------------------------------------------------------------

# Deep learning model
model = Sequential()

# convolutional layer 1 
model.add(Convolution1D(input_dim=4,
                        input_length=input_length[0],
                        nb_filter=conv_filters[0],
                        filter_length=filter_length[0],
                        border_mode="same",
                        #init='normal', 
                        activation=conv_activation,
                        subsample_length=1,
			W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv),
                        b_regularizer=l1l2(l1=b_l1_conv, l2=b_l2_conv)))			
model.add(MaxPooling1D(pool_length=pool_length[0], stride=pool_length[0]))


# convolutional layer 2
model.add(Convolution1D(input_dim=conv_filters[0],
                        input_length=input_length[1],
                        nb_filter=conv_filters[1],
                        filter_length=filter_length[1],
                        border_mode="same",
                        #init='normal', 
                        activation=conv_activation,
                        subsample_length=1,
                        W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv),
                        b_regularizer=l1l2(l1=b_l1_conv, l2=b_l2_conv)))                     
model.add(MaxPooling1D(pool_length=pool_length[1], stride=pool_length[1]))
model.add(Dropout(dropout_conv))

# convolutional layer 3
model.add(Convolution1D(input_dim=conv_filters[1],
                        input_length=input_length[2],
                        nb_filter=conv_filters[2],
                        filter_length=filter_length[2],
                        #init='normal', 
                        border_mode="same",
                        activation=conv_activation,
                        subsample_length=1,
                        W_regularizer=l1l2(l1=W_l1_conv, l2=W_l2_conv),
                        b_regularizer=l1l2(l1=b_l1_conv, l2=b_l2_conv)))                         


# flatten feature maps for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(input_dim=input_length[2]*conv_filters[2], 
                output_dim=fully_connected[0], 
                #init='normal', 
                activation=fc_activation, 
                W_regularizer=l1l2(l1=W_l1_fc, l2=W_l2_fc),
                b_regularizer=l1l2(l1=b_l1_fc, l2=b_l2_fc))) 
model.add(Dropout(dropout_conv))

# sigmoid output layer
model.add(Dense(input_dim=fully_connected[0], 
                output_dim=num_labels,
                #init='normal', 
                activation=output_activation, 
                W_regularizer=l1l2(l1=W_l1_fc, l2=W_l2_fc),
                b_regularizer=l1l2(l1=b_l1_fc, l2=b_l2_fc))) 
#model.add(Dropout(dropout_fc))

# loss function and optimization method
model.compile(loss=loss, 
              optimizer=optimizer)

# save models during training
checkpointer = ModelCheckpoint(filepath=os.path.join(outdir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
                               verbose=0, 
                               save_best_only=False,
                               monitor='val_loss')

# early stopping with validation loss
earlystopper = EarlyStopping(monitor='val_loss', 
                             patience=5, 
                             verbose=1)

# train model
model.fit(X_train, y_train, batch_size=batch_size, 
                            nb_epoch=nb_epoch, 
                            shuffle=True, 
                            show_accuracy=True, 
            			    validation_data=(X_valid, y_valid), 
            			    callbacks=[checkpointer, earlystopper])

# run trained model on test set
results = model.evaluate(X_test, y_test, show_accuracy=True)

print results

