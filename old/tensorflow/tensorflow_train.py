import os
import sys
import time
import numpy as np
from six.moves import cPickle
import tensorflow as tf

#------------------------------------------------------------------------------
filename = 'N=100000_S=200_M=10_G=20_data.pickle'

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
num_labels = y_train.shape[1]   # number of labels (output units)

#------------------------------------------------------------------------------

# sess = tf.InteractiveSession()

# create placeholders
x = tf.placeholder(tf.float32, shape=[None, seq_length, dim, 1])
# x_seq = tf.reshape(x, [-1, seq_length, dim, 1])
x_seq = tf.transpose(x, (0, 1, 3, 2))
y_ = tf.cast(tf.placeholder(tf.float32, shape=[None, num_labels]), 'float32')
	
# 1D convolutional layer - layer 1
filter_length_1 = 12;
num_filters_1 = 200; 
max_pool_1 = 4;
W_1 = tf.Variable(tf.truncated_normal([filter_length_1, 1, dim, num_filters_1], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.05, shape=[num_filters_1]))
conv_1 = tf.nn.conv2d(x_seq, W_1, strides=[1, 1, 1, 1], padding='VALID')
h_1 = tf.nn.relu(conv_1 + b_1)
h_pool_1 = tf.nn.max_pool(h_1, ksize=[1, max_pool_1, 1, 1], strides=[1, max_pool_1, 1, 1], padding='SAME')

# 1D convolutional layer - layer 2
filter_length_2 = 8;
num_filters_2 = 200; 
max_pool_2 = 2;
W_2 = tf.Variable(tf.truncated_normal([filter_length_2, 1, num_filters_1, num_filters_2], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.05, shape=[num_filters_2]))
conv_2 = tf.nn.conv2d(h_pool_1, W_2, strides=[1, 1, 1, 1], padding='VALID')
h_2 = tf.nn.relu(conv_2 + b_2)
h_pool_2 = tf.nn.max_pool(h_2, ksize=[1, max_pool_2, 1, 1], strides=[1, max_pool_2, 1, 1], padding='SAME')

# fully connected layer 4
input_size = int(h_pool_2.get_shape()[1])*int(h_pool_2.get_shape()[3])
num_output = 100
W_fc1 = tf.Variable(tf.truncated_normal([input_size, num_output], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.05, shape=[num_output]))
input_flat = tf.reshape(h_pool_2, [-1, input_size])
h_fc1 = tf.nn.relu(tf.matmul(input_flat, W_fc1) + b_fc1)

# dropout 
h_fc1_drop = tf.nn.dropout(h_fc1, .2)

# fully connected output layer 5
W_fc2 = tf.Variable(tf.truncated_normal([num_output, num_labels], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.05, shape=[num_labels]))
y_out = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
expy = tf.exp(y_out)
sumexpy = tf.reduce_sum(expy)
y_out = tf.div(expy, sumexpy)

# cost function
#cross_entropy = -tf.reduce_sum(y_*tf.log(y_out))
cross_entropy = -tf.reduce_mean(y_*tf.log(y_out) + (1-y_)*tf.log(1-y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy metric
predictions = tf.cast(tf.greater_equal(y_out, 0.5), tf.float32)
correct_prediction = tf.equal(predictions, y_)
#correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run session
sess = tf.Session()

# initialize variables
sess.run(tf.initialize_all_variables())

# mini-batch stochastic gradient descent
batchSize = 500;
bar_length = 20
num_batches = num_data // batchSize
for i in range(100):

	acc = 0
	loss = 0
	start_time = time.time()
	for j in xrange(num_batches):
		#print "batch " + str(j) + " out of " + str(num_batches)
		index = range(j*batchSize,(j+1)*batchSize)
		x_batch = X_train[index].astype(np.float32)
		x_batch = x_batch.reshape(x_batch.shape + (1,)) 
		y_batch = y_train[index].astype(np.float32)
		sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
		acc += sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch})
		loss += sess.run(cross_entropy, feed_dict={x: x_batch, y_:y_batch})

		# progress bar
		remaining_time = (time.time()-start_time)*(num_batches-j-1)/(j+1)
		percent = (j+1.)/num_batches
		progress = '='*int(round(percent*bar_length))
		spaces = ' '*int(bar_length-round(percent*bar_length))
		sys.stdout.write("\rEpoch %d [%s] %.1f%% -- est.time=%ds -- cost=%.2f" \
					%(i+1, progress+spaces, percent*100, remaining_time, acc/(j+1)*100))
		sys.stdout.flush()
	sys.stdout.write("\n")


print "finished epoch"
print("test accuracy %g"% sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))

"""
# interactive session tests
sess.run(tf.initialize_all_variables())

j = 1
index = range(j*batchSize,(j+1)*batchSize)
x_batch = X_train[index].astype(np.float32)
x_batch = x_batch.reshape(x_batch.shape + (1,)) 
y_batch = y_train[index].astype(np.float32)
test = train_step.run(feed_dict={x: x_batch, y_: y_batch})
sess.close()

"""

"""
f = open(filename+'_model.pickle', 'wb')
options = cPickle.load(f)
model = cPickle.load(f)
seq_model = cPickle.load(f)
"""
