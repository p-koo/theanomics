import numpy as np
import h5py
import scipy.io
import tensorflow as tf

#------------------------------------------------------------------------------

# load training set
trainmat = h5py.File('train.mat')
validmat = scipy.io.loadmat('valid.mat')
testmat = scipy.io.loadmat('test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T

X_test = np.transpose(testmat['testxdata'],axes=(0,2,1))
y_test = testmat['testdata']
"""
X_valid = np.transpose(validmat['validxdata'],axes=(0,2,1))
y_valid = validmat['validdata']

"""

num_data, seq_length, dim = X_train.shape
num_labels = y_train.shape[1]


# X_train = X_train.reshape(num_data,seq_length*dim)
#X_test = X_test.reshape(X_test.shape[0],seq_length*dim)
#X_valid = X_valid.reshape(X_valid.shape[0],seq_length*dim)

#------------------------------------------------------------------------------

# sess = tf.InteractiveSession()

# create placeholders
x = tf.placeholder(tf.float32, shape=[None, seq_length, dim, 1])
# x_seq = tf.reshape(x, [-1, seq_length, dim, 1])
x_seq = tf.transpose(x, (0, 1, 3, 2))
y_ = tf.cast(tf.placeholder(tf.float32, shape=[None, num_labels]), 'float32')
	
# 1D convolutional layer - layer 1
filter_length_1 = 8;
num_filters_1 = 320; 
max_pool_1 = 4;
W_1 = tf.Variable(tf.truncated_normal([filter_length_1, 1, dim, num_filters_1], stddev=0.1))
b_1 = tf.Variable(tf.constant(0.1, shape=[num_filters_1]))
conv_1 = tf.nn.conv2d(x_seq, W_1, strides=[1, 1, 1, 1], padding='VALID')
h_1 = tf.nn.relu(conv_1 + b_1)
h_pool_1 = tf.nn.max_pool(h_1, ksize=[1, max_pool_1, 1, 1], strides=[1, max_pool_1, 1, 1], padding='SAME')

# 1D convolutional layer - layer 2
filter_length_2 = 8;
num_filters_2 = 480; 
max_pool_2 = 4;
W_2 = tf.Variable(tf.truncated_normal([filter_length_2, 1, num_filters_1, num_filters_2], stddev=0.1))
b_2 = tf.Variable(tf.constant(0.1, shape=[num_filters_2]))
conv_2 = tf.nn.conv2d(h_pool_1, W_2, strides=[1, 1, 1, 1], padding='VALID')
h_2 = tf.nn.relu(conv_2 + b_2)
h_pool_2 = tf.nn.max_pool(h_2, ksize=[1, max_pool_2, 1, 1], strides=[1, max_pool_2, 1, 1], padding='SAME')

# 1D convolutional layer - layer 3
filter_length_3 = 8;
num_filters_3 = 960; 
max_pool_3 = 2;
W_3 = tf.Variable(tf.truncated_normal([filter_length_3, 1, num_filters_2, num_filters_3], stddev=0.1))
b_3 = tf.Variable(tf.constant(0.1, shape=[num_filters_3]))
conv_3 = tf.nn.conv2d(h_pool_2, W_3, strides=[1, 1, 1, 1], padding='VALID')
h_3 = tf.nn.relu(conv_3 + b_3)
h_pool_3 = tf.nn.max_pool(h_3, ksize=[1, max_pool_3, 1, 1], strides=[1, max_pool_3, 1, 1], padding='SAME')

# fully connected layer 4
input_size = int(h_pool_3.get_shape()[1])*int(h_pool_3.get_shape()[3])
num_output = 925
W_fc1 = tf.Variable(tf.truncated_normal([input_size, num_output], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[num_output]))
input_flat = tf.reshape(h_pool_3, [-1, input_size])
h_fc1 = tf.nn.relu(tf.matmul(input_flat, W_fc1) + b_fc1)

# dropout 
h_fc1_drop = tf.nn.dropout(h_fc1, .5)

# fully connected output layer 5
W_fc2 = tf.Variable(tf.truncated_normal([num_output, num_labels], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_labels]))
y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy metric
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run session
sess = tf.Session()

# initialize variables
sess.run(tf.initialize_all_variables())

# mini-batch stochastic gradient descent
batchSize = 100;
num_batches = int(np.floor(num_data/batchSize))
for i in range(1):
    for j in xrange(num_batches):
		print "batch " + str(j) + " out of " + str(num_batches)
		index = range(j*batchSize,(j+1)*batchSize)
		x_batch = X_train[index].astype(np.float32)
		x_batch = x_batch.reshape(x_batch.shape + (1,)) 
		y_batch = y_train[index].astype(np.float32)
		sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})


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
