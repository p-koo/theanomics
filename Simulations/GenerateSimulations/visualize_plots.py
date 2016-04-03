from keras.utils.visualize_util import plot
from keras.utils.dot_utils import Grapher


plot(model, to_file='model.png')


grapher = Grapher()
grapher.plot(model, 'model.png')

# ____________________________________________
# Imports
import matplotlib.pyplot as plt

import theano

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation


# Model 
model = Sequential()

model.add(Convolution2D(28, 1, 3, 3, border_mode='full')) 
convout1 = Activation('relu')
model.add(convout1)


# Data loading + reshape to 4D
(X_train, y_train), (X_test, y_test) = mnist_dataset = mnist.load_data()
reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])


from random import randint
img_to_visualize = randint(0, len(X_train) - 1)


# Generate function to visualize first layer
convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
convolutions = convout1_f(reshaped[img_to_visualize: img_to_visualize+1])

midout_fn = theano.function([graph.inputs['input1'].get_input(train=False),
                                 graph.inputs['input2'].get_input(train=False)],
                                midout.get_output(train=False))
                                

%matplotlib inline
#The non-magical version of the previous line is this:
#get_ipython().magic(u'matplotlib inline')
imshow = plt.imshow #alias
plt.title("Image used: #%d (digit=%d)" % (img_to_visualize, y_train[img_to_visualize]))
imshow(X_train[img_to_visualize])


plt.title("First convolution:")
imshow(convolutions[0][0])


print "The second dimension tells us how many convolutions do we have: %s (%d convolutions)" % (
    str(convolutions.shape),
    convolutions.shape[1])


for i, convolution in enumerate(convolutions[0]):
    plt.figure()
    plt.title("Convolution %d" % (i))
    plt.imshow(convolution)