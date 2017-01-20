#!/usr/bin/python
####################################################
#
# File: kaggle.py
# Author: Sean Sullivan (sean@seansullivan.io)
# Date: Jan. 18, 2017
# Description: Script to create a convolutional
#   neural net for image classification for leaf
#   species for a Kaggle competition.
#
#   https://www.kaggle.com/c/leaf-classification
#
####################################################

import sys
import os
import random
from datetime import datetime
import PIL.Image
import PIL.ImageOps
import tensorflow as tf
import numpy as np
import pandas as pd


####################################################
#    Constants
####################################################
DATA_DIR     = 'data/kaggle'
IMAGE_DIR    = os.path.join(DATA_DIR, 'images')
IMAGE_SIZE   = 128
N_INPUT      = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1   # Greyscale, not RGB--1 channel
PIXEL_DEPTH  = IMAGE_SIZE - 1
TRAIN_CSV_FILENAME  = 'train.csv'
TEST_CSV_FILENAME   = 'test.csv'
NUM_LEAVES_TO_TRAIN = 10    # If less than full 99
N_CLASSES = NUM_LEAVES_TO_TRAIN
DROPOUT_RATE        = 0.75  # Probability to keep unit
BATCH_SIZE          = 20
TRAINING_ITERATIONS = 10000


####################################################
#    Classes
####################################################

class Image:
    """Encasulates an image, it's meta-data and it's pixel data.
    """

    dir = ''
    filename = ''
    num_rows = -1
    num_cols = -1
    is_opened = False
    img = None

    def __init__(self, dir, filename):
        self.dir = dir
        self.filename = filename
    
    def __init__(self, dir, filename, num_rows, num_cols):
        self.dir = dir
        self.filename = filename
        self.num_rows = num_rows
        self.num_cols = num_cols
        
    def getFilepath(self):
        return os.path.join(self.dir, self.filename)

    def open(self):
        if not self.is_opened:
            self.img = PIL.Image.open(self.getFilepath())
            self.img = self.img.resize((self.num_rows, self.num_cols))
            is_opened = True
        return self.img

    def close(self):
        if self.is_opened:
            img.close()
            self.is_opened = False
            
    def getData(self):
        """Returns numpy array filled with resized and re-scaled image pixel data.
        """

        self.open()
        a = np.array(self.img).astype(np.float32)
        a = (a - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        return a

    def getRotatedData(self, degrees):
        """Returns numpy array filled with resized and re-scaled image pixel data.
        """

        self.open()
        rotated_img = self.img.rotate(degrees)
        a = np.array(rotated_img).astype(np.float32)
        a = (a - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        return a

    def invertImage(self):
        """Returns numpy array filled with resized and re-scaled image pixel data.
        """

        self.open()
        self.img = PIL.ImageOps.invert(self.img)
    

class LeafImage(Image):
    """Encapsulates leaf image from Kaggle competition.
    """
    
    species = None
    label = None

    def __init__(self, dir, filename):
        Image.__init__(self, dir, filename)
    
    def __init__(self, dir, filename, num_rows, num_cols):
        Image.__init__(self, dir, filename, num_rows, num_cols)

    def setSpecies(self, species):
        self.species = species

    def getSpecies(self):
        return self.species

    def setLabel(self, label):
        self.label = label

    def getLabel(self):
        return self.label


class TrainingData:

    arr = None
    tensors_and_labels = []
    current_offset = 0

    def getNumItems(self):
        return len(tensors_and_labels)
    
    def addItem(self, item):
        tensors_and_lables.append(item)

    def createDataArray(self):
        pass
        
    def getAsArray(self):
        if arr is None:
            pass
        
    def nextBatch(self, batch_size):
        return None

    def createValidationSet(self, percent):
        return None



class ConvolutionalNeuralNetwork():
    """
    """

    layers = None
    
    def __init__(self, image_size, ):
        layers = []


        
####################################################
#    Routines
####################################################
def print_line(string):
    """Prepends the current time before printing the passed string.
    
    Arg: string -- passed string to print
    """
    s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s += ": "
    s += string
    print(s)


####################################################
#    Main
####################################################
if __name__ == '__main__':

    print_line('Starting...')

    # Load the images and perform the transforms
    #print_line('Loading Images: ' + IMAGE_DIR)
    #images = []
    #for image_filename in os.listdir(IMAGE_DIR):
        #img = LeafImage(IMAGE_DIR, image_filename, IMAGE_SIZE, IMAGE_SIZE)
        #images.append(img)
    #print_line('Num Images: %d' % len(images))

    # Create the dictionary of label (species) -> list of LeafImages
    train_filepath = os.path.join(DATA_DIR, TRAIN_CSV_FILENAME)
    print_line('Reading: ' + train_filepath)
    train = pd.read_csv(train_filepath)
    ids = train["id"]
    print_line('Num ids: %d' % len(ids))
    species = train["species"]
    # Assert ids and species are same size
    data_dict = {}
    for i in range(len(ids)):
        id = ids[i]
        leaf_species = species[i]
        leaf_filename = "%d.jpg" % id
        leaf = LeafImage(IMAGE_DIR, leaf_filename, IMAGE_SIZE, IMAGE_SIZE)
        leaf.setSpecies(leaf_species)
        if leaf_species not in data_dict:
            data_dict[leaf_species] = []
        data_dict.get(leaf_species).append(leaf)

    # Create numpy array of training data and labels
    train_data = None
    train_labels = np.zeros(shape=(400,), dtype=np.int64)
    num_leaves = 0
    keys = data_dict.keys()
    random.shuffle(keys)
    current_label = 1
    current_label_index = 0
    for key in keys:
        print_line('Training: ' + key)
        num_leaves += 1
        if num_leaves > NUM_LEAVES_TO_TRAIN:
            break
        for leaf in data_dict[key]:
            data = leaf.getData()
            if train_data is None:
                train_data = data
            else:
                train_data = np.dstack((train_data, data))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot1 = leaf.getRotatedData(90)
            train_data = np.dstack((train_data, rot1))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot2 = leaf.getRotatedData(180)
            train_data = np.dstack((train_data, rot2))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot3 = leaf.getRotatedData(270)
            train_data = np.dstack((train_data, rot3))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            leaf.close()
            species = leaf.getSpecies()
        current_label += 1
            
    train_data = np.transpose(train_data)
    new_shape = train_data.shape + (1,)
    train_data = train_data.reshape(new_shape)

    validation_data = train_data[:80]
    validation_labels = train_labels[:80]
    train_data = train_data[80:]
    train_labels = train_labels[80:]

    print('Train Shape: ', train_data.shape)
    train_size = train_data.shape[0]
    print('Train Labels Shape: ', train_labels.shape)

    print('Validation Shape: ', validation_data.shape)
    train_size = train_data.shape[0]
    print('Validation Labels Shape: ', validation_labels.shape)
    
    # Tensorflow variables
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    y = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 128, 128, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)
        
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 32*32*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([32*32*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    
    pred = conv_net(x, weights, biases, keep_prob)

    # Define loss and optimizer
    print('Pred Shape: ', pred.get_shape())
    print('Y Shape: ', y.get_shape())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=tf.float32)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    offset = 0
    with tf.Session() as sess:
        sess.run(init)
        print('Initialized')
        step = 1
        # Keep training until reach max iterations
        while step * BATCH_SIZE < TRAINING_ITERATIONS:
            batch_x = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_y = train_labels[offset:(offset + BATCH_SIZE)]
                                
            #batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT_RATE})
    
    print_line('Finished')

