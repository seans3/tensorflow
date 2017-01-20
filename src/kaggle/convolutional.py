# ==============================================================================
#
# File:   convolutional.py
# Author: Sean Sullivan (sean@seansullivan.io)
# Date:   Jan. 17, 2017
# Description:
#   Convolutional Neural Network using tensorflow to classify images.
#   Used to classify MNIST digit images and Kaggle leaf images.
#
#   Kaggle leaf competition: https://www.kaggle.com/c/leaf-classification
#
#   To run leaf classification use: --leaf_images flag
#
# NOTE: Modified from the same named file from Google. Copyright for that
# file follows:
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
from datetime import datetime
import random

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pandas as pd
import PIL.Image
import PIL.ImageOps

# Constants for kaggle leaf classification (not MNIST)
KAGGLE_VALIDATION_SIZE = 80  # Size of the validation set.
KAGGLE_BATCH_SIZE      = 20
KAGGLE_EVAL_BATCH_SIZE = 20
KAGGLE_IMAGE_SIZE      = 64
KAGGLE_LABELS          = 10
DATA_DIR               = 'data/kaggle'
IMAGE_DIR              = os.path.join(DATA_DIR, 'images')
TRAIN_CSV_FILENAME     = 'train.csv'
NUM_LEAVES_TO_TRAIN    = 10    # If less than full 99 leaf species


# Constants for MNIST classification
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None


####################################################
#    Classes
####################################################


class Image:
    """Encasulates an image, it's meta-data and it's pixel data."""

    dir = ''        # directory for image file
    filename = ''   # filename for image
    num_rows = -1   # number of rows for image resize
    num_cols = -1   # num of columns for image resize
    is_opened = False  # is image file opened
    img = None         # reference to opened image file

    def __init__(self, dir, filename):
      self.dir = dir
      self.filename = filename
      
    def __init__(self, dir, filename, num_rows, num_cols):
      self.dir = dir
      self.filename = filename
      self.num_rows = num_rows
      self.num_cols = num_cols
      
    def getFilepath(self):
      """Returns full filepath to image file."""
      return os.path.join(self.dir, self.filename)

    def open(self):
      """Stores a reference to the opened image file if not already open. """
      if not self.is_opened:
        self.img = PIL.Image.open(self.getFilepath())
        self.img = self.img.resize((self.num_rows, self.num_cols))
        is_opened = True
      return self.img

    def close(self):
      """Saves file handles by closing an open image file."""
      if self.is_opened:
        img.close()
        self.is_opened = False
          
    def getData(self):
      """Returns numpy array filled with resized and re-scaled image pixel data."""
      self.open()
      a = numpy.array(self.img).astype(numpy.float32)
      a = (a - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  # pixel data [0, 255] -> [-0.5, 0.5]
      new_shape = a.shape + (1,)
      a = a.reshape(new_shape)  # reshape from [x, x] -> [x, x, 1] (for channels)
      return a

    def getRotatedData(self, degrees):
      """Returns numpy array filled with resized, rotated, re-scaled image pixel data."""
      self.open()
      rotated_img = self.img.rotate(degrees)       # rotation should be 90, 180, or 270
      a = numpy.array(rotated_img).astype(numpy.float32)
      a = (a - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH  # pixel data [0, 255] -> [-0.5, 0.5]
      new_shape = a.shape + (1,)
      a = a.reshape(new_shape)  # reshape from [x, x] -> [x, x, 1] (for channels)
      return a

    def invertImage(self):
      """Inverts colors of opened image file (black -> white, white -> black)."""
      self.open()
      self.img = PIL.ImageOps.invert(self.img)
    

class LeafImage(Image):
    """Encapsulates leaf image from Kaggle competition. Inherits from Image."""
    
    species = None  # String. Example: Acer_Saccharinum or Quercus_Pontica
    label = None    # Integer. Example 4

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


# TODO(sean): This is too long. Break it up. Create TrainingData class with
#   simple interface to get next batch of training data. Example:
#
#   batch_x, batch_y = training_data.next_batch(batch_size=20)
def extract_kaggle_data(train_data_dir):
    """Generate the kaggle leaf training, label, validation_data and label arrays."""
    # Create the dictionary of label (species) -> list of LeafImages
    train_filepath = os.path.join(train_data_dir, TRAIN_CSV_FILENAME)
    print_line('Reading: ' + train_filepath)
    training_csv = pd.read_csv(train_filepath)
    ids = training_csv["id"]
    print_line('Num Images: %d' % len(ids))
    species = training_csv["species"]
    # Assert ids and species are same size
    species_image_map = {}
    for i in range(len(ids)):
        id = ids[i]
        leaf_species = species[i]
        leaf_filename = "%d.jpg" % id
        leaf = LeafImage(IMAGE_DIR, leaf_filename, KAGGLE_IMAGE_SIZE, KAGGLE_IMAGE_SIZE)
        leaf.setSpecies(leaf_species)
        if leaf_species not in species_image_map:
            species_image_map[leaf_species] = []
        species_image_map.get(leaf_species).append(leaf)

    # choose a subset of the leaves to start (not entire 99 leaf species)
    species = species_image_map.keys()
    random.shuffle(species)
    species = species[:NUM_LEAVES_TO_TRAIN]

    # Create the integer labels for each leaf species of the subset
    label_map = {}
    current_label = 0
    for leaf in species:
        label_map[leaf] = current_label
        current_label += 1

    # Create a list of tuples of (label, pixel data array)
    validation_items = []
    training_items = []
    for leaf_species in species:
      leaf_images = species_image_map.get(leaf_species)
      # 2 out of 10 leaf images (20%) are used to validate
      validation_images = leaf_images[:2]
      leaf_images = leaf_images[2:]
      # assert 10 LeafImages
      label = label_map[leaf_species]
      for leaf_image in leaf_images:
        training_items.append((label, leaf_image.getData()))
        training_items.append((label, leaf_image.getRotatedData(90)))
        training_items.append((label, leaf_image.getRotatedData(180)))
        training_items.append((label, leaf_image.getRotatedData(270)))
        leaf_image.close()
      for validation_image in validation_images:
        validation_items.append((label, validation_image.getData()))
        validation_items.append((label, validation_image.getRotatedData(90)))
        validation_items.append((label, validation_image.getRotatedData(180)))
        validation_items.append((label, validation_image.getRotatedData(270)))
        validation_image.close()

    random.shuffle(training_items)
    random.shuffle(validation_items)

    train_size = len(training_items)
    validation_size = len(validation_items)
    print("Num Training Data Points: %d" % train_size)
    print("Num Validation Data Points: %d" % validation_size)
    
    # Create numpy array of training data and labels
    training_data = numpy.ndarray(shape=(train_size, KAGGLE_IMAGE_SIZE, KAGGLE_IMAGE_SIZE, 1),
                                  dtype=numpy.float32)
    training_labels = numpy.zeros(shape=(train_size,), dtype=numpy.int64)
    for i in xrange(train_size):
      training_item = training_items[i]
      current_label = training_item[0]
      current_array = training_item[1]
      training_labels[i] = current_label
      training_data[i] = current_array

    # Create numpy array of validation data and labels
    validation_data = numpy.ndarray(
      shape=(validation_size, KAGGLE_IMAGE_SIZE, KAGGLE_IMAGE_SIZE, 1), dtype=numpy.float32)
    validation_labels = numpy.zeros(shape=(validation_size,), dtype=numpy.int64)
    for i in xrange(validation_size):
      validation_item = validation_items[i]
      current_label = validation_item[0]
      current_array = validation_item[1]
      validation_labels[i] = current_label
      validation_data[i] = current_array

    return training_data, training_labels, validation_data, validation_labels


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):

  batch_size = BATCH_SIZE
  image_size = IMAGE_SIZE
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  elif FLAGS.leaf_images:
    print('Kaggle Leaves')
    # Extract it into numpy arrays.
    train_data, train_labels, validation_data, validation_labels = extract_kaggle_data(DATA_DIR)
    num_epochs = NUM_EPOCHS
    batch_size = KAGGLE_BATCH_SIZE
    image_size = KAGGLE_IMAGE_SIZE
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(batch_size, image_size, image_size, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(batch_size,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(batch_size, image_size, image_size, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([image_size // 4 * image_size // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * batch_size,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  #optimizer = tf.train.MomentumOptimizer(learning_rate,
  #                                       0.9).minimize(loss,
  #                                                     global_step=batch)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss);

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, batch_size):
      end = begin + batch_size
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-batch_size:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  print_line('Starting...')
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    # num_steps = int(num_epochs * train_size) // batch_size
    num_steps = 10000
    print('Num Steps: %d' % num_steps)
    for step in xrange(num_steps):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * batch_size) % (train_size - batch_size)
      batch_data = train_data[offset:(offset + batch_size), ...]
      batch_labels = train_labels[offset:(offset + batch_size)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * batch_size / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()

    # Finally print the result!
    # test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    # print('Test error: %.1f%%' % test_error)
    # print('Finished: ' + str(datetime.now()))
    # if FLAGS.self_test:
    #   print('test_error', test_error)
    #   assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
    #       test_error,)

  print_line('FINISHED')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')
  parser.add_argument(
      '--leaf_images',
      default=False,
      action='store_true',
      help='True if classifying kaggle leaves.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

