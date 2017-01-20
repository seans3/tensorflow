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


KAGGLE_VALIDATION_SIZE = 80  # Size of the validation set.
KAGGLE_BATCH_SIZE = 20
KAGGLE_EVAL_BATCH_SIZE = 20
KAGGLE_IMAGE_SIZE = 64
KAGGLE_LABELS = 10
DATA_DIR     = 'data/kaggle'
IMAGE_DIR    = os.path.join(DATA_DIR, 'images')
TRAIN_CSV_FILENAME  = 'train.csv'
NUM_LEAVES_TO_TRAIN = 10    # If less than full 99

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
        a = numpy.array(self.img).astype(numpy.float32)
        a = (a - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        return a

    def getRotatedData(self, degrees):
        """Returns numpy array filled with resized and re-scaled image pixel data.
        """

        self.open()
        rotated_img = self.img.rotate(degrees)
        a = numpy.array(rotated_img).astype(numpy.float32)
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



def extract_kaggle_data(train_data_dir):
    """Generate the kaggle leaf data array and the label array."""

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
        leaf = LeafImage(IMAGE_DIR, leaf_filename, KAGGLE_IMAGE_SIZE, KAGGLE_IMAGE_SIZE)
        leaf.setSpecies(leaf_species)
        if leaf_species not in data_dict:
            data_dict[leaf_species] = []
        data_dict.get(leaf_species).append(leaf)

    # choose a subset of the leaves to start
    species = data_dict.keys()
    random.shuffle(species)
    species = species[:NUM_LEAVES_TO_TRAIN]

    # Create the labels for each leaf species
    label_map = {}
    current_label = 0
    for leaf in species:
        label_map[leaf] = current_label
        current_label += 1

    # Create a list of tuples of (label, pixel data array)

        
    # Create numpy array of training data and labels
    train_data = None
    train_labels = numpy.zeros(shape=(400,), dtype=numpy.int64)
    num_leaves = 0
    keys = data_dict.keys()
    
    current_label = 0
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
                train_data = numpy.dstack((train_data, data))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot1 = leaf.getRotatedData(90)
            train_data = numpy.dstack((train_data, rot1))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot2 = leaf.getRotatedData(180)
            train_data = numpy.dstack((train_data, rot2))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            rot3 = leaf.getRotatedData(270)
            train_data = numpy.dstack((train_data, rot3))
            train_labels[current_label_index] = current_label
            current_label_index += 1
            leaf.close()
            species = leaf.getSpecies()
        current_label += 1
            
    train_data = numpy.transpose(train_data)
    new_shape = train_data.shape + (1,)
    train_data = train_data.reshape(new_shape)

    return train_data, train_labels


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
    train_data, train_labels = extract_kaggle_data(IMAGE_DIR)

    # Generate a validation set.
    validation_data = train_data[:KAGGLE_VALIDATION_SIZE, ...]
    validation_labels = train_labels[:KAGGLE_VALIDATION_SIZE]
    train_data = train_data[KAGGLE_VALIDATION_SIZE:, ...]
    train_labels = train_labels[KAGGLE_VALIDATION_SIZE:]
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
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

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
  print('Starting: ' + str(datetime.now()))
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

