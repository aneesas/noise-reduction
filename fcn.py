# FCN implementation
# Modified from TensorFlow tutorial available at:
# https://www.tensorflow.org/tutorials/layers (Last accessed: April 2018)
# Aneesa Sonawalla
# April 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_LENGTH = 512  # number of samples in each input waveform
NUM_FILTERS = 15
FILTER_SIZE = 11

# 1 convolutional layer, with padding (i.e., zero-pad so that output size = input size)
# 15 filters, each with a filter size of 11
# ReLU activation functions, Adam training w/batch normalization
# Objective function: minimize MSE between clean and noisy waveforms

def fcn_model(noisy_in, clean_exp, mode):
    """Model function for FCN"""
    # Input layer
    input_layer = tf.reshape(noisy_in["x"], [-1, INPUT_LENGTH, 1])

    conv1 = tf.layers.conv1d(
        inputs = input_layer,
        filters = NUM_FILTERS,
        kernel_size = FILTER_SIZE,
        padding = 'same',
        activation = tf.nn.relu)

    loss = tf.losses.mean_squared_error(
        clean_exp,
        conv1,
        reduction = tf.losses.Reduction.MEAN)

if __name__ == "__main__":
    tf.app.run()