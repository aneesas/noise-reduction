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
import glob
import h5py
import os

tf.logging.set_verbosity(tf.logging.INFO)

INPUT_LENGTH = 512  # number of samples in each input waveform
NUM_FILTERS = 5
FILTER_SIZE = 20
LEARNING_RATE = 0.1
BATCH_SIZE = 100
NUM_EPOCHS = 1
NUM_STEPS = 10000

checkpoints_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/fcn_model"
data_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/"

# 1 convolutional layer, with padding (i.e., zero-pad so that output size = input size)
# 15 filters, each with a filter size of 11
# ReLU activation functions, Adam training w/batch normalization
# Objective function: minimize MSE between clean and noisy waveforms

def fcn_model(features, labels, mode):
    """Model function for FCN"""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, INPUT_LENGTH, 1])

    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=NUM_FILTERS,
        kernel_size=FILTER_SIZE,
        padding="same",
        activation=None)

    conv2 = tf.layers.conv1d(
        inputs=conv1,
        filters=1,
        kernel_size=FILTER_SIZE,
        padding="same",
        #activation=tf.nn.relu)
        activation=None)

    output = tf.reshape(conv2, [-1, INPUT_LENGTH])

    predictions = {
        "output": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(
        labels,
        output)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=output)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    print("Loading train and eval data")
    # Load training and eval data
    # Data should have dimensions (number of signals, signal length in samples)

    if os.path.exists(data_path+"training_data_full.npz"):
        print("Loading train data from npz file")
        npzfile = np.load(data_path+"training_data_full.npz")
        train_data_noisy = npzfile["noisy"].astype('float32')
        train_data_clean = npzfile["clean"].astype('float32')
    else:
        train_data_noisy = np.empty((0, INPUT_LENGTH), dtype='float32')
        train_data_clean = np.empty((0, INPUT_LENGTH), dtype='float32')

        train_data_files = sorted(glob.glob(data_path+"training_data_*.hdf5"))
        for f in train_data_files:
            print("Loading from file " + os.path.basename(f))
            with h5py.File(f, 'r') as fp:
                clean = fp["clean"].value  # loads as ndarray
                num_segments = len(clean)//INPUT_LENGTH
                clean = clean[:num_segments*INPUT_LENGTH]
                clean_split = clean.reshape((num_segments, INPUT_LENGTH))

                for i in range(1,91):
                    noisy = fp["n{:03d}".format(i)].value
                    noisy = noisy[:num_segments*INPUT_LENGTH]
                    noisy_split = noisy.reshape((num_segments, INPUT_LENGTH))
                    train_data_noisy = np.vstack((train_data_noisy, noisy_split))
                    train_data_clean = np.vstack((train_data_clean, clean_split))

        train_data_noisy = train_data_noisy.astype('float32')
        train_data_clean = train_data_clean.astype('float32')

        # Save as npz file for easier future use
        np.savez(data_path+"training_data_full.npz",
                 noisy=train_data_noisy,
                 clean=train_data_clean)
        print("Saved train data as npz")

    if os.path.exists(data_path+"testing_data_full.npz"):
        print("Loading eval data from npz file")
        npzfile = np.load(data_path+"testing_data_full.npz")
        eval_data_noisy = npzfile["noisy"].astype('float32')
        eval_data_clean = npzfile["clean"].astype('float32')
    else:
        eval_data_noisy = np.empty((0, INPUT_LENGTH), dtype='float32')
        eval_data_clean = np.empty((0, INPUT_LENGTH), dtype='float32')

        eval_data_files = sorted(glob.glob(data_path+"testing_data_*.hdf5"))
        for f in eval_data_files:
            print("Loading from file " + os.path.basename(f))
            with h5py.File(f, 'r') as fp:
                clean = fp["clean"].value
                num_segments = len(clean)//INPUT_LENGTH
                clean = clean[:num_segments*INPUT_LENGTH]
                clean_split = clean.reshape((num_segments, INPUT_LENGTH))

                for i in range(91, 101):
                    noisy = fp["n{:03d}".format(i)].value
                    noisy = noisy[:num_segments*INPUT_LENGTH]
                    noisy_split = noisy.reshape((num_segments, INPUT_LENGTH))
                    eval_data_noisy = np.vstack((eval_data_noisy, noisy_split))
                    eval_data_clean = np.vstack((eval_data_clean, clean_split))
                noisy = fp["WGN"].value
                noisy = noisy[:num_segments*INPUT_LENGTH]
                noisy_split = noisy.reshape((num_segments, INPUT_LENGTH))
                eval_data_noisy = np.vstack((eval_data_noisy, noisy_split))
                eval_data_clean = np.vstack((eval_data_clean, clean_split))

        eval_data_noisy = eval_data_noisy.astype('float32')
        eval_data_clean = eval_data_clean.astype('float32')

        # Save as npz file for easier future use
        np.savez(data_path+"testing_data_full.npz",
                 noisy=eval_data_noisy,
                 clean=eval_data_clean)
        print("Saved eval data as npz")

    # Create the estimator
    fcn_DAE = tf.estimator.Estimator(model_fn=fcn_model, model_dir=checkpoints_path)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data_noisy},
        y=train_data_clean,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    fcn_DAE.train(input_fn=train_input_fn, steps=NUM_STEPS)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data_noisy},
        y=eval_data_clean,
        num_epochs=1,
        shuffle=False)
    eval_results = fcn_DAE.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()