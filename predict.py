import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import fcn

tf.logging.set_verbosity(tf.logging.INFO)

checkpoints_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/fcn_model"
data_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/"

fcn_DAE = tf.estimator.Estimator(model_fn=fcn.fcn_model, model_dir=checkpoints_path)

npzfile = np.load(data_path+"training_data_full.npz")
train_data_noisy = npzfile["noisy"].astype('float32')
train_data_clean = npzfile["clean"].astype('float32')

npzfile = np.load(data_path+"testing_data_full.npz")
eval_data_noisy = npzfile["noisy"].astype('float32')
eval_data_clean = npzfile["clean"].astype('float32')

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": eval_data_noisy},
	num_epochs=1,
	batch_size=100,
	shuffle=False)

predictions = list(fcn_DAE.predict(input_fn=predict_input_fn))

predictions = [p["output"] for p in predictions]


# F2, 