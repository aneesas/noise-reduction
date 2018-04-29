import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import struct
import wave

import fcn

SAMPLING_RATE = 16000  # Hz

tf.logging.set_verbosity(tf.logging.INFO)

checkpoints_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/fcn_model"
data_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/"
save_path = data_path+"testwav/network1/"

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

# Save data (50 predictions) as wav files in testwav to then calculate PESQ, STOI
pred1 = np.hstack(predictions[:50])
with wave.open(save_path+"pred1.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred1))+'f', *pred1))

org1 = eval_data_clean[:50].reshape((50*512,))
with wave.open(save_path+"org1.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org1))+'f', *org1))

noisy1 = eval_data_noisy[:50].reshape((50*512,))
with wave.open(save_path+"noisy1.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy1))+'f', *noisy1))

# Plot the stuff you care about, save it
t = np.arange(0, 50*512)*1000/SAMPLING_RATE + 100
plt.figure()
plt.plot(t, noisy1, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred1, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org1, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FCJF0, Utterance SA1 w/N91 (Yawn)")
plt.legend()
plt.savefig(save_path+"f2_SA1_n91.png")
plt.show()




