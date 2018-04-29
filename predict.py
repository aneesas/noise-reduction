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

checkpoints_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/fcn_model_M15_K10_10000steps"
data_path = "/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/"
save_path = data_path+"testwav/network5/"

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

# Get network output on first 20 training inputs
predict_input_fn_2 = tf.estimator.inputs.numpy_input_fn(
	x={"x": train_data_noisy[:98*20]},
	num_epochs=1,
	batch_size=100,
	shuffle=False)

predictions_train = list(fcn_DAE.predict(input_fn=predict_input_fn_2))

predictions_train = [p["output"] for p in predictions_train]

# Save data (50 predictions) as wav files in testwav to then calculate PESQ, STOI
# F2, N91
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
plt.title("Speaker FAEM0 (DR2), Utterance SA1 w/N91 (Yawn)")
plt.legend()
plt.savefig(save_path+"f2_SA1_n91.png")
plt.show()


# F2, WGN
pred2 = np.hstack(predictions[980:980+50])
with wave.open(save_path+"pred2.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred2))+'f', *pred2))

org2 = eval_data_clean[980:980+50].reshape((50*512,))
with wave.open(save_path+"org2.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org2))+'f', *org2))

noisy2 = eval_data_noisy[980:980+50].reshape((50*512,))
with wave.open(save_path+"noisy2.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy2))+'f', *noisy2))

plt.figure()
plt.plot(t, noisy2, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred2, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org2, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FAEM0 (DR2), Utterance SA1 w/WGN")
plt.legend()
plt.savefig(save_path+"f2_SA1_WGN.png")
plt.show()


# F3, N91
# 98*11 + 85*11 = 2013
pred3 = np.hstack(predictions[2013:2013+50])
with wave.open(save_path+"pred3.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred3))+'f', *pred3))

org3 = eval_data_clean[2013:2013+50].reshape((50*512,))
with wave.open(save_path+"org3.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org3))+'f', *org3))

noisy3 = eval_data_noisy[2013:2013+50].reshape((50*512,))
with wave.open(save_path+"noisy3.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy3))+'f', *noisy3))

plt.figure()
plt.plot(t, noisy3, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred3, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org3, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FALK0 (DR3), Utterance SA1 w/N91 (Yawn)")
plt.legend()
plt.savefig(save_path+"f3_SA1_n91.png")
plt.show()

# F3, WGN
# 2013 + 94*10 = 2953
pred4 = np.hstack(predictions[2953:2953+50])
with wave.open(save_path+"pred4.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred4))+'f', *pred4))

org4 = eval_data_clean[2953:2953+50].reshape((50*512,))
with wave.open(save_path+"org4.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org4))+'f', *org4))

noisy4 = eval_data_noisy[2953:2953+50].reshape((50*512,))
with wave.open(save_path+"noisy4.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy4))+'f', *noisy4))

plt.figure()
plt.plot(t, noisy4, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred4, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org4, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FALK0 (DR3), Utterance SA1 w/WGN")
plt.legend()
plt.savefig(save_path+"f3_SA1_WGN.png")
plt.show()


# F1, N1
pred5 = np.hstack(predictions_train[:50])
with wave.open(save_path+"pred5.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred5))+'f', *pred5))

org5 = train_data_clean[:50].reshape((50*512,))
with wave.open(save_path+"org5.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org5))+'f', *org5))

noisy5 = train_data_noisy[:50].reshape((50*512,))
with wave.open(save_path+"noisy5.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy5))+'f', *noisy5))

plt.figure()
plt.plot(t, noisy5, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred5, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org5, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FAEM0 (DR2), Utterance SA1 w/N1 (Crowd)")
plt.legend()
plt.savefig(save_path+"f2_SA1_n1.png")
plt.show()

# F1, N18
pred6 = np.hstack(predictions_train[98*17:98*17+50])
with wave.open(save_path+"pred6.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(pred6))+'f', *pred6))

org6 = train_data_clean[98*17:98*17+50].reshape((50*512,))
with wave.open(save_path+"org6.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(org6))+'f', *org6))

noisy6 = train_data_noisy[98*17:98*17+50].reshape((50*512,))
with wave.open(save_path+"noisy6.wav", 'w') as fp:
	fp.setnchannels(1)
	fp.setsampwidth(4)
	fp.setframerate(16000)
	fp.writeframes(struct.pack('<'+str(len(noisy6))+'f', *noisy6))

plt.figure()
plt.plot(t, noisy6, linewidth=1.0, label="Noisy speech")
plt.plot(t, pred6, linewidth=1.0, label="Predicted clean speech")
plt.plot(t, org6, linewidth=1.0, label="Original clean speech")
plt.xlabel("Time (ms)")
plt.ylabel("Normalized amplitude")
plt.title("Speaker FAEM0 (DR2), Utterance SA1 w/N18 (Machine)")
plt.legend()
plt.savefig(save_path+"f2_SA1_n18.png")
plt.show()

