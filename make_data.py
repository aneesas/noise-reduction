# Make clean/noisy utterance pairs using TIMIT data and OSU noise
# Aneesa Sonawalla
# April 2018

import os
import wave
import nnresample
import struct
import numpy as np

SAMPLE_RATE = 8000 # Hz

path = '/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/'
if not os.path.isdir(path+'data'):
	os.mkdir(path+'data')

# Load noise types--will need these repeatedly
# Resample them to 8 kHz
noise_path = '/Users/aneesasonawalla/Documents/GT/ECE6255/Project/OSU_100noise/'
training_noise_filenames = ['n001.wav', 'n045.wav', 'n021.wav']
testing_noise_filenames = ['n089.wav'] # and WGN for stationary noise

train_noise = []
for filename in training_noise_filenames:
	fp = wave.open(noise_path+filename, mode='rb')
	raw_data = fp.readframes(fp.getnframes())
	signal = struct.unpack('<'+str(fp.getnframes())+'h', raw_data)
	train_noise.append(nnresample.resample(signal, SAMPLE_RATE,
										   fp.getframerate()))
	fp.close()

test_noise = []
for filename in testing_noise_filenames:
	fp = wave.open(noise_path+filename, mode='rb')
	raw_data = fp.readframes(fp.getnframes())
	signal = struct.unpack('<'+str(fp.getnframes())+'h', raw_data)
	test_noise.append(nnresample.resample(signal, SAMPLE_RATE,
										  fp.getframerate()))
	fp.close()


# Iterate through training data to get names of all files? Randomly pick x of them from list?

# Then copy each to data/train/clean with the original name
# Load original, add noise, and save in data/train/noisy


# Do the same for test data