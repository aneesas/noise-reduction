# Make clean/noisy utterance pairs using TIMIT data and OSU noise
# Aneesa Sonawalla
# April 2018

import os
import wave
import struct
import numpy as np
import glob
import h5py

SAMPLE_RATE = 16000 # Hz
SNR = 5  # dB

path = '/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/'

# Load noise types
noise_path = '/Users/aneesasonawalla/Documents/GT/ECE6255/Project/OSU_100noise/'
noise_files = sorted(glob.glob(noise_path+'*'))
training_noise_files = noise_files[:90]
testing_noise_files = noise_files[90:] # and WGN for stationary noise

train_noise = []
for filepath in training_noise_files:
	with wave.open(filepath, mode='rb') as fp:
		raw_data = fp.readframes(fp.getnframes())
		signal = np.array(struct.unpack('<'+str(fp.getnframes())+'h', raw_data),
						  dtype='float32')
		# Normalize to -1, 1
		signal = signal/np.max(signal)
		train_noise.append(signal)

test_noise = []
for filepath in testing_noise_files:
	with wave.open(filepath, mode='rb') as fp:
		raw_data = fp.readframes(fp.getnframes())
		signal = np.array(struct.unpack('<'+str(fp.getnframes())+'h', raw_data),
						  dtype='float32')
		# Normalize to -1, 1
		signal = signal/np.max(signal)
		test_noise.append(signal)


# Training data--all utterances from first female speaker in DR2
training_data_files = sorted(glob.glob(path+'originals/train/f2*'))

count = 0
for filepath in training_data_files:
	with wave.open(filepath, mode='rb') as fp:
		raw_data = fp.readframes(fp.getnframes())
		s = np.array(struct.unpack('<'+str(fp.getnframes())+'h', raw_data),
					 dtype='float32')
		# Normalize to -1, 1
		s = s/np.max(s)
		# Skip first and last 100 ms to avoid silence
		cutoff = int(SAMPLE_RATE*.1)
		s = s[cutoff:-cutoff]

	with h5py.File(path+'training_data_{}.hdf5'.format(count), 'w') as fp:
		fp.create_dataset("clean", data=s)

		# Add 90 types of noise at SNR and save each one
		noise_count = 1
		P_s = np.sum(np.abs(s)**2)/len(s)  # signal power
		for noise in train_noise:
			if len(noise) >= len(s):
				noise_ext = noise[:len(s)]
			else:
				num_repeats = int(np.ceil(len(s)/len(noise)))
				noise_ext = noise
				for n in range(num_repeats):
					noise_ext = np.concatenate((noise_ext, noise))
				noise_ext = noise_ext[:len(s)]

			# Adjust SNR
			P_n = np.sum(np.abs(noise_ext)**2)/len(noise_ext)
			k = np.sqrt(P_n/P_s)*10**(SNR/20.)  # amplitude adjustment factor
			s_noisy = s + noise_ext/k
			s_noisy = s_noisy.astype('float32')

			fp.create_dataset("n{:03d}".format(noise_count), data=s_noisy)
			noise_count += 1

	count += 1

# Testing data--one of the same utterances from female speaker from DR2,
# one of the same utterances from female speaker from DR1 and DR3
# All with testing noise types
testing_data_files = [path+'originals/train/f2_SA1.WAV',
					  path+'originals/train/f1_SA1.WAV',
					  path+'originals/train/f3_SA1.WAV']

count = 0
for filepath in testing_data_files:
	with wave.open(filepath, mode='rb') as fp:
		raw_data = fp.readframes(fp.getnframes())
		s = np.array(struct.unpack('<'+str(fp.getnframes())+'h', raw_data),
					 dtype='float32')
		# Normalize to -1, 1
		s = s/np.max(s)
		# Skip first and last 100 ms
		cutoff = int(SAMPLE_RATE*.1)
		s = s[cutoff:-cutoff]

	with h5py.File(path+'testing_data_{}.hdf5'.format(count), 'w') as fp:
		fp.create_dataset("clean", data=s)

		# Add 10 types of non-stationary noise and save each one
		noise_count = 91
		P_s = np.sum(np.abs(s)**2)/len(s)  # signal power
		for noise in test_noise:
			if len(noise) >= len(s):
				noise_ext = noise[:len(s)]
			else:
				num_repeats = int(np.ceil(len(s)/len(noise)))
				noise_ext = noise
				for n in range(num_repeats):
					noise_ext = np.concatenate((noise_ext, noise))
				noise_ext = noise_ext[:len(s)]

			# Adjust SNR
			P_n = np.sum(np.abs(noise_ext)**2)/len(noise_ext)
			k = np.sqrt(P_n/P_s)*10**(SNR/20.)  # amplitude adjustment factor
			s_noisy = s + noise_ext/k
			s_noisy = s_noisy.astype('float32')

			fp.create_dataset("n{:03d}".format(noise_count), data=s_noisy)
			noise_count += 1

		# Add stationary noise (WGN, 0 mean, unit variance)
		white_noise = np.random.normal(0,1,len(s))
		# Normalize to -1, 1
		white_noise = white_noise/np.max(white_noise)

		P_n = np.sum(np.abs(white_noise)**2)/len(white_noise)
		k = np.sqrt(P_n/P_s)*10**(SNR/20.)
		s_noisy = s + white_noise/k
		s_noisy = s_noisy.astype('float32')

		fp.create_dataset("WGN", data=s_noisy)

	count += 1
