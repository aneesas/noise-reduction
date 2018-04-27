#!/bin/bash

# This script creates an even number of training and testing utterances
# by taking 1 male and 1 female speaker from each of the 8 dialectic regions
# in the TIMIT database and converting the files from the NIST SPHERE format
# to WAV format using the sph2pipe tool
# (Source: https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools)
# Aneesa Sonawalla
# April 2018

train_path='/Users/aneesasonawalla/Documents/GT/ECE6255/Project/TIMIT/TRAIN/'
train_dest='/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/originals/train/'

test_path='/Users/aneesasonawalla/Documents/GT/ECE6255/Project/TIMIT/TEST/'
test_dest='/Users/aneesasonawalla/Documents/GT/ECE6255/Project/source/data/originals/test/'

drs='DR1 DR2 DR3 DR4 DR5 DR6 DR7 DR8'

echo "Converting training data"

count=1

for dr in $drs; do
	cd $train_path$dr

	males=(M*/)
	cd "${males[0]}"
	files=*.WAV
	for file in $files; do
		outname="m${count}_${file}"
		/Users/aneesasonawalla/Documents/GT/ECE6255/Project/sph2pipe_v2.5/sph2pipe -p -f wav $file $train_dest$outname
	done
	cd ..

	females=(F*/)
	cd "${females[0]}"
	files=*.WAV
	for file in $files; do
		outname="f${count}_${file}"
		/Users/aneesasonawalla/Documents/GT/ECE6255/Project/sph2pipe_v2.5/sph2pipe -p -f wav $file $train_dest$outname
	done

	((count++))
done

echo "Converting testing data"

count=1

for dr in $drs; do
	cd $test_path$dr

	males=(M*/)
	cd "${males[0]}"
	files=*.WAV
	for file in $files; do
		outname="m${count}_${file}"
		/Users/aneesasonawalla/Documents/GT/ECE6255/Project/sph2pipe_v2.5/sph2pipe -p -f wav $file $test_dest$outname
	done
	cd ..

	females=(F*/)
	cd "${females[0]}"
	files=*.WAV
	for file in $files; do
		outname="f${count}_${file}"
		/Users/aneesasonawalla/Documents/GT/ECE6255/Project/sph2pipe_v2.5/sph2pipe -p -f wav $file $test_dest$outname
	done

	((count++))
done

echo "All done"