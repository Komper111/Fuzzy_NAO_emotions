#! /usr/bin/env python

import sys
import numpy
import aubio 
from aubio import fvec, source, pvoc, filterbank, pitch, freqtomidi
from numpy import vstack, zeros

def getPitches(filename, pitch):
	downsample = 1
	samplerate = 16000
	hop_s = 512 / downsample
	win_s = 4096 / downsample
	pitch_sum = 0
	pitch_dif_sum = 0

	pitch_o = pitch("yin", win_s, hop_s, samplerate)
	s = source(filename, samplerate, hop_s)

	pitches = []

	i=0
	minimum = 100
	maximum = 0

	while True:
		
		samples, read = s()
		i = i + 1
		pitch = pitch_o(samples)[0]
		if (pitch < minimum) & (pitch != 0):
			minimum = pitch
			#print i
			#print pitch
		if (pitch > maximum) & (pitch < 20000):
			maximum = pitch
		pitches.append(pitch)

		pitch_sum = pitch_sum + pitch
		if read < hop_s: break

	 
	pitch_mean = pitch_sum / len(pitches)


	for j in pitches:
		pitch_difference = pow((j - pitch_mean), 2)
		pitch_dif_sum = pitch_dif_sum + pitch_difference


	pitch_variance = pitch_dif_sum / len(pitches)
	pitch_max = maximum
	pitch_min = minimum
	pitch_range = pitch_max - pitch_min


	#print "pitch difference summary:",  pitch_dif_sum
	#print "pitch summary:", pitch_sum 
	print "pitch mean:", pitch_mean 
	print "pitch variance:", pitch_variance 
	print "pitch maximum:", pitch_max
	print "pitch minimum:", pitch_min
	print "pitch_range", pitch_range

def getEnergies(filename):
	win_s = 512                 # fft size
	hop_s = win_s / 4           # hop size

	samplerate = 0
	s = source(filename, samplerate, hop_s)
	samplerate = s.samplerate

	pv = pvoc(win_s, hop_s)

	f = filterbank(40, win_s)
	f.set_mel_coeffs_slaney(samplerate)

	energies = zeros((40,))
	o = {}

	total_frames = 0
	downsample = 2

	while True:
	    samples, read = s()
	    fftgrain = pv(samples)
	    new_energies = f(fftgrain)
	    energies = vstack( [energies, new_energies] )
	    total_frames += read
	    if read < hop_s: break

	su = 0
	energy_minimum = 100
	energy_maximum = 0
	avg_energies = []
	for i in range(1,len(energies)):
		for j in range(1, (len(energies[i])-1)):
			e = energies[i][j]
			su += e
			if energy_maximum < e:
				energy_maximum = e
			elif energy_minimum > e:
				energy_minimum = e
		avg_energies.append(su/(len(energies[i])-1))
		su = 0

	su2 = 0
	for e in avg_energies:
		su += e
		su2 += pow(e, 2)
	energy = su/len(avg_energies)
	energy_variance = su2/len(avg_energies)
	energy_range = energy_maximum - energy_minimum

	print "Energy: " + str(energy)
	print "Max energy: " + str(energy_maximum)
	print "Energy range: " + str(energy_range)
	print "Energy variance: " + str(energy_variance)

getPitches(sys.argv[1], pitch)
getEnergies(sys.argv[1])