#! /usr/bin/env python

import sys
import numpy
from aubio import fvec, source, pvoc, filterbank
from numpy import vstack, zeros

win_s = 512                 # fft size
hop_s = win_s / 4           # hop size

if len(sys.argv) < 2:
    print "Usage: %s <filename> [samplerate]" % sys.argv[0]
    sys.exit(1)

filename = sys.argv[1]

samplerate = 16000
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

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
print "Min energy: " + str(energy_minimum)
print "Energy range: " + str(energy_range)
print "Energy variance: " + str(energy_variance)

sys.exit([energy, energy_variance, energy_maximum, energy_range])