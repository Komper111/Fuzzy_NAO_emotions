import sys
import time
import copy
import skfuzzy as fuzz
import numpy as np
from enum import Enum
from naoqi import ALProxy
from aubio import fvec, source, pvoc, filterbank
from numpy import vstack, zeros
import pyaudio
import wave

#### CONSTANTS ####
NAO_IP = "147.232.24.133" 
emotion_combination = [[None, 0, 1, 2, 23, 16, 18, 20],[0, None, 3, 4, 5, 23, 19, 21], [1, 3, None, 6, 7, 8, 23, 22], [2, 4, 6, None, 9, 24, 10, 23], [23, 5, 7, 9, None, 11, 12, 13], [16, 23, 8, 24, 11, None, 14, 15], [18, 19, 23, 10, 12, 14, None, 17], [20, 21, 22, 23, 13, 15, 17, None]]

#### Enumerations ####
class InVariables(Enum):
	pitch_mean = 0
	pitch_variance = 1
	pitch_maximum = 2
	pitch_minimum = 3
	pitch_range = 4
	energy_mean = 5
	energy_variance = 6
	energy_maximum = 7
	energy_range = 8

class BasicEmotions(Enum):
	joy = 0
	trust = 1
	fear = 2
	surprise = 3
	sadness = 4
	disgust = 5
	anger = 6
	anticipation = 7

class OtherEmotions(Enum):
	love = 0
	guilt = 1
	delight = 2
	submission =3
	curiosity = 4
	sentimentality = 5
	alarm = 6
	despair = 7
	shame = 8
	disappointment = 9
	outrage = 10
	remorse = 11
	envy = 12
	pessimism = 13
	contempt = 14
	cynism = 15
	morbidness = 16
	aggresion = 17
	pride = 18
	dominance = 19
	optimism = 20
	fatalism = 21
	anxiety = 22
	conflicted = 23
	unclasified = 24

#naoMotion = ALProxy("ALBehaviorManager", NAO_IP, 9559)
#naoSpeech = ALProxy("ALTextToSpeech", NAO_IP, 9559)

##### Input universes #####
inputUniverses = []
inputUniverses.append(np.arange(0, 5, .1)) 
inputUniverses.append(np.arange(0, 5, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))
inputUniverses.append(np.arange(0, 10, .1))

##### Membership functions #####
### membership functions of pitch mean ###
low_pitch_mean = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 0, 5)
high_pitch_mean = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
### membership functions of pitch variance ###
low_pitch_variance = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 0, 5)
high_pitch_variance = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 5, 5)
### membership functions of pitch maximum ###
low_pitch_maximum = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
high_pitch_maximum = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
### membership functions of pitch minimum ###
low_pitch_minimum = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
high_pitch_minimum = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
### membership functions of pitch range ###
low_pitch_range = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
high_pitch_range = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
### membership functions of energy mean ###
low_energy_mean = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
high_energy_mean = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
### membership functions of pitch variance ###
low_energy_variance = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
high_energy_variance = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
### membership functions of pitch maximum ###
low_energy_maximum = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
high_energy_maximum = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
### membership functions of pitch range ###
low_energy_range = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
high_energy_range = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)

#### Sound detection ####
def recordVoice(NAME, SECONDS):
	CHUNK = 512 
	FORMAT = pyaudio.paInt16 #paInt8
	CHANNELS = 1 
	RATE = 44100 #sample rate
	RECORD_SECONDS = SECONDS
	WAVE_OUTPUT_FILENAME = NAME

	p = pyaudio.PyAudio()

	stream = p.open(format=FORMAT,
	                channels=CHANNELS,
	                rate=RATE,
	                input=True,
	                frames_per_buffer=CHUNK) #buffer

	print("* recording")

	frames = []

	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	    data = stream.read(CHUNK)
	    frames.append(data) # 2 bytes(16 bits) per channel

	print("* done recording")

	stream.stop_stream()
	stream.close()
	p.terminate()

	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()

#### Sound analyze functions ####
def getPitches(filename):
	pass

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
	#print "Energy: " + str(energy)
	#print "Max energy: " + str(energy_maximum)
	#print "Energy range: " + str(energy_range)
	#print "Energy variance: " + str(energy_variance)
	return dict(e = energy, e_varience = energy_variance, e_maximum = energy_maximum, e_range = energy_range)

#### Category functions ####
def pitch_mean_category(pitch_mean_in):
    pitch_mean_cat_low = fuzz.interp_membership(inputUniverses[InVariables.pitch_mean.value],low_pitch_mean,pitch_mean_in)
    pitch_mean_cat_high = fuzz.interp_membership(inputUniverses[InVariables.pitch_mean.value],high_pitch_mean,pitch_mean_in) 
    return dict(low = pitch_mean_cat_low,high = pitch_mean_cat_high)

def pitch_variance_category(pitch_variance_in):
    pitch_variance_cat_low = fuzz.interp_membership(inputUniverses[InVariables.pitch_variance.value], low_pitch_variance, pitch_variance_in)
    pitch_variance_cat_high = fuzz.interp_membership(inputUniverses[InVariables.pitch_variance.value], high_pitch_variance, pitch_variance_in) 
    return dict(low = pitch_variance_cat_low,high = pitch_variance_cat_high)

def pitch_maximum_category(pitch_maximum_in):
    pitch_maximum_cat_low = fuzz.interp_membership(inputUniverses[InVariables.pitch_maximum.value], low_pitch_maximum, pitch_maximum_in)
    pitch_maximum_cat_high = fuzz.interp_membership(inputUniverses[InVariables.pitch_maximum.value], high_pitch_maximum, pitch_maximum_in) 
    return dict(low = pitch_maximum_cat_low,high = pitch_maximum_cat_high)

def pitch_minimum_category(pitch_minimum_in):
    pitch_minimum_cat_low = fuzz.interp_membership(inputUniverses[InVariables.pitch_minimum.value], low_pitch_minimum, pitch_minimum_in)
    pitch_minimum_cat_high = fuzz.interp_membership(inputUniverses[InVariables.pitch_minimum.value], high_pitch_minimum, pitch_minimum_in) 
    return dict(low = pitch_minimum_cat_low,high = pitch_minimum_cat_high)

def pitch_range_category(pitch_range_in):
    pitch_range_cat_low = fuzz.interp_membership(inputUniverses[InVariables.pitch_range.value], low_pitch_range, pitch_range_in)
    pitch_range_cat_high = fuzz.interp_membership(inputUniverses[InVariables.pitch_range.value], high_pitch_range, pitch_range_in) 
    return dict(low = pitch_range_cat_low,high = pitch_range_cat_high)

def energy_mean_category(energy_mean_in):
    energy_mean_cat_low = fuzz.interp_membership(inputUniverses[InVariables.energy_mean.value], low_energy_mean, energy_mean_in)
    energy_mean_cat_high = fuzz.interp_membership(inputUniverses[InVariables.energy_mean.value], high_energy_mean, energy_mean_in) 
    return dict(low = energy_mean_cat_low,high = energy_mean_cat_high)

def energy_variance_category(energy_variance_in):
    energy_variance_cat_low = fuzz.interp_membership(inputUniverses[InVariables.energy_variance.value], low_energy_variance, energy_variance_in)
    energy_variance_cat_high = fuzz.interp_membership(inputUniverses[InVariables.energy_variance.value], high_energy_variance, energy_variance_in) 
    return dict(low = energy_variance_cat_low,high = energy_variance_cat_high)

def energy_maximum_category(energy_maximum_in):
    energy_maximum_cat_low = fuzz.interp_membership(inputUniverses[InVariables.energy_maximum.value], low_energy_maximum, energy_maximum_in)
    energy_maximum_cat_high = fuzz.interp_membership(inputUniverses[InVariables.energy_maximum.value], high_energy_maximum, energy_maximum_in) 
    return dict(low = energy_maximum_cat_low,high = energy_maximum_cat_high)

def energy_range_category(energy_range_in):
    energy_range_cat_low = fuzz.interp_membership(inputUniverses[InVariables.energy_range.value], low_energy_range, energy_range_in)
    energy_range_cat_high = fuzz.interp_membership(inputUniverses[InVariables.energy_range.value], high_energy_range, energy_range_in) 
    return dict(low = energy_range_cat_low,high = energy_range_cat_high)

#### other functions ####
def getMaxEmotion(emotions):	
	treshold = .1
	max1 = 0
	max2 = 0
	idMax1 = 0
	idMax2 = 0
	i = 0
	for EM in emotions:
		if(max1 < EM):
			max2 = max1
			idMax2 = idMax1
			max1 = EM
			idMax1 = i
		elif (max2 < EM):
			max2 = EM
			idMax2 = i
		i = i+1
	if (abs(max1-max2) < treshold):
		return OtherEmotions(emotion_combination[idMax1][idMax2])
	else:
		return BasicEmotions(idMax1)

def makeBehaveior(emotion):
	if(type(emotion) == BasicEmotions):
		naoMotion.runBehavior(emotion.name)	
		print "[NAO]: Making " + emotion.name + " behaveior." 
	else:
		s = "I feel " + emotion.name
		print "[NAO]: Saying " + emotion.name + " behaveior." 
		naoSpeech.say(s)

#### Basic emotions values ####
basic_emotions = [0,0,0,0,0,0,0,0]
	

while True:
	#nahravaj
	recordVoice("recording.wav", 5)

	pitches = getPitches("recording.wav")
	energies = getEnergies("recording.wav")

	#### Fuzzyfication ####
	pitch_mean_in = pitch_mean_category(pitches['p'])
	pitch_variance_in = pitch_variance_category(pitches['p_variance'])
	pitch_maximum_in = pitch_maximum_category(pitches['p_maximum'])
	pitch_minimum_in = pitch_minimum_category(pitches['p_minimum'])
	pitch_range_in = pitch_range_category(pitches['p_range'])
	energy_mean_in = energy_mean_category(energies['e'])
	energy_variance_in = energy_variance_category(energies['e_varience'])
	energy_maximum_in = energy_maximum_category(energies['e_maximum'])
	energy_range_in = energy_range_category(energies['e_range'])

	#### Geting emotional output ####
	basic_emotions[BasicEmotions.joy.value] = (pitch_mean_in['low'] + pitch_variance_in['high']) / 2
	basic_emotions[BasicEmotions.trust.value] = (pitch_mean_in['low'] + pitch_variance_in['high']) / 2
	#basic_emotions[BasicEmotions.fear.value] = (pitch_mean_in['low'] + pitch_variance_in['high']) / 2
	#basic_emotions[BasicEmotions.surprise.value] = (pitch_mean_in['high'] + pitch_variance_in['low']) / 2
	#basic_emotions[BasicEmotions.sadness.value] = (pitch_mean_in['low'] + pitch_variance_in['high']) / 2
	#basic_emotions[BasicEmotions.disust.value] = (pitch_mean_in['high'] + pitch_variance_in['low']) / 2
	#basic_emotions[BasicEmotions.anger.value] = (pitch_mean_in['low'] + pitch_variance_in['high']) / 2
	#basic_emotions[BasicEmotions.anticipation.value] = (pitch_mean_in['high'] + pitch_variance_in['low']) / 2

	makeBehaveior(getMaxEmotion(copy.copy(basic_emotions)))

	raw_input("Press ENTER to detect new voice.")
	

