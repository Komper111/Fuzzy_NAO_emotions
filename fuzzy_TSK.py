import sys
import time
import copy
import skfuzzy as fuzz
import numpy as np
from enum import Enum
from naoqi import ALProxy
from aubio import fvec, source, pvoc, filterbank, pitch, freqtomidi
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
inputUniverses = [None] * 9
inputUniverses[InVariables.pitch_mean.value] = np.arange(0, 2000, .1)
inputUniverses[InVariables.pitch_variance.value] = np.arange(0, 40000000, .1))
inputUniverses[InVariables.pitch_maximum.value] = np.arange(0, 15000, .1))
inputUniverses[InVariables.pitch_minimum.value] = np.arange(0, 100, .1))
inputUniverses[InVariables.pitch_range.value] = np.arange(0, 15000, .1))
inputUniverses[InVariables.energy_mean.value] = np.arange(0, 1, .0001))
inputUniverses[InVariables.energy_variance.value] = np.arange(0, 1, .00001))
inputUniverses[InVariables.energy_maximum.value] = np.arange(0, 5, .1))
inputUniverses[InVariables.energy_range.value] = np.arange(0, 5, .1))

##### Membership functions #####
### membership functions of pitch mean ###
memb_func = [[None] * 8] * 9
memb_func[InVariables.pitch_mean.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 0, 2)
memb_func[InVariables.pitch_mean.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
memb_func[InVariables.pitch_mean.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_mean.value], 5, 5)
### membership functions of pitch variance ###
memb_func[InVariables.pitch_variance.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 0, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 5, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 0, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 5, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 0, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 5, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 0, 5)
memb_func[InVariables.pitch_variance.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_variance.value], 5, 5)
### membership functions of pitch maximum ###
memb_func[InVariables.pitch_maximum.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
memb_func[InVariables.pitch_maximum.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_maximum.value], 0, 2)
### membership functions of pitch minimum ###
memb_func[InVariables.pitch_minimum.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
memb_func[InVariables.pitch_minimum.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_minimum.value], 0, 2)
### membership functions of pitch range ###
memb_func[InVariables.pitch_range.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
memb_func[InVariables.pitch_range.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.pitch_range.value], 0, 2)
### membership functions of energy mean ###
memb_func[InVariables.energy_mean.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
memb_func[InVariables.energy_mean.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_mean.value], 0, 2)
### membership functions of energy variance ###
memb_func[InVariables.energy_variance.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
memb_func[InVariables.energy_variance.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_variance.value], 0, 2)
### membership functions of energy maximum ###
memb_func[InVariables.energy_maximum.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
memb_func[InVariables.energy_maximum.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_maximum.value], 0, 2)
### membership functions of energy range ###
memb_func[InVariables.energy_range.value][BasicEmotions.joy.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.trust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.fear.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.surprise.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.sadness.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.disgust.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.anger.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)
memb_func[InVariables.energy_range.value][BasicEmotions.anticipation.value] = fuzz.gaussmf(inputUniverses[InVariables.energy_range.value], 0, 2)

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
def getPitches(filename, pitch):
	downsample = 1
	samplerate = 44100
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
	return [pitch_mean, pitch_dif_sum / len(pitches), maximum, minimum, pitch_max - pitch_min]

def getEnergies(filename):
	win_s = 512                 # fft size
	hop_s = win_s / 4           # hop size

	samplerate = 44100
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
	return [energy, energy_variance, energy_maximum, energy_range]

#### Category function ####
def get_memberships(inputs):
	categories = [[None] * 8] * 9
	for i in range(0, len(memb_func)):
		for j in range(0, len(memb_func[i])):
			categories[i][j] = fuzz.interp_membership(inputUniverses[i], memb_func[i][j], inputs[i])
	return categories

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
basic_emotions = [0] * 8

#### Main algorithm ####	
while True:
	raw_input("Press ENTER to detect new voice.")
	doc = "recording.wav"
	#nahravaj
	recordVoice(doc, 5)
 	inputs= getPitches(doc, pitch).extend(getEnergies(doc))

	#### Fuzzyfication ####
	memberships = get_memberships(inputs)

	#### Geting emotional output ####
	for i in range(0, len(basic_emotions)):
		for j in range(0, len(memberships)):
			basic_emotions[i] += memberships[j][i]

	makeBehaveior(getMaxEmotion(copy.copy(basic_emotions)))

	

