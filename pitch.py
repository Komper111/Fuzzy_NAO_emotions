import aubio
import sys
from aubio import source, pitch, freqtomidi

filename = sys.argv[1]

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
	pitch = pitch_o.get_confidence()
	print pitch
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

#print pitches


