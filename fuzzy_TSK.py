import skfuzzy as fuzz
import numpy as np

##### INPUTS #####
pitch_mean = np.arange(0, 10, .1)

##### Membership functions #####
low_pitch_mean = fuzz.gaussmf(pitch_mean, 0, 1.5)

mf = fuzz.interp_membership(pitch_mean,low_pitch_mean,4)

print (str(mf))
