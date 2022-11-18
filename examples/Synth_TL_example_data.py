import numpy as np
import pyfar as pf
import h5py


#TODO: add position 3 for a new 

fs = 51200
c = 20.047 * np.sqrt(273.15 + 20) # speed of sound at 20 °C

#input signal
# input = pf.signals.noise(n_samples=4*fs, spectrum='white',rms=0.1, sampling_rate=fs)
input = pf.signals.impulse(n_samples=4*fs, sampling_rate=fs, amplitude=0.7, delay=0)

# delay from input to position 1
s_i1 = 0.3  # disance from input to mic 1
dt = s_i1 / c
dN = int(dt * fs)
dirac_i_pos1 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)

# incident signal at position 1
pos1_incident = pf.dsp.convolve(signal1=input, signal2=dirac_i_pos1)


# delay input from position 1 to position 2
s_12 = 0.5 - 0.085
dt = s_12 / c
dN = int(dt * fs)
dirac_pos1_pos2 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)

# incident signal at position 2
pos2_incident = pf.dsp.convolve(signal1=pos1_incident, signal2=dirac_pos1_pos2)

# incident signal at position 3
s_23 = 0.085 # distance pos2-pos3
dt = s_23 / c
dN = int(dt * fs)
dirac_pos2_pos3 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
pos3_incident = pf.dsp.convolve(signal1=pos2_incident, signal2=dirac_pos2_pos3)

# x1 = 0.7+0.1+0.25 # position 1 to basotect
s_3B = 0.2 + 0.1 + 0.225
dt = (s_3B) / c # time delay position 3 to basotect
dN = int(dt * fs)
dirac_pos3_basotect = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
basotect = pf.dsp.convolve(signal1=pos3_incident, signal2=dirac_pos3_basotect)


# uniform spectrum 0.1 gain
reflection_factor = 0.0
basotect_reflected = reflection_factor * basotect

# reflected signal at position 3
pos3_reflected = pf.dsp.convolve(signal1=basotect_reflected, signal2=dirac_pos3_basotect)

# reflected signal at position 2
pos2_reflected = pf.dsp.convolve(signal1=pos3_reflected, signal2=dirac_pos2_pos3)

# reflected signal at position 1
pos1_reflected = pf.dsp.convolve(signal1=pos2_reflected, signal2=dirac_pos1_pos2)


# to synthesize the signals after the middle canal

absorbtion_factor = 0.0
transmission_factor = np.round(1 - reflection_factor - absorbtion_factor,2)

basotect_transmissioned = transmission_factor * basotect

# incident signal at position 4
s_B4 = 0.275 + 0.1 + 0.2
dt = (s_B4) / c # time delay position 3 to basotect
dN = int(dt * fs)
dirac_basotect_pos4 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
pos4_incident = pf.dsp.convolve(signal1=basotect_transmissioned, signal2=dirac_basotect_pos4)

# incident signal at position 5
s_45 = 0.085
dt = (s_45) / c # time delay position 3 to basotect
dN = int(dt * fs)
dirac_pos4_pos5 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
pos5_incident = pf.dsp.convolve(signal1=pos4_incident, signal2=dirac_pos4_pos5)

# incident signal at position 6
s_56 = 0.415
dt = (s_56) / c # time delay position 3 to basotect
dN = int(dt * fs)
dirac_pos5_pos6 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
pos6_incident = pf.dsp.convolve(signal1=pos5_incident, signal2=dirac_pos5_pos6)



# crop signals and add incident and reflected signal
start = 0
stop = fs-1
pos1_incident  = pf.dsp.time_window(signal=pos1_incident,  window='boxcar', interval=[start,stop], crop='window')
pos1_reflected = pf.dsp.time_window(signal=pos1_reflected, window='boxcar', interval=[start,stop], crop='window')
pos2_incident  = pf.dsp.time_window(signal=pos2_incident,  window='boxcar', interval=[start,stop], crop='window')
pos2_reflected = pf.dsp.time_window(signal=pos2_reflected, window='boxcar', interval=[start,stop], crop='window')
pos3_incident  = pf.dsp.time_window(signal=pos3_incident,  window='boxcar', interval=[start,stop], crop='window')
pos3_reflected = pf.dsp.time_window(signal=pos3_reflected, window='boxcar', interval=[start,stop], crop='window')

pos4_incident = pf.dsp.time_window(signal=pos4_incident,  window='boxcar', interval=[start,stop], crop='window')
pos5_incident = pf.dsp.time_window(signal=pos5_incident,  window='boxcar', interval=[start,stop], crop='window')
pos6_incident = pf.dsp.time_window(signal=pos6_incident,  window='boxcar', interval=[start,stop], crop='window')


signal_pos1 = pos1_incident #+ pos1_reflected 
signal_pos2 = pos2_incident #+ pos2_reflected 
signal_pos3 = pos3_incident #+ pos3_reflected
signal_pos4 = pos4_incident
signal_pos5 = pos5_incident
signal_pos6 = pos6_incident

# TODO: mix signals to one object with 6 channels (acoular excpects array of shape [N_samples x N_channels])
time_data = np.concatenate((signal_pos1.time, signal_pos2.time, signal_pos3.time, signal_pos4.time, signal_pos5.time, signal_pos6.time), axis=0).T
time_data = time_data.astype('float32')
# time_data = pf.Signal(data=data, sampling_rate=fs, domain='time')

# Write to file
hf = h5py.File('./Resources/example_TL_synth.h5','w')
hf.create_dataset('time_data',data=time_data)
hf['time_data'].attrs['sample_freq'] = fs
hf.close()

