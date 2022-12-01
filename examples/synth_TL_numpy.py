import numpy as np
import pyfar as pf
import h5py
from os.path import join, isdir
import matplotlib.pyplot as plt
from scipy import fft

h5name = 'TL_Synth_np_amp1_tr08'
reflection_factor = 0.0
absorbtion_factor = 0.9


#TODO: add position 3 for a new 

fs = 51200
c = 20.047 * np.sqrt(273.15 + 20) # speed of sound at 20 °C

#input signal
# input = pf.signals.noise(n_samples=4*fs, spectrum='white',rms=0.1, sampling_rate=fs)
input = np.zeros(51200)
input[0] = 1
# delay from input to position 1
s_i1 = 0.3  # disance from input to mic 1
dt = s_i1 / c
dN = np.zeros(int(dt * fs))
pos1_incident = np.concatenate((dN,input[0:-len(dN)]))


# delay input from position 1 to position 2
s_12 = 0.5 - 0.085
dt = s_12 / c
dN = np.zeros(int(dt * fs))
pos2_incident = np.concatenate((dN,pos1_incident[0:-len(dN)]))

# incident signal at position 3
s_23 = 0.085 # distance pos2-pos3
dt = s_23 / c
dN = np.zeros(int(dt * fs))
pos3_incident = np.concatenate((dN, pos2_incident[0:-len(dN)]))

# x1 = 0.7+0.1+0.25 # position 1 to basotect
s_3B = 0.2 + 0.1 + 0.225
dt = (s_3B) / c # time delay position 3 to basotect
dN = np.zeros(int(dt * fs))
basotect = np.concatenate((dN,pos3_incident[0:-len(dN)]))


# to synthesize the signals after the middle canal
transmission_factor = np.round(1 - reflection_factor - absorbtion_factor,2)

basotect_transmissioned = transmission_factor * basotect

# incident signal at position 4
s_B4 = 0.275 + 0.1 + 0.2
dt = (s_B4) / c # time delay position 3 to basotect
dN = np.zeros(int(dt * fs))
pos4_incident = np.concatenate((dN,basotect_transmissioned[0:-len(dN)]))

# incident signal at position 5
s_45 = 0.085
dt = (s_45) / c # time delay position 3 to basotect
dN = np.zeros(int(dt * fs))
pos5_incident = np.concatenate((dN,pos4_incident[0:-len(dN)]))

# incident signal at position 6
s_56 = 0.415
dt = (s_56) / c # time delay position 3 to basotect
dN = np.zeros(int(dt * fs))
pos6_incident = np.concatenate((dN,pos5_incident[0:-len(dN)]))



signal_pos1 = pos1_incident #+ pos1_reflected 
signal_pos2 = pos2_incident #+ pos2_reflected 
signal_pos3 = pos3_incident #+ pos3_reflected
signal_pos4 = pos4_incident
signal_pos5 = pos5_incident
signal_pos6 = pos6_incident



# TODO: mix signals to one object with 6 channels (acoular excpects array of shape [N_samples x N_channels])
time_data = [signal_pos1, signal_pos2, signal_pos3, signal_pos4, signal_pos5, signal_pos6]
#np.concatenate((signal_pos1, signal_pos2, signal_pos3, signal_pos4, signal_pos5, signal_pos6), axis=1)
#time_data = time_data.astype('float32')
# time_data = pf.Signal(data=data, sampling_rate=fs, domain='time')

'''
freq_input = fft.fft(input.time)
mag_input = np.abs(freq_input)
phase_input = np.angle(freq_input)

fig = plt.subplot(311)

ax1 = plt.subplot(3,1,1)
ax1.plot(input.time)
plt.xlim([0,1000])

ax2 = plt.subplot(3,1,2)
ax2.plot(mag_input)
plt.xlim([0,1000])
plt.ylim([-2,2])
ax3 = plt.subplot(3,1,3)
ax3.plot(phase_input)
plt.xlim([0,1000])

plt.show()
'''
#'''
freq_data = []


for i in range(0,np.shape(time_data)[0]):

    freq_data.append(fft.fft(time_data[i]))
    
mag = np.abs(freq_data)
phase = np.angle(freq_data)

#freq_data = fft.fft(time_data.T[0])
#mag = np.abs(freq_data)
#phase = np.angle(freq_data)

g13 = freq_data[2]/freq_data[0]
g46 = freq_data[5]/freq_data[1]
g = g13
mag_g = np.abs(g)
phase_g = np.angle(g)


fig = plt.subplot(311)

ax1 = plt.subplot(3,1,1)
for i in range(0,np.shape(time_data)[0]):
    ax1.plot(time_data[i])
    ax1.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])

ax2 = plt.subplot(3,1,2)
ax2.plot(mag_g)
    #ax2.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])
plt.ylim([0,1.1])
ax3 = plt.subplot(3,1,3)
ax3.plot(phase_g)
    #ax3.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])


plt.show()

'''
fig = plt.subplot(311)

ax1 = plt.subplot(3,1,1)
for i in range(0,np.shape(time_data)[0]):
    ax1.plot(time_data[i])
    ax1.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])

ax2 = plt.subplot(3,1,2)
for i in range(0,np.shape(mag)[0]):
    ax2.plot(mag[i])
    #ax2.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])
plt.ylim([0,1.1])
ax3 = plt.subplot(3,1,3)
for i in range(0,np.shape(phase)[0]):
    ax3.plot(phase[i])
    #ax3.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
plt.xlim([0,1000])


plt.show()
#'''
# Write to file
hf = h5py.File('./Resources/'+h5name+'.h5','w')
hf.create_dataset('time_data',data=time_data)
hf['time_data'].attrs['sample_freq'] = fs
hf.close()

