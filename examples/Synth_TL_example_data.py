import numpy as np
import pyfar as pf
import h5py
import matplotlib.pyplot as plt
from scipy import fft
import cmath


fs = 51200
c = 20.047 * np.sqrt(273.15 + 20) # speed of sound at 20 °C


saveMat = True
savePlot = True

signal = 'sine'
  
m1 = cmath.rect(1,0)
m2 = cmath.rect(1,(-1/2)*np.pi)
m3 = cmath.rect(1,(-1/2)*np.pi)
m4 = cmath.rect(1,(-1/2)*np.pi)#(1/2)*np.pi)
m5 = cmath.rect(1,(-1/2)*np.pi)
m6 = cmath.rect(1,(-1/2)*np.pi)

# setting 1 for direct, 2 for mic1-2 switch, 3 for mic1-3 switch, 
# 4 for mic1-4 switch, 5 for mic1-5 switch, 6 for mic1-6 switch

mics_switch = [[m1,m2,m3,m4,m5,m6],[m2,m1,m3,m4,m5,m6],[m3,m2,m1,m4,m5,m6],[m4,m2,m3,m1,m5,m6],[m5,m2,m3,m4,m1,m6],[m6,m2,m3,m4,m5,m1]]
#mics_switch = [[m1,m2,m3,m4,m5,m6]] #for only direct configuration
setting = 1
samples = fs*1

start = 0
stop = fs-1

def shift_err(signal, mic_err, noise):

    signal  = pf.dsp.time_window(signal=signal,  window='boxcar', interval=[start,stop], crop='window')
    noise  = pf.dsp.time_window(signal=noise,  window='boxcar', interval=[start,stop], crop='window')

    signal_fft = np.fft.rfft(signal.time)
    signal_fft = signal_fft*mic_err
    signal_shifted = np.fft.irfft(signal_fft)

    signal_shifted = signal_shifted + noise.time

    return signal_shifted


for i, mics in enumerate(mics_switch):

    if signal == 'sweep':

        amp = 'Sweep Amp = 1'
        h5name = 'TL_Synth_Sweep_tr01'
        plotname = 'TL_Synth_Sweep_tr01_signal'
        transmission_factor = 0.1
        reflection_factor = 0.0
        #absorbtion_factor = 0.05
        input = pf.signals.exponential_sweep_time(frequency_range=[1,fs/2], n_samples=samples, amplitude=1, sampling_rate=fs)
        noise = pf.signals.noise(n_samples=samples, spectrum='white',rms=0.00, sampling_rate=fs)

    if signal == 'sine':
        amp = 'Sine Amp = 1'
        h5name = 'TL_Synth_noisySine_tr05_loopTest_set1' + str(i+1)
        plotname = 'TL_Synth_noisySine_tr05_loopTest_signal_set1' + str(i+1)
        transmission_factor = 0.5
        reflection_factor = 0.0
        #absorbtion_factor = 0.05
        input = pf.signals.sine(frequency=1000, n_samples=samples, amplitude=1, sampling_rate=fs)
        noise = pf.signals.noise(n_samples=samples, spectrum='white',rms=0.02, sampling_rate=fs)


    if signal == 'noise':
        amp = 'White Noise RMS = 1'
        h5name = 'TL_Synth_wNoise_tr05cancel_set' + str(i+1)
        plotname = 'TL_Synth_wNoise_tr05cancel_signal_set' + str(i+1)
        transmission_factor = 0.5
        reflection_factor = 0.0
        input = pf.signals.noise(n_samples=samples, spectrum='white',rms=1, sampling_rate=fs)
        noise = pf.signals.noise(n_samples=samples, spectrum='white',rms=0, sampling_rate=fs)


    if signal == 'dirac':
        amp = 'Dirac Impulse Amp = 1'
        h5name = 'TL_Synth_noisyDirac_tr05'
        plotname = 'TL_Synth_noisyDirac_tr05_signal'
        transmission_factor = 0.5
        reflection_factor = 0.0
        input = pf.signals.impulse(n_samples=samples, sampling_rate=fs, amplitude=1, delay=0)
        noise = pf.signals.noise(n_samples=samples, spectrum='white',rms=0.02, sampling_rate=fs)



    #input signal
    # delay from input to position 1
    s_i1 = 0.3  # disance from input to mic 1
    dt = s_i1 / c
    dN = int(dt * fs)
    dirac_i_pos1 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)

    # incident signal at position 1
    noise_pos1 = pf.dsp.convolve(signal1=noise, signal2=dirac_i_pos1)
    pos1_incident = pf.dsp.convolve(signal1=input, signal2=dirac_i_pos1)


    # delay input from position 1 to position 2
    s_12 = 0.5 - 0.085
    dt = s_12 / c
    dN = int(dt * fs)
    dirac_pos1_pos2 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)

    # incident signal at position 2
    noise_pos2 = pf.dsp.convolve(signal1=noise_pos1, signal2=dirac_pos1_pos2)
    pos2_incident = pf.dsp.convolve(signal1=pos1_incident, signal2=dirac_pos1_pos2)

    # incident signal at position 3
    s_23 = 0.085 # distance pos2-pos3
    dt = s_23 / c
    dN = int(dt * fs)

    dirac_pos2_pos3 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    noise_pos3 = pf.dsp.convolve(signal1=noise_pos2, signal2=dirac_pos2_pos3)
    pos3_incident = pf.dsp.convolve(signal1=pos2_incident, signal2=dirac_pos2_pos3)

    # x1 = 0.7+0.1+0.25 # position 1 to basotect
    s_3B = 0.2 + 0.1 + 0.225
    dt = (s_3B) / c # time delay position 3 to basotect
    dN = int(dt * fs)
    dirac_pos3_basotect = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    basotect = pf.dsp.convolve(signal1=pos3_incident, signal2=dirac_pos3_basotect)
    noise_basotect = pf.dsp.convolve(signal1=noise_pos3, signal2=dirac_pos3_basotect)



    # to synthesize the signals after the middle canal

    #transmission_factor = np.round(1 - reflection_factor - absorbtion_factor,2)
    #s_iB = s_i1+s_12+s_23+s_3B
    #dt = s_iB/c
    #dN = int(dt*fs)-2 # sample correction due to different summation of distances
    #dirac_posi_basotect = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    #delayedSine = pf.dsp.convolve(signal1=0.4*inputSine, signal2=dirac_posi_basotect)
    basotect_transmissioned = transmission_factor * basotect #+ delayedSine

    # incident signal at position 4
    s_B4 = 0.275 + 0.1 + 0.2
    dt = (s_B4) / c # time delay position 3 to basotect
    dN = int(dt * fs)
    dirac_basotect_pos4 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    pos4_incident = pf.dsp.convolve(signal1=basotect_transmissioned, signal2=dirac_basotect_pos4)
    noise_pos4 = pf.dsp.convolve(signal1=noise_basotect, signal2=dirac_basotect_pos4)

    # incident signal at position 5
    s_45 = 0.085
    dt = (s_45) / c # time delay position 3 to basotect
    dN = int(dt * fs)
    dirac_pos4_pos5 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    pos5_incident = pf.dsp.convolve(signal1=pos4_incident, signal2=dirac_pos4_pos5)
    noise_pos5 = pf.dsp.convolve(signal1=noise_pos4, signal2=dirac_pos4_pos5)

    # incident signal at position 6
    s_56 = 0.415
    dt = (s_56) / c # time delay position 3 to basotect
    dN = int(dt * fs)
    dirac_pos5_pos6 = pf.signals.impulse(n_samples=dN+1, delay=dN, sampling_rate=fs)
    pos6_incident = pf.dsp.convolve(signal1=pos5_incident, signal2=dirac_pos5_pos6)
    noise_pos6 = pf.dsp.convolve(signal1=noise_pos5, signal2=dirac_pos5_pos6)


    signal_pos1 = shift_err(pos1_incident,mics[0],noise_pos1)
    signal_pos2 = shift_err(pos2_incident,mics[1],noise_pos2)
    signal_pos3 = shift_err(pos3_incident,mics[2],noise_pos3)
    signal_pos4 = shift_err(pos4_incident,mics[3],noise_pos4)
    signal_pos5 = shift_err(pos5_incident,mics[4],noise_pos5)
    signal_pos6 = shift_err(pos6_incident,mics[5],noise_pos6)



    # TODO: mix signals to one object with 6 channels (acoular excpects array of shape [N_samples x N_channels])
    time_data = np.concatenate((signal_pos1, signal_pos2, signal_pos3, signal_pos4, signal_pos5, signal_pos6), axis=0).T
    time_data = time_data.astype('float32')


    freq_data = []


    for i in range(0,np.shape(time_data)[1]):

        freq_data.append(fft.fft(time_data.T[i]))
        
    mag = np.abs(freq_data)/(samples/2)
    phase = np.angle(freq_data)


    fig = plt.subplot(311)
    fig.set_title(amp + ' and TR = ' + str(transmission_factor))
    ax1 = plt.subplot(3,1,1)
    for i in range(0,np.shape(time_data)[1]):
        ax1.plot(time_data.T[i])
        ax1.legend(['Mic1, err='+str(np.around(mics[0],2)), 'Mic2, err='+str(np.around(mics[1],2)), 'Mic3, err='+str(np.around(mics[2],2)), 'Mic4, err='+str(np.around(mics[3],2)), 'Mic5, err='+str(np.around(mics[4],2)), 'Mic6, err='+str(np.around(mics[5],2))])
    ax1.set_xlabel('samples')
    ax1.set_xlim([0,fs/80])

    ax2 = plt.subplot(3,1,2)
    for i in range(0,np.shape(mag)[0]):
        ax2.plot(mag[i])
        #ax2.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('f in Hz')
    ax2.set_xlim([0,fs/20])
    ax2.set_ylim([0,np.max(mag)+0.1])
    ax3 = plt.subplot(3,1,3)
    for i in range(0,np.shape(phase)[0]):
        ax3.plot(phase[i])
        #ax3.legend(['Mic1', 'Mic2', 'Mic3', 'Mic4', 'Mic5', 'Mic6'])
    ax3.set_xlim([0,fs/20])
    ax3.set_ylabel('Angle')
    ax3.set_xlabel('f in Hz')


    fig.figure.tight_layout()

    plt.show()

    if savePlot:
        filename_plot = plotname+'.pdf'
        fig.figure.savefig('./Plots/'+filename_plot)

    # Write to file
    if saveMat:
        hf = h5py.File('./Resources/'+h5name+'.h5','w')
        hf.create_dataset('time_data',data=time_data)
        hf['time_data'].attrs['sample_freq'] = fs
        hf.close()

