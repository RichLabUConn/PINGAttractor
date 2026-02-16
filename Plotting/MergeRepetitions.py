# -*- coding: utf-8 -*-
"""
Spyder Editor

Start: 02/14/23

Basic E-I Network in Python
"""




"""
Imports
"""
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import sys
# import os
import csv
# import time
from scipy.signal import find_peaks
# import seaborn as sns


"""
Parameters
"""
c=1.0               #Capacitance
g_na=24.0           #Sodium conductance
g_kdr=3.0           #Driving potassium conductance
g_l=0.02            #Leak conductance
e_na=55.0           #Sodium reversal potential
e_k=-90.0           #Potassium reversal potential
e_l=-60.0           #Leak reversal potential

g_ks=0.0            #Slow potassium conductance

# esyn_i=-75.0        #GABA reversal potential
esyn_e=0.0          #Glutamate reversal potential
tau_rise=0.2        #Synaptic rise constant
tau_decay_e=3.0     #Synaptic decay constant, excitatory
tau_decay_i=5.5     #Synaptic decya constant, inhibitory

# T=2500              #Length of simulation, ms
T=20000
# dt=0.05             #Time step
dt=.02

numcells_e=800      #Number of excitatory neurons
numcells_i=200      #Number of inhibitory neurons

az=1.0              #z time constant
ah=1.0              #h time constant

pcon_ee=0.3         #E-E connectivity probability
pcon_ei=0.5         #E-I connectivity probability
pcon_ii=0.3         #I-I connectivity probability
pcon_ie=0.5         #I-E connectivity probability

history=10         #Number of past time steps to include

hetero_e=0.1        #Heterogeneity in excitatory inputs
hetero_i=0.05       #Heterogeneity in inhibitory inputs
# Iext_e_mean=1.0          #Mean excitatory input
Iext_i_mean=-0.2         #Mean inhibitory input




"""
Input Variables
"""
if len(sys.argv)==1:
    repetitions=100      #Number of repetitions
    noiseamp=0          #Noise label
    Iext_e_mean=30      #Ie label
else:
    repetitions=int(sys.argv[1])
    noiseamp=int(sys.argv[2])
    Iext_e_mean=int(sys.argv[3])





"""
Functions
"""
# Conductance-based model equations
def minf(v):
    output=(1/(1+np.exp((-v-30.0)/9.5)))
    return output

def hinf(v):
    output=(1/(1+np.exp((v+53.0)/7.0)))
    return output

def th(v):
    output=0.37+(2.78/(1+np.exp((v+40.5)/6.0)))
    return output

def ninf(v):
    output=(1/(1+np.exp((-v-30.0)/10.0)))
    return output

def tn(v):
    output=0.37+(1.85/(1+np.exp((v+27.0)/15.0)))
    return output

def zinf(v):
    output= (1/(1+np.exp((-v-39.0)/5.0)))
    return output

def dvdt(v,z,h,n,Iapp):
    output=(-g_na*(minf(v)**3)*h*(v-e_na)
            -g_kdr*(n**4)*(v-e_k)
            -g_ks*z*(v-e_k)
            -g_l*(v-e_l) 
            + Iapp)/c
    return output

def dzdt(v,z,h,n,Iapp):
    output=(az*(zinf(v)-z))/75.0
    return output
    
def dhdt(v,z,h,n,Iapp):
    output=(ah*(hinf(v)-h))/th(v)
    return output

def dndt(v,z,h,n,Iapp):
    output=(ninf(v)-n)/tn(v)
    return output

# Connectivity matrix equation
def create_conmatrix(numoutput, numinput, p_con, g):
    output=np.zeros((numoutput, numinput))
    for i in range(0,numoutput):
        for j in range(0, numinput):
            if numoutput==numinput:
                if j != i:
                    if np.random.rand() < p_con:
                        output[i,j]=1
            else:
                if np.random.rand() < p_con:
                    output[i,j]=1
    output=output*g
    return output

# Synapse equations
def syn_e(t_spike,i):
    output=(np.exp(-(((i*dt)-t_spike)/tau_decay_e))
            -np.exp(-(((i*dt)-t_spike)/tau_rise)))
    return output

syn_e_vectorize=np.vectorize(syn_e)

def syn_i(t_spike,i):
    output=(np.exp(-(((i*dt)-t_spike)/tau_decay_i))
            -np.exp(-(((i*dt)-t_spike)/tau_rise)))
    return output

syn_i_vectorize=np.vectorize(syn_i)




# RK4 integration
def rk4(v,z,h,n,I):
    k1_v=dvdt(v,z,h,n,I)
    k1_z=dzdt(v,z,h,n,I)
    k1_h=dhdt(v,z,h,n,I)
    k1_n=dndt(v,z,h,n,I)
    
    k2_v=dvdt(v+(dt/2)*k1_v, z+(dt/2)*k1_z, h+(dt/2)*k1_h, n+(dt/2)*k1_n, I)
    k2_z=dzdt(v+(dt/2)*k1_v, z+(dt/2)*k1_z, h+(dt/2)*k1_h, n+(dt/2)*k1_n, I)
    k2_h=dhdt(v+(dt/2)*k1_v, z+(dt/2)*k1_z, h+(dt/2)*k1_h, n+(dt/2)*k1_n, I)
    k2_n=dndt(v+(dt/2)*k1_v, z+(dt/2)*k1_z, h+(dt/2)*k1_h, n+(dt/2)*k1_n, I)

    k3_v=dvdt(v+(dt/2)*k2_v, z+(dt/2)*k2_z, h+(dt/2)*k2_h, n+(dt/2)*k2_n, I)
    k3_z=dzdt(v+(dt/2)*k2_v, z+(dt/2)*k2_z, h+(dt/2)*k2_h, n+(dt/2)*k2_n, I)
    k3_h=dhdt(v+(dt/2)*k2_v, z+(dt/2)*k2_z, h+(dt/2)*k2_h, n+(dt/2)*k2_n, I)
    k3_n=dndt(v+(dt/2)*k2_v, z+(dt/2)*k2_z, h+(dt/2)*k2_h, n+(dt/2)*k2_n, I)

    k4_v=dvdt(v+(dt)*k3_v, z+(dt)*k3_z, h+(dt)*k3_h, n+(dt)*k3_n, I)
    k4_z=dzdt(v+(dt)*k3_v, z+(dt)*k3_z, h+(dt)*k3_h, n+(dt)*k3_n, I)
    k4_h=dhdt(v+(dt)*k3_v, z+(dt)*k3_z, h+(dt)*k3_h, n+(dt)*k3_n, I)
    k4_n=dndt(v+(dt)*k3_v, z+(dt)*k3_z, h+(dt)*k3_h, n+(dt)*k3_n, I)

    out_v=v+(dt/6)*(k1_v+2*k2_v+2*k3_v+k4_v)            
    out_z=z+(dt/6)*(k1_z+2*k2_z+2*k3_z+k4_z)            
    out_h=h+(dt/6)*(k1_h+2*k2_h+2*k3_h+k4_h)            
    out_n=n+(dt/6)*(k1_n+2*k2_n+2*k3_n+k4_n)

    return out_v, out_z, out_h, out_n    

# Detect spikes
def spikedetect(v_2, v_1, v_0):
    if v_2 >0 and v_2<v_1 and v_1>v_0:
        return True
    else:
        return False

# Shifts
def shift(X):
    X=np.roll(X,-1)
    X[:,-1]=0
    return X

# Create random input currents
def sample_inputs(N,Iext,hetero):
    output=Iext+np.random.uniform(-1,1,N)*Iext*hetero
    return output

# Create random initial conditions
def sample_initialconditions(N):
    v=-60+40*np.random.uniform(-1,1,N)
    z=.5+.3*np.random.uniform(-1,1,N)
    h=.5+.3*np.random.uniform(-1,1,N)
    n=.5+.3*np.random.uniform(-1,1,N)
    # z=np.zeros(N)
    # h=np.zeros(N)
    # n=np.zeros(N)
    return v,z,h,n




# Synchrony Measure
def syncmeasure(cellnum, spikes, gauss_width, duration_sec):
    # spikes_e and spikes_i should be the correct format!
    # So no need for first few lines
    
    srate=1000      # Sampling rate (in Hz) to construct voltage signal from spikes
    duration=duration_sec      # Duration of recording (in seconds)
    
    times=np.arange(1000/srate, duration*1000+1, 1000/srate)
    timeseries=np.zeros((int(srate*duration),cellnum))
    
    # First, convert spiketimes to a time series of 1s and 0s sampled at 1000 Hz
    for i in range(cellnum):
        times, timeseries[:,i]=convert_spiketimes(spikes[i],duration,srate)
        
    conv_sig=0*timeseries;
    
    if gauss_width<=0:
        gauss_width=2
        
    # Next, convolve each timeseries with a gaussian of width "gauss_width" (in ms)
    for i in range(cellnum):
        conv_sig[:,i]=conv_gaussian(timeseries[:,i],srate,gauss_width)
        
    G_rescaled, freq, meansig, numpeaks, peaks= golomb_synch(conv_sig, times)   
    
    MFF=0
    for i in range(len(spikes)):
        MFF=MFF+len(spikes[i])
    MFF=MFF/(cellnum*duration)
    
    return G_rescaled, freq, MFF, meansig, numpeaks, peaks
    
def convert_spiketimes(spike_times, duration,srate):
    # this function takes a vector 'spike_times' (which lists spike times in
    # milliseconds), then converts the list of spike times into a time series of
    # ones and zeros, with a one indicating a spike at a certain time. The user
    # specifies the sampling rate (in Hz) and the duration of the resulting
    # signal (in seconds)
    
    times = 1000*np.arange(1/srate,duration+1/srate,1/srate); #time stamps (in ms, not seconds) corresonding to each entry in signal
    signal = np.zeros((int(duration*srate),1)); #number of points in final signal will be the number of seconds multiplied by samples per second
    Npoints = len(signal);
    #use the vector 'spike_times' to insert ones in 'signal' where appropriate
    for i_s in range(len(spike_times)):
        #1000*duration is the duration of the signal in milliseconds;
        #multiply the fraction of the way through the signal the spike
        #occurs by the number of points in the discrete signal, and round
        #to obtain the index of the spike
        if int(np.floor(spike_times[i_s]/(duration*1000)*Npoints))==1000:
            signal[int(np.floor(spike_times[i_s]/(duration*1000)*Npoints))-1] = 1;
        else:
            signal[int(np.floor(spike_times[i_s]/(duration*1000)*Npoints))] = 1;

    return times, signal[:,0]

def conv_gaussian(signal,srate,gauss_width):
    # This function takes a time series sampled at a rate 'srate' (specified in Hz)
    # and convolves it with a Gaussain with sd 'gauss_width (ms)
    
    #require that the Gaussian run from -3 to +3 SDs, and sample the Gaussian
    #according to the specified sampling rate. The number fo points in the 
    #Gaussian is given below
    N_gauss = round(6*(gauss_width/1000)*srate);   #want a standard deviation of 2ms typically

    #Want the number of points in the Gaussian to be odd, so that it is
    #centrally peaked, rather than flat at the top
    if (N_gauss%2)==0:
        N_gauss=N_gauss-1
        
    x=np.linspace(-3,3, N_gauss)
    g=np.exp(-x**2) #note that this does not integrate to 1; instead, the peak is always 1
    
    newsig=np.convolve(signal, g, 'same')
    
    return newsig 


def golomb_synch(sig_mat, times):
    #This function calculates the "Golomb synchrony" of multiple signals
    #(Golomb and Rinzel, 1993/1994). sig_mnat is a matrix of signals in which
    #each COLUMN is a different signal
    
    N = len(sig_mat[:,0]); #number of neurons (or signals) is equal to number of columns in sig_mat
    mean_sig = np.nanmean(sig_mat,axis=1); #calculate the mean signal
    variances = np.var(sig_mat,axis=0); #this is a row vector, where each entry is the variance of one of the columns in sig_mat
    G = np.sqrt(np.var(mean_sig,axis=0)/np.nanmean(variances)); #calculate the actual synchrony measure
    
    #the Golomb measure only goes to zero for asynchronous signals in the limit as the number of
    #neurons goes to infinity; with N neurons, the lowest possible value is
    #1/sqrt(N), so we rescale G according to N so that it goes from 0 to 1
    G_rescaled = (G-1/np.sqrt(N)) / (1-1/np.sqrt(N));
    #with this transformation, G_rescaled can go slightly negative due to
    #randomness in the signals, so we should just set G_rescaled to 0 if it
    #goes negative
    if G_rescaled < 0:
        G_rescaled = 0
        
    if G_rescaled>.5:  
        peaks, _ = find_peaks(mean_sig, .2)
        if len(peaks)>2:
            freq_list=[]
            for i in range(len(peaks)-1):
                if 1000/(times[peaks[i+1]]-times[peaks[i]])<100:
                    freq_list.append(1000/(times[peaks[i+1]]-times[peaks[i]]))
            if len(freq_list)>0:
                freq=sum(freq_list)/len(freq_list)
                numpeaks=len(peaks)
            else:
                freq=0 
                numpeaks=0
        else:
            freq=0 
            numpeaks=0
    elif G_rescaled<=.5 and G_rescaled>.2:
        peaks, _ = find_peaks(mean_sig, .15)
        if len(peaks)>2:
            freq_list=[]
            for i in range(len(peaks)-1):
                if 1000/(times[peaks[i+1]]-times[peaks[i]])<100:
                    freq_list.append(1000/(times[peaks[i+1]]-times[peaks[i]]))
            if len(freq_list)>0:
                freq=sum(freq_list)/len(freq_list)
                numpeaks=len(peaks)
            else:
                freq=0 
                numpeaks=0
        else:
            freq=0 
            numpeaks=0
    else:
        peaks, _ = find_peaks(mean_sig, .1)
        if len(peaks)>2:
            freq_list=[]
            for i in range(len(peaks)-1):
                if 1000/(times[peaks[i+1]]-times[peaks[i]])<100:
                    freq_list.append(1000/(times[peaks[i+1]]-times[peaks[i]]))
            if len(freq_list)>0:
                freq=sum(freq_list)/len(freq_list)
                numpeaks=len(peaks)
            else:
                freq=0 
                numpeaks=0
        else:
            freq=0 
            numpeaks=0


    return G_rescaled, freq, mean_sig, numpeaks, peaks


def processISImeansForSync(spikes, start, end, numcells):
    spikes_end=[]
    for k in range(numcells):
        spikes_end.append([(j-(start)) for j in spikes[k] if (j>start) and (j<end)])
    return spikes_end




#%% Intitialize storage

esyn_i_min=-75.0
esyn_i_max=-65.0
esyn_i_step=1.0
numesyn=len(np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step))

sync_means=np.zeros((repetitions,numesyn))
sync_varOverTime=np.zeros((repetitions,numesyn))
freq_means=np.zeros((repetitions,numesyn))
freq_varOverTime=np.zeros((repetitions,numesyn))
MFF_means=np.zeros((repetitions,numesyn))
MFF_varOverTime=np.zeros((repetitions,numesyn))
percent_means=np.zeros((repetitions,numesyn))
percent_varOverTime=np.zeros((repetitions,numesyn))
sync_onset_num=np.zeros((repetitions,numesyn))
sync_offset_num=np.zeros((repetitions,numesyn))
sync_duration=np.zeros((repetitions,numesyn))
DCtoSpike_corr_e=np.zeros((repetitions,numesyn))
DCtoSpike_corr_i=np.zeros((repetitions,numesyn))
DCtoISImean_corr_e=np.zeros((repetitions,numesyn))
DCtoISImean_corr_i=np.zeros((repetitions,numesyn))
DCtoISIstd_corr_e=np.zeros((repetitions,numesyn))
DCtoISIstd_corr_i=np.zeros((repetitions,numesyn))
ISImeantoISIstd_corr_e=np.zeros((repetitions,numesyn))
ISImeantoISIstd_corr_i=np.zeros((repetitions,numesyn))
linreg_r=np.zeros((repetitions,numesyn))
linreg_slope=np.zeros((repetitions,numesyn))



#%% Import and store data for repetitions
for rep in np.arange(1, repetitions+1, 1):    
    
    print(rep)
    str1='CSV/StatsOverESynI_OU_Trials1_NoiseAmp%d_%d.csv' \
         % (noiseamp, rep)
         
    stats_csv=np.genfromtxt(str1, delimiter=',')
    # stats_csv=stats_csv[-21:]
    
    # Pad with Nan's
    if int(stats_csv[-1][0])!=esyn_i_max:
        for pad in np.arange(stats_csv[-1][0]+1,esyn_i_max+1,1):
            temp=np.empty((1,22))
            temp[0,0]=pad
            temp[0,1:]=np.nan
            stats_csv=np.append(stats_csv, temp, axis=0)

    
    sync_means[rep-1]=stats_csv[:,1]
    sync_varOverTime[rep-1]=stats_csv[:,2]
    freq_means[rep-1]=stats_csv[:,3]
    freq_varOverTime[rep-1]=stats_csv[:,4]
    MFF_means[rep-1]=stats_csv[:,5]
    MFF_varOverTime[rep-1]=stats_csv[:,6]
    percent_means[rep-1]=stats_csv[:,7]
    percent_varOverTime[rep-1]=stats_csv[:,8]
    sync_onset_num[rep-1]=stats_csv[:,9]
    sync_offset_num[rep-1]=stats_csv[:,10]
    sync_duration[rep-1]=stats_csv[:,11]
    DCtoSpike_corr_e[rep-1]=stats_csv[:,12]
    DCtoSpike_corr_i[rep-1]=stats_csv[:,13]
    DCtoISImean_corr_e[rep-1]=stats_csv[:,14]
    DCtoISImean_corr_i[rep-1]=stats_csv[:,15]
    DCtoISIstd_corr_e[rep-1]=stats_csv[:,16]
    DCtoISIstd_corr_i[rep-1]=stats_csv[:,17]
    ISImeantoISIstd_corr_e[rep-1]=stats_csv[:,18]
    ISImeantoISIstd_corr_i[rep-1]=stats_csv[:,19]
    linreg_r[rep-1]=stats_csv[:,20]
    linreg_slope[rep-1]=stats_csv[:,21]

    
    

    
#%% Calculate means and SDs
sync_means_mean=np.nanmean(sync_means,axis=0)
sync_means_std=np.nanstd(sync_means,axis=0)
sync_varOverTime_mean=np.nanmean(sync_varOverTime,axis=0)
sync_varOverTime_std=np.nanstd(sync_varOverTime,axis=0)

freq_means_mean=np.nanmean(freq_means,axis=0)
freq_means_std=np.nanstd(freq_means,axis=0)
freq_varOverTime_mean=np.nanmean(freq_varOverTime,axis=0)
freq_varOverTime_std=np.nanstd(freq_varOverTime,axis=0)

MFF_means_mean=np.nanmean(MFF_means,axis=0)
MFF_means_std=np.nanstd(MFF_means,axis=0)
MFF_varOverTime_mean=np.nanmean(MFF_varOverTime,axis=0)
MFF_varOverTime_std=np.nanstd(MFF_varOverTime,axis=0)

percent_means_mean=np.nanmean(percent_means,axis=0)
percent_means_std=np.nanstd(percent_means,axis=0)
percent_varOverTime_mean=np.nanmean(percent_varOverTime,axis=0)
percent_varOverTime_std=np.nanstd(percent_varOverTime,axis=0)

sync_onset_num_mean=np.nanmean(sync_onset_num,axis=0)
sync_onset_num_std=np.nanstd(sync_onset_num,axis=0)
sync_offset_num_mean=np.nanmean(sync_offset_num,axis=0)
sync_offset_num_std=np.nanstd(sync_offset_num,axis=0)
sync_duration_mean=np.nanmean(sync_duration,axis=0)
sync_duration_std=np.nanstd(sync_duration,axis=0)

DCtoSpike_corr_e_mean=np.nanmean(DCtoSpike_corr_e,axis=0)
DCtoSpike_corr_e_std=np.nanstd(DCtoSpike_corr_e,axis=0)
DCtoSpike_corr_i_mean=np.nanmean(DCtoSpike_corr_i,axis=0)
DCtoSpike_corr_i_std=np.nanstd(DCtoSpike_corr_i,axis=0)

DCtoISImean_corr_e_mean=np.nanmean(DCtoISImean_corr_e,axis=0)
DCtoISImean_corr_e_std=np.nanstd(DCtoISImean_corr_e,axis=0)
DCtoISImean_corr_i_mean=np.nanmean(DCtoISImean_corr_i,axis=0)
DCtoISImean_corr_i_std=np.nanstd(DCtoISImean_corr_i,axis=0)

DCtoISIstd_corr_e_mean=np.nanmean(DCtoISIstd_corr_e,axis=0)
DCtoISIstd_corr_e_std=np.nanstd(DCtoISIstd_corr_e,axis=0)
DCtoISIstd_corr_i_mean=np.nanmean(DCtoISIstd_corr_i,axis=0)
DCtoISIstd_corr_i_std=np.nanstd(DCtoISIstd_corr_i,axis=0)

ISImeantoISIstd_corr_e_mean=np.nanmean(ISImeantoISIstd_corr_e,axis=0)
ISImeantoISIstd_corr_e_std=np.nanstd(ISImeantoISIstd_corr_e,axis=0)
ISImeantoISIstd_corr_i_mean=np.nanmean(ISImeantoISIstd_corr_i,axis=0)
ISImeantoISIstd_corr_i_std=np.nanstd(ISImeantoISIstd_corr_i,axis=0)

linreg_r_mean=np.nanmean(linreg_r,axis=0)
linreg_r_std=np.nanstd(linreg_r,axis=0)
linreg_slope_mean=np.nanmean(linreg_slope,axis=0)
linreg_slope_std=np.nanstd(linreg_slope,axis=0)




#%% Save data in CSV
esyn_i=stats_csv[:,0]

str1='AverageOverRepetitionsPlots/VaryEsynI_Repetitions%d_NoiseAmp%d.csv' \
    % (repetitions, noiseamp,)

with open(str1, 'w', newline='') as csvfile:
    outputwriter=csv.writer(csvfile)
    
    for i in range(len(esyn_i)):
        mystr=[esyn_i[i], sync_means_mean[i], sync_means_std[i], sync_varOverTime_mean[i], sync_varOverTime_std[i],
               freq_means_mean[i], freq_means_std[i], freq_varOverTime_mean[i], freq_varOverTime_std[i],
               MFF_means_mean[i], MFF_means_std[i], MFF_varOverTime_mean[i], MFF_varOverTime_std[i],
               percent_means_mean[i], percent_means_std[i], percent_varOverTime_mean[i], percent_varOverTime_std[i],
               sync_onset_num_mean[i], sync_onset_num_std[i], sync_offset_num_mean[i], sync_offset_num_std[i],
               sync_duration_mean[i], sync_duration_std[i],
               DCtoSpike_corr_e_mean[i], DCtoSpike_corr_e_std[i], DCtoSpike_corr_i_mean[i], DCtoSpike_corr_i_std[i],
               DCtoISImean_corr_e_mean[i], DCtoISImean_corr_e_std[i], DCtoISImean_corr_i_mean[i], DCtoISImean_corr_i_std[i],
               DCtoISIstd_corr_e_mean[i], DCtoISIstd_corr_e_std[i], DCtoISIstd_corr_i_mean[i], DCtoISIstd_corr_i_std[i],
               ISImeantoISIstd_corr_e_mean[i], ISImeantoISIstd_corr_e_std[i], ISImeantoISIstd_corr_i_mean[i], ISImeantoISIstd_corr_i_std[i],
               linreg_r_mean[i], linreg_r_std[i], linreg_slope_mean[i], linreg_slope_std[i]]
        outputwriter.writerow(mystr)
        csvfile.flush()
    
    





#%% Plots  
esyn_i=stats_csv[:,0]

#Whole Simulation
plt.rcParams['font.size']=18
fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, sync_means_mean, sync_means_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Synchrony Measure')
ax1.set_ylim([0, 1])

ax2.errorbar(esyn_i, freq_means_mean, freq_means_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Burst Frequency')
ax2.set_ylim([0, 150])

ax3.errorbar(esyn_i, MFF_means_mean, MFF_means_std, fmt='o', markersize=8, capsize=10)
ax3.set_ylabel('Spiking Frequency')
ax3.set_ylim([0, 100])

ax4.errorbar(esyn_i, percent_means_mean, percent_means_std, fmt='o', markersize=8, capsize=10)
ax4.set_ylabel('Spiking Percentage\n Per Burst')
ax4.set_xlabel(r'$E_{GABA}$ (mV)')
ax4.set_ylim([0, 1])

titlestr='Whole Simulation; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_WholeSim_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()


#Over Time
plt.rcParams['font.size']=18
fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, sync_varOverTime_mean, sync_varOverTime_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Synchrony Measure')

ax2.errorbar(esyn_i, freq_varOverTime_mean, freq_varOverTime_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Burst Frequency')

ax3.errorbar(esyn_i, MFF_varOverTime_mean, MFF_varOverTime_std, fmt='o', markersize=8, capsize=10)
ax3.set_ylabel('Spiking Frequency')

ax4.errorbar(esyn_i, percent_varOverTime_mean, percent_varOverTime_std, fmt='o', markersize=8, capsize=10)
ax4.set_ylabel('Spiking Percentage\n Per Burst')
ax4.set_xlabel(r'$E_{GABA}$ (mV)')

titlestr='Variability (SD) Over Time; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_VarOverTime_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()


#Sync Transition Details
plt.rcParams['font.size']=18
fig, (ax1, ax2, ax3)=plt.subplots(3,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, sync_onset_num_mean, sync_onset_num_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Synchrony Onsets')

ax2.errorbar(esyn_i, sync_offset_num_mean, sync_offset_num_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Synchrony Offsets')

ax3.errorbar(esyn_i, sync_duration_mean, sync_duration_std, fmt='o', markersize=8, capsize=10)
ax3.set_ylabel('Synchrony Duration')
ax3.set_xlabel(r'$E_{GABA}$ (mV)')

titlestr='Synchrony Transitions/Duration; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_SyncTransitions_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()


#Correlations, E
plt.rcParams['font.size']=18
fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, DCtoSpike_corr_e_mean, DCtoSpike_corr_e_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Correlation, DC Input \n to Spiking Frequency')

ax2.errorbar(esyn_i, DCtoISImean_corr_e_mean, DCtoISImean_corr_e_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Correlation, DC Input \n to Mean ISI')

ax3.errorbar(esyn_i, DCtoISIstd_corr_e_mean, DCtoISIstd_corr_e_std, fmt='o', markersize=8, capsize=10)
ax3.set_ylabel('Correlation, DC Input \n to SD ISI')

ax4.errorbar(esyn_i, ISImeantoISIstd_corr_e_mean, ISImeantoISIstd_corr_e_std, fmt='o', markersize=8, capsize=10)
ax4.set_ylabel('Correlation, Mean ISI \n to SD ISI')
ax4.set_xlabel(r'$E_{GABA}$ (mV)')

titlestr='Correlations, Excitatory; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_CorrelationsE_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()


#Correlations, I
plt.rcParams['font.size']=18
fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, DCtoSpike_corr_i_mean, DCtoSpike_corr_i_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Correlation, DC Input \n to Spiking Frequency')

ax2.errorbar(esyn_i, DCtoISImean_corr_i_mean, DCtoISImean_corr_i_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Correlation, DC Input \n to Mean ISI')

ax3.errorbar(esyn_i, DCtoISIstd_corr_i_mean, DCtoISIstd_corr_i_std, fmt='o', markersize=8, capsize=10)
ax3.set_ylabel('Correlation, DC Input \n to SD ISI')

ax4.errorbar(esyn_i, ISImeantoISIstd_corr_i_mean, ISImeantoISIstd_corr_i_std, fmt='o', markersize=8, capsize=10)
ax4.set_ylabel('Correlation, Mean ISI \n to SD ISI')
ax4.set_xlabel(r'$E_{GABA}$ (mV)')

titlestr='Correlations, Inhibitory; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_CorrelationsI_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()
   

#Linear Regression
plt.rcParams['font.size']=18
fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)
fig.set_size_inches(24,17)

ax1.errorbar(esyn_i, linreg_r_mean, linreg_r_std, fmt='o', markersize=8, capsize=10)
ax1.set_ylabel('Linear Regression Rsquared')

ax2.errorbar(esyn_i, linreg_slope_mean, linreg_slope_std, fmt='o', markersize=8, capsize=10)
ax2.set_ylabel('Linear Regression Slope')

ax2.set_xlabel(r'$E_{GABA}$ (mV)')

titlestr='Linear Regression; Noise Amplitude=%1.7f' \
    % (noiseamp/1000000)
fig.suptitle(titlestr)
  
plt.tight_layout()


str2='AverageOverRepetitionsPlots/VaryEsynI_LinearRegression_Repetitions%d_NoiseAmp%d.png' \
    % (repetitions, noiseamp)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()

       