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
import matplotlib.cm as cm
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

# Iext_e=-75.0        #GABA reversal potential
esyn_e=0.0          #Glutamate reversal potential
tau_rise=0.2        #Synaptic rise constant
tau_decay_e=3.0     #Synaptic decay constant, excitatory
tau_decay_i=5.5     #Synaptic decya constant, inhibitory

# T=2500              #Length of simulation, ms
T=20000
# dt=0.05             #Time step
dt=.1

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
# else:
#     repetitions=int(sys.argv[1])
#     noiseamp=int(sys.argv[2])
#     Iext_e_mean=int(sys.argv[3])




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
    mean_sig = np.mean(sig_mat,axis=1); #calculate the mean signal
    variances = np.var(sig_mat,axis=0); #this is a row vector, where each entry is the variance of one of the columns in sig_mat
    G = np.sqrt(np.var(mean_sig,axis=0)/np.mean(variances)); #calculate the actual synchrony measure
    
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


def processSpikesForSync(spikes, start, end, numcells):
    spikes_end=[]
    for k in range(numcells):
        spikes_end.append([(j-(start)) for j in spikes[k] if (j>start) and (j<end)])
    return spikes_end















#%% Noise Values
noiseAmps=[0, 1, 20, 70, 200, 600, 1000, 2000, 4000, 6000, 8000]
extralabel='all'

noiseAmps=[noiseAmps[x] for x in [0,1,2,3,5,7,8,9,10]]
extralabel='sample'


Iextnum=11
Iextmin=.25
Iextmax=.5
Istep=0.025

repetitions=100

#%% Intitialize storage
sync_means_mean=np.zeros((Iextnum, len(noiseAmps)))
sync_varOverTime_mean=np.zeros((Iextnum, len(noiseAmps)))
freq_means_mean=np.zeros((Iextnum, len(noiseAmps)))
freq_varOverTime_mean=np.zeros((Iextnum, len(noiseAmps)))
MFF_means_mean=np.zeros((Iextnum, len(noiseAmps)))
MFF_varOverTime_mean=np.zeros((Iextnum, len(noiseAmps)))
percent_means_mean=np.zeros((Iextnum, len(noiseAmps)))
percent_varOverTime_mean=np.zeros((Iextnum, len(noiseAmps)))

sync_means_std=np.zeros((Iextnum, len(noiseAmps)))
sync_varOverTime_std=np.zeros((Iextnum, len(noiseAmps)))
freq_means_std=np.zeros((Iextnum, len(noiseAmps)))
freq_varOverTime_std=np.zeros((Iextnum, len(noiseAmps)))
MFF_means_std=np.zeros((Iextnum, len(noiseAmps)))
MFF_varOverTime_std=np.zeros((Iextnum, len(noiseAmps)))
percent_means_std=np.zeros((Iextnum, len(noiseAmps)))
percent_varOverTime_std=np.zeros((Iextnum, len(noiseAmps)))

synconset_mean=np.zeros((Iextnum,len(noiseAmps)))
synconset_std=np.zeros((Iextnum,len(noiseAmps)))
syncoffset_mean=np.zeros((Iextnum,len(noiseAmps)))
syncoffset_std=np.zeros((Iextnum,len(noiseAmps)))
syncduration_mean=np.zeros((Iextnum,len(noiseAmps)))
syncduration_std=np.zeros((Iextnum,len(noiseAmps)))

DCtoSpike_corr_e_mean=np.zeros((Iextnum,len(noiseAmps)))
DCtoSpike_corr_i_mean=np.zeros((Iextnum,len(noiseAmps)))
DCtoISImean_corr_e_mean=np.zeros((Iextnum,len(noiseAmps)))
DCtoISImean_corr_i_mean=np.zeros((Iextnum,len(noiseAmps)))
DCtoISIstd_corr_e_mean=np.zeros((Iextnum,len(noiseAmps)))
DCtoISIstd_corr_i_mean=np.zeros((Iextnum,len(noiseAmps)))
ISImeantoISIstd_corr_e_mean=np.zeros((Iextnum,len(noiseAmps)))
ISImeantoISIstd_corr_i_mean=np.zeros((Iextnum,len(noiseAmps)))

DCtoSpike_corr_e_std=np.zeros((Iextnum,len(noiseAmps)))
DCtoSpike_corr_i_std=np.zeros((Iextnum,len(noiseAmps)))
DCtoISImean_corr_e_std=np.zeros((Iextnum,len(noiseAmps)))
DCtoISImean_corr_i_std=np.zeros((Iextnum,len(noiseAmps)))
DCtoISIstd_corr_e_std=np.zeros((Iextnum,len(noiseAmps)))
DCtoISIstd_corr_i_std=np.zeros((Iextnum,len(noiseAmps)))
ISImeantoISIstd_corr_e_std=np.zeros((Iextnum,len(noiseAmps)))
ISImeantoISIstd_corr_i_std=np.zeros((Iextnum,len(noiseAmps)))

linreg_r_mean=np.zeros((Iextnum,len(noiseAmps)))
linreg_r_std=np.zeros((Iextnum,len(noiseAmps)))
linreg_slope_mean=np.zeros((Iextnum,len(noiseAmps)))
linreg_slope_std=np.zeros((Iextnum,len(noiseAmps)))

syncstart_mean=np.zeros((Iextnum,len(noiseAmps)))
syncstart_std=np.zeros((Iextnum,len(noiseAmps)))
syncstop_mean=np.zeros((Iextnum,len(noiseAmps)))
syncstop_std=np.zeros((Iextnum,len(noiseAmps)))
syncdurationNEW_mean=np.zeros((Iextnum,len(noiseAmps)))
syncdurationNEW_std=np.zeros((Iextnum,len(noiseAmps)))

syncpercent=np.zeros((Iextnum,len(noiseAmps)))



#%% Import and store data for plotting
for i, noiseamp in enumerate(noiseAmps): 
    
    print(noiseamp)
    str1='AverageOverRepetitionsPlots/VaryIext_Repetitions%d_NoiseAmp%d.csv' \
         % (repetitions, noiseamp)
         
    stats_csv=np.genfromtxt(str1, delimiter=',')
    
    str2='AverageOverRepetitionsPlots/NewBurstFreq_VaryIext_Repetitions%d_NoiseAmp%d.csv' \
         % (repetitions, noiseamp)
         
    burstfreq_csv=np.genfromtxt(str2, delimiter=',')

    
    sync_means_mean[:,i]=stats_csv[:,1]
    sync_means_std[:,i]=stats_csv[:,2]
    sync_varOverTime_mean[:,i]=stats_csv[:,3]
    sync_varOverTime_std[:,i]=stats_csv[:,4]

    freq_means_mean[:,i]=burstfreq_csv[:,1]
    freq_means_std[:,i]=burstfreq_csv[:,2]
    freq_varOverTime_mean[:,i]=stats_csv[:,7]
    freq_varOverTime_std[:,i]=stats_csv[:,8]

    MFF_means_mean[:,i]=stats_csv[:,9]
    MFF_means_std[:,i]=stats_csv[:,10]
    MFF_varOverTime_mean[:,i]=stats_csv[:,11]
    MFF_varOverTime_std[:,i]=stats_csv[:,12]

    percent_means_mean[:,i]=stats_csv[:,13]
    percent_means_std[:,i]=stats_csv[:,14]
    percent_varOverTime_mean[:,i]=stats_csv[:,15]
    percent_varOverTime_std[:,i]=stats_csv[:,16]    

    synconset_mean[:,i]=stats_csv[:,17]
    synconset_std[:,i]=stats_csv[:,18]
    syncoffset_mean[:,i]=stats_csv[:,19]
    syncoffset_std[:,i]=stats_csv[:,20]
    syncduration_mean[:,i]=stats_csv[:,21]
    syncduration_std[:,i]=stats_csv[:,22]
    
    DCtoSpike_corr_e_mean[:,i]=stats_csv[:,23]
    DCtoSpike_corr_e_std[:,i]=stats_csv[:,24]
    DCtoSpike_corr_i_mean[:,i]=stats_csv[:,25]
    DCtoSpike_corr_i_std[:,i]=stats_csv[:,26]

    DCtoISImean_corr_e_mean[:,i]=stats_csv[:,27]
    DCtoISImean_corr_e_std[:,i]=stats_csv[:,28]
    DCtoISImean_corr_i_mean[:,i]=stats_csv[:,29]
    DCtoISImean_corr_i_std[:,i]=stats_csv[:,30]

    DCtoISIstd_corr_e_mean[:,i]=stats_csv[:,31]
    DCtoISIstd_corr_e_std[:,i]=stats_csv[:,32]
    DCtoISIstd_corr_i_mean[:,i]=stats_csv[:,33]
    DCtoISIstd_corr_i_std[:,i]=stats_csv[:,34]

    ISImeantoISIstd_corr_e_mean[:,i]=stats_csv[:,35]
    ISImeantoISIstd_corr_e_std[:,i]=stats_csv[:,36]
    ISImeantoISIstd_corr_i_mean[:,i]=stats_csv[:,37]
    ISImeantoISIstd_corr_i_std[:,i]=stats_csv[:,38]
    
    linreg_r_mean[:,i]=stats_csv[:,39]
    linreg_r_std[:,i]=stats_csv[:,40]
    linreg_slope_mean[:,i]=stats_csv[:,41]
    linreg_slope_std[:,i]=stats_csv[:,42]
    
    syncstart_mean[:,i]=stats_csv[:,43]
    syncstart_std[:,i]=stats_csv[:,44]
    syncstop_mean[:,i]=stats_csv[:,45]
    syncstop_std[:,i]=stats_csv[:,46]
    syncdurationNEW_mean[:,i]=stats_csv[:,47]
    syncdurationNEW_std[:,i]=stats_csv[:,48]

    syncpercent[:,i]=stats_csv[:,49]


#%% Plots  
Iext_e=stats_csv[:,0]
cmap=cm.get_cmap('jet')

NoiseAmp_to_VoltageSD=dict([(0,0),(1,0.023),(20,0.108), (70, 0.199), (200, 0.325), (600, 0.581), (1000, 0.750), (2000,1.087), (4000,1.439), (6000, 1.842), (8000, 2.234)])





#Sync Percent
plt.rcParams['font.size']=24
fig, ax=plt.subplots()
fig.set_size_inches(12,7)

for i, noiseamp in enumerate(noiseAmps):
    str1='%1.3f' % (NoiseAmp_to_VoltageSD[noiseamp])
    if i==0:
        linewidth=5
    else:
        linewidth=3
    ax.plot(Iext_e[:-1], syncpercent[:-1,i], color=cmap(i/(len(noiseAmps)-1)), label=str1, lw=linewidth)
    # ax.fill_between(Iext_e, syncdurationNEW_mean[:,i]+syncdurationNEW_std[:,i], syncdurationNEW_mean[:,i]-syncdurationNEW_std[:,i], 
    #                 color=cmap(i/(len(noiseAmps)-1)), alpha=.1)
ax.legend(title=r'$SD_V$ (mV)', ncol=1, bbox_to_anchor=(1.00, 1.0))

ax.set_ylabel('Proportion of simulations that \n exhibit stable PING-like activity')
ax.set_xlabel(r'$I_{ext}$ ($\mu$ A)')
ax.set_xticks(np.arange(Iextmin,.475+.01,Istep*2))
# ax.set_ylim([0, 1])
ax.spines[['right', 'top']].set_visible(False)


str2=r'PaperFigures/Figure4/VaryIextE_NewSyncPercent_Repetitions%d_%s.png' \
    % (repetitions, extralabel)
plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

plt.show(block=False)
plt.close()
       