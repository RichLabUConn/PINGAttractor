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
# import time
from scipy.signal import find_peaks
# import seaborn as sns
import csv
# from sklearn.linear_model import LinearRegression

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

T=5000              #Length of simulation, ms
# T=2000
dt=.02              #Time step

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
    number=0                # Label/random seed
    # trials=1              # Number of trials
    trials=1                # Number of trials
    # esyn_i=-75.0          # GABA reversal potential
    g_ee=0.0                # E-E synaptic weight
    g_ei=0.00235            # E-I synaptic weight
    g_ii=0.025              # I-I synaptic weight
    g_ie=0.003              # I-E synaptic weight
    # Iext_e_mean=0.25        # Mean excitatory input
    dt=0.02                  # Time step
    esyn_i_min=-70.0        # Min esyn_i for loop
    esyn_i_max=-70.0        # Max esyn_i for loop
    esyn_i_step=1.0         # esyn_i step for loop
    syncsample=int(1000/dt) # How often to calculate synchrony
    D=0.0                   # Noise variance/amplitude
    # D=0.0001                # Noise variance/amplitude


else:
    number=int(sys.argv[1])
    trials=int(sys.argv[2])
    D=float(sys.argv[3])

    g_ee=0.0                # E-E synaptic weight
    g_ei=0.00235            # E-I synaptic weight
    g_ii=0.025              # I-I synaptic weight
    g_ie=0.003              # I-E synaptic weight

    dt=0.02                  # Time step
    esyn_i_min=-75.0        # Min esyn_i for loop
    esyn_i_max=-65.0        # Max esyn_i for loop
    esyn_i_step=1.0         # esyn_i step for loop
    syncsample=int(1000/dt) # How often to calculate synchrony






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




# Euler-Marayama Integration
def eulermaruyama(v,z,h,n,I,D,ou_i_last):
    geo=0
    tau=5
    Ae=np.sqrt(((D*tau)/2)*(1-np.exp(-2*dt/tau)))
    ou_i=geo+(ou_i_last-geo)*np.exp((-dt/tau))+Ae*np.random.randn(len(v),1)
    dv=dt*dvdt(v,z,h,n,I).reshape(-1,1)+dt*(ou_i)
    dz=dt*dzdt(v,z,h,n,I)
    dh=dt*dhdt(v,z,h,n,I)
    dn=dt*dndt(v,z,h,n,I)
    
    out_v=v.reshape(-1,1)+dv
    out_z=z+dz
    out_h=h+dh
    out_n=n+dn
    
    out_v=out_v.flatten()
    
    return out_v, out_z, out_h, out_n, ou_i




# Detect spikes
def spikedetect(v_2, v_1, v_0, lastspike, t):
    if v_2 >0 and v_2<v_1 and v_1>v_0 and t-lastspike>1:
        return True
    else:
        return False
    
    # if v_2>10 and v_1<10:
    #     return True
    # else:
    #     return False
    
    
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


def histogram_SpikeRateIndividualCells(spikes, start, end, numcells):
    spikes_end=processSpikesForSync(spikes,start,end,numcells)
    spikerates=[]
    for i in range(numcells):
        spikerates.append(len(spikes_end[i])/((end-start)/1000))
    return spikerates

def calculate_ISIs(spikes, start, end, numcells):
    spikes_end=processSpikesForSync(spikes,start,end,numcells)
    ISI_mean=[]
    ISI_std=[]
    for i in range(numcells):
        ISIs=np.array(spikes_end[i][1:])-np.array(spikes_end[i][:-1])
        ISI_mean.append(np.mean(ISIs))
        ISI_std.append(np.std(ISIs))
    return ISI_mean, ISI_std
        

def conv_gaussian_NEW(signal,srate,gauss_width):   
    # This function takes a time series sampled at a rate 'srate' (specified in Hz)
    # and convolves it with a Gaussain with sd 'gauss_width (ms)
    
    #require that the Gaussian run from -4 to +4 SDs, and sample the Gaussian
    #according to the specified sampling rate. The number fo points in the 
    #Gaussian is given below
    N_gauss = round(7*(gauss_width/1000)*srate);   #want a standard deviation of 2ms typically

    #Want the number of points in the Gaussian to be odd, so that it is
    #centrally peaked, rather than flat at the top
    if (N_gauss%2)==0:
        N_gauss=N_gauss-1
        
    x=np.linspace(-3.5*gauss_width,3.5*gauss_width, N_gauss)
    g=(1/(gauss_width*np.sqrt(2*np.pi)))*np.exp((-(x**2))/(2*gauss_width**2))
    
    newsig=np.convolve(signal, g, 'same')
    
    return newsig 



def firingRateGaussian(spikes, sd, scale):
    gaussian_sum=np.zeros(int(T/dt/scale))
    for i in range(len(spikes)):
        if np.mod(i,100)==0:
            print('SD= ', sd, '; Cell= ', i)
        timeseries=convert_spiketimes(spikes[i], T/1000, 1000/dt/scale)
        gaussian=conv_gaussian_NEW(timeseries[1], 1000/dt/scale, sd)
        gaussian_sum=gaussian_sum+gaussian
    # gaussian_sum=gaussian_sum/len(spikes)
    
    return gaussian_sum
        



















"""
Turn simulation into a function call to facilitate looping
"""
def run_sim(labeltext, D_perturb, D_set):
    
    #%% Initialize variables
        
    v_e=np.zeros((numcells_e, history))
    v_i=np.zeros((numcells_i, history))
    z_e=np.zeros((numcells_e, history))
    z_i=np.zeros((numcells_i, history))
    h_e=np.zeros((numcells_e, history))
    h_i=np.zeros((numcells_i, history))
    n_e=np.zeros((numcells_e, history))
    n_i=np.zeros((numcells_i, history))
        
    spikes_e=[[] for i in range(numcells_e)]
    spikes_i=[[] for i in range(numcells_i)]
    
    Iext_e=np.zeros((numcells_e,1))
    Iext_i=np.zeros((numcells_i,1))
    
    Isyn_e=np.zeros((numcells_e,1))
    Isyn_i=np.zeros((numcells_i,1))
    
    Isynin_ee=np.zeros((numcells_e,1))
    Isynin_ie=np.zeros((numcells_e,1))
    Isynin_ii=np.zeros((numcells_i,1))
    Isynin_ei=np.zeros((numcells_i,1))
    
    Iapp_e=np.zeros((numcells_e,1))
    Iapp_i=np.zeros((numcells_i,1))
    
    Isyn_e_future=np.zeros((numcells_e, int(50/dt)))
    Isyn_i_future=np.zeros((numcells_i, int(50/dt)))
    EPSP=syn_e_vectorize(0,np.arange(0,50/dt,1))
    IPSP=syn_i_vectorize(0,np.arange(0,50/dt,1))

    
            
    #%% Running the model

    #Set random seed
    np.random.seed(number)
        

    
    #%%Trial loop
    for q in range(trials):
        print('Starting Trial', q, ' Esyn_i=', esyn_i)
        
        # Reset variables to 0
        v_e=np.zeros((numcells_e, history))
        v_i=np.zeros((numcells_i, history))
        z_e=np.zeros((numcells_e, history))
        z_i=np.zeros((numcells_i, history))
        h_e=np.zeros((numcells_e, history))
        h_i=np.zeros((numcells_i, history))
        n_e=np.zeros((numcells_e, history))
        n_i=np.zeros((numcells_i, history))
    
        spikes_e=[[] for i in range(numcells_e)]
        spikes_i=[[] for i in range(numcells_i)]
        
        Iext_e=np.zeros((numcells_e,1))
        Iext_i=np.zeros((numcells_i,1))
        
        Isyn_e=np.zeros((numcells_e,1))
        Isyn_i=np.zeros((numcells_i,1))
        
        Isynin_ee=np.zeros((numcells_e,1))
        Isynin_ie=np.zeros((numcells_e,1))
        Isynin_ii=np.zeros((numcells_i,1))
        Isynin_ei=np.zeros((numcells_i,1))
    
        Iapp_e=np.zeros((numcells_e,1))
        Iapp_i=np.zeros((numcells_i,1))
        
        Isyn_e_future=np.zeros((numcells_e, int(50/dt)))
        Isyn_i_future=np.zeros((numcells_i, int(50/dt)))
        
    
    
    
    
        # Create connectivity matricies
        conmatrix_ee=create_conmatrix(numcells_e, numcells_e, pcon_ee, g_ee)
        conmatrix_ei=create_conmatrix(numcells_i, numcells_e, pcon_ei, g_ei)
        conmatrix_ii=create_conmatrix(numcells_i, numcells_i, pcon_ii, g_ii)
        conmatrix_ie=create_conmatrix(numcells_e, numcells_i, pcon_ie, g_ie)
        
        
        # Generate external input currents
        Iext_e=sample_inputs(numcells_e, Iext_e_mean, hetero_e)
        Iext_i=sample_inputs(numcells_i, Iext_i_mean, hetero_i)
        
        
        # Set initial conditions
        v_e[:,-1], z_e[:,-1], h_e[:,-1], n_e[:,-1]=sample_initialconditions(numcells_e)
        v_i[:,-1], z_i[:,-1], h_i[:,-1], n_i[:,-1]=sample_initialconditions(numcells_i)
        ou_i_e=np.zeros((numcells_e,1))
        ou_i_i=np.zeros((numcells_i,1))
        
    
    
        #%%Time loop
        for i in range(int(T/dt)):
            v_e=shift(v_e)
            v_i=shift(v_i)
            z_e=shift(z_e)
            z_i=shift(z_i)
            h_e=shift(h_e)
            h_i=shift(h_i)
            n_e=shift(n_e)
            n_i=shift(n_i)
            
            Isyn_e_future=shift(Isyn_e_future)
            Isyn_i_future=shift(Isyn_i_future)
            
            
            #Perturbation
            if 3250<i*dt<3750:
                D_use=D_perturb+D_set
            else:
                D_use=D_set

            
            
            #Detect spikes
            if i>2:
                for j in range(numcells_e):
                    if len(spikes_e[j])==0:
                        if spikedetect(v_e[j,-2], v_e[j,-3], v_e[j,-4], 0, i*dt)==True:
                            spikes_e[j].append(i*dt)
                            if i*dt>250:
                                Isyn_e_future[j]=Isyn_e_future[j]+EPSP
                    else:
                        if spikedetect(v_e[j,-2], v_e[j,-3], v_e[j,-4], spikes_e[j][-1], i*dt)==True:
                            spikes_e[j].append(i*dt)
                            if i*dt>250:
                                Isyn_e_future[j]=Isyn_e_future[j]+EPSP

                for j in range(numcells_i):
                    if len(spikes_i[j])==0:
                        if spikedetect(v_i[j,-2], v_i[j,-3], v_i[j,-4], 0, i*dt)==True:
                            spikes_i[j].append(i*dt)
                            if i*dt>250:
                                Isyn_i_future[j]=Isyn_i_future[j]+IPSP
                    else:
                        if spikedetect(v_i[j,-2], v_i[j,-3], v_i[j,-4], spikes_i[j][-1], i*dt)==True:
                            spikes_i[j].append(i*dt)
                            if i*dt>250:
                                Isyn_i_future[j]=Isyn_i_future[j]+IPSP
            
            
            #Calculate output synaptic currents
            Isyn_e=Isyn_e_future[:,0]
            Isyn_i=Isyn_i_future[:,0]
    
            
            #Calculate synaptic inputs
            Isynin_ee=np.matmul(conmatrix_ee,Isyn_e)
            Isynin_ei=np.matmul(conmatrix_ei,Isyn_e)
            Isynin_ii=np.matmul(conmatrix_ii,Isyn_i)
            Isynin_ie=np.matmul(conmatrix_ie,Isyn_i)
            
            
            #Calculate Iapp, net of Iext and Isyn
            Iapp_e=Iext_e-np.multiply(Isynin_ee,(v_e[:,-2]-esyn_e))-np.multiply(Isynin_ie,(v_e[:,-2]-esyn_i))
            Iapp_i=Iext_i-np.multiply(Isynin_ei,(v_i[:,-2]-esyn_e))-np.multiply(Isynin_ii,(v_i[:,-2]-esyn_i))
            
            
            #Integrate
            v_e[:,-1], z_e[:,-1], h_e[:,-1], n_e[:,-1], ou_i_e=eulermaruyama(v_e[:,-2], z_e[:,-2], h_e[:,-2], n_e[:,-2], Iapp_e, D_use, ou_i_e)
            v_i[:,-1], z_i[:,-1], h_i[:,-1], n_i[:,-1], ou_i_i=eulermaruyama(v_i[:,-2], z_i[:,-2], h_i[:,-2], n_i[:,-2], Iapp_i, D_use, ou_i_i)
    
        
            #Output
            if np.mod(i,10000)==0:
                print("Time= ",i*dt)
                
            # #Calculate Sync
            # if i>(1250/dt) and np.mod(i,syncsample)==0:
            #     spikes_e_end=[]
            #     for k in range(numcells_e):
            #         spikes_e_end.append([(j-(i*dt-1000)) for j in spikes_e[k] if j>((i*dt)-1000)])
            #     sync_overtime[0].append(i*dt)
            #     sync, freq, MFF=syncmeasure(numcells_e, spikes_e_end, 2)
            #     sync_overtime[1].append(sync)
            #     sync_overtime[2].append(freq)
            #     sync_overtime[3].append(MFF)
                
            # # Track voltage of one neuron for initial sanity checks
            # track.append(v_e[0,-1])
                
        #Time Loop Ended
        
        
        
        
        #%% Output CSV
                    
        str1='CSV_Spiking/SpikingData_OU_Esyni%1.0f_NoiseAmp%1.0f_%s.csv' \
             % ( esyn_i, D_set*1000000, labeltext)
             
        with open(str1, 'a', newline='') as csvfile_spike:
            outputwriter_spike=csv.writer(csvfile_spike)
            
            for j in range(numcells_e+numcells_i):
                if j<numcells_e:
                    for k in range(len(spikes_e[j])):
                        mystr=[j, spikes_e[j][k]]
                        outputwriter_spike.writerow(mystr)
                else:
                    for k in range(len(spikes_i[j-numcells_e])):
                        mystr=[j, spikes_i[j-numcells_e][k]]
                        outputwriter_spike.writerow(mystr)
            csvfile_spike.flush()
                
                         
        


        #%% Plot Raster     
        timestart=2000
        timestop=T     
        
        #%% Calculate firing rate gaussians and corresponding peak measures
        # names=dict([(0, 'shorter'), (1, 'short'), (2, 'mid'), (3, 'long')])
        
        window_shorter=2
        scale=12.5
        
        
        firingrate_shorter=firingRateGaussian(spikes_e, window_shorter, scale)
            
            
        #%% Setup plotting
        plt.rcParams['font.size']=16
        fig=plt.figure(figsize=(7,7))
        gs = fig.add_gridspec(3,1)
        ax1=fig.add_subplot(gs[0:2,0])
        ax2=fig.add_subplot(gs[2,0])
        
        
        #%% Plot Raster
        spikes_e_plot=[]
        for i in range(len(spikes_e)):
            for j in range(len(spikes_e[i])):
                spikes_e_plot.append([i,spikes_e[i][j]])
        spikes_e_plot_array=np.array(spikes_e_plot)
        
        spikes_i_plot=[]
        for i in range(len(spikes_i)):
            for j in range(len(spikes_i[i])):
                spikes_i_plot.append([i+numcells_e,spikes_i[i][j]])
        spikes_i_plot_array=np.array(spikes_i_plot)
        
        
        ax1.scatter(spikes_e_plot_array[:,1], spikes_e_plot_array[:,0], marker=".", s=.1, c='g')
        ax1.scatter(spikes_i_plot_array[:,1], spikes_i_plot_array[:,0], marker=".", s=.1, c='r')
        ax1.set_ylim(0,1000)
        ax1.set_xlim(timestart,timestop)
        
        # ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Neuron Index")
        ax1.tick_params(labelbottom=False)       
        
        
        #%% Plot Firing Rate Gaussian
        legstr1='Kernel: %d ms' % window_shorter
        
        ax2.plot(np.arange(0,T,dt*scale), firingrate_shorter,label=legstr1)
        ax2.set_xlim(timestart, timestop)
        ax2.set_ylim(0, 125)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel('Mean excitatory spikes\nper cell per second', multialignment='center', fontsize=14)     
        # ax2.legend(loc='upper right', ncol=2)
        
        
        #%% Save
        filenamestr=r'Rasters/FiringRateGauss_Raster_Esyni%1.0f_NoiseAmp%1.0f_%d_start%1.0f_stop%1.0f_%s.png' \
            % (esyn_i, D_set*1000000, number, timestart, timestop, labeltext)
        
        plt.savefig(filenamestr, dpi=600, format='png', bbox_inches='tight')




    

  
    
        print('Ending Trial', q, ' Esyn_i=', esyn_i)
        
        
        
        
    return 



















#%% Run 
for D_set in [0.0, 0.00002, 0.0006, 0.002]:

    for D_perturb in [0.000001, 0.00002, 0.00007, 0.0002, 0.0006, 0.001, 0.002, 0.004, 0.006, 0.008]:
        labeltext='NoisePerturb%d_500ms' % (D_perturb*1000000)
        
        # Adjust external input for Euler
        Iext_e_mean_adjust=dict([(.1,0.52),(.05,0.38), (.02,0.3), (.01,0.28)])
        Iext_e_mean=Iext_e_mean_adjust[dt]
        
        
        for esyn_i in np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step):
            run_sim(labeltext, D_perturb, D_set)
        






#%% Final plot for EsynI loop
 
# plt.rcParams['font.size']=18
# fig, (ax1, ax2, ax3, ax4)=plt.subplots(4,1, sharex=True)
# fig.set_size_inches(24,17)

# ax1.errorbar(np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step), sync_FullSim_esyni, yerr=sync_stdOverTime_esyni, fmt='o', markersize=8, capsize=10)

# titlestr='g_ee= %1.5f, g_ei= %1.5f, g_ii= %1.5f, gie= %1.5f, I_e= %1.2f, D=%1.6f' \
#     % (g_ee, g_ei, g_ii, g_ie, Iext_e_mean, D)
# ax1.set_title(titlestr)
# ax1.set_ylabel('Synchrony Measure', multialignment='center', fontsize=16)
# ax1.set_ylim(0,1)
# # ax1._xlabel('ESyn_i, mV')

# ax2.errorbar(np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step), freq_FullSim_esyni, yerr=freq_stdOverTime_esyni, fmt='o', markersize=8, capsize=10)
# ax2.set_ylabel('Burst Frequency', multialignment='center', fontsize=16)
# ax2.set_ylim(0,90)
# # ax2._xlabel('ESyn_i, mV')

# ax3.errorbar(np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step), MFF_FullSim_esyni, yerr=MFF_stdOverTime_esyni, fmt='o', markersize=8, capsize=10)
# ax3.set_ylabel('MFF', multialignment='center', fontsize=16)
# # ax3.set_xlabel('ESyn_i, mV')
# ax3.set_ylim(0,50)

# ax4.errorbar(np.arange(esyn_i_min, esyn_i_max+1, esyn_i_step), meanSpikePercentagePerBurst_FullSim_esyni, yerr=meanSpikePercentagePerBurst_stdOverTime_esyni, fmt='o', markersize=8, capsize=10)
# ax4.set_ylabel('Spiking Percentage \n Per Burst', multialignment='center', fontsize=16)
# ax4.set_xlabel('ESyn_i, mV')
# ax4.set_ylim(0,1)


# plt.tight_layout()

# # manager = plt.get_current_fig_manager()
# # manager.window.showMaximized()

# str2='SyncStats_Ie%1.0f_NoiseAmp%1.0f_%d.png' \
#     % (Iext_e_mean*100, D*1000000, number)
# plt.savefig(str2, dpi=300, format='png', bbox_inches='tight')

# plt.show(block=False)
# plt.close('all')

    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
