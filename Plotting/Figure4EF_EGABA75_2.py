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
# import sys
import os
import csv
# import time
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from functools import partial
import matplotlib.animation as animation
import seaborn as sns
import warnings
from scipy import signal


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
# if len(sys.argv)==1:
#     number=0                # Label/random seed
#     trials=1                # Number of trials
#     esyn_i=-75.0            # GABA reversal potential
#     Iext_e_mean=0.25        # Mean excitatory input

# else:
#     number=int(sys.argv[1])
#     trials=int(sys.argv[2])
#     esyn_i=float(sys.argv[3])
#     Iext_e_mean=float(sys.argv[4])







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

# def synoutput_e(spikes,i):
#     # output=0
#     # for i in range(len(spikes)):
#     #     output=output+syn_e(spikes[i])
#     # return output
#     output=syn_e_vectorize(spikes,i).sum()
#     return output

# def synoutput_i(spikes,i):
#     # output=0
#     # for i in range(len(spikes)):
#     #     output=output+syn_i(spikes[i])
#     # return output
#     output=syn_i_vectorize(spikes,i).sum()
#     return output


# def syncalc_e(spikes_e,i):
#     Isyn_e=np.zeros((numcells_e,1))
#     for j in range(numcells_e):
#         spikes_e_array=np.array([x for x in spikes_e[j] if x>100 and x>(i*dt-30)])
#         if len(spikes_e_array)>0:
#             Isyn_e[j]=synoutput_e(spikes_e_array,i)
#         else:
#             Isyn_e[j]=0
#     return Isyn_e
            
# def syncalc_i(spikes_i,i):
#     Isyn_i=np.zeros((numcells_i,1))
#     for j in range(numcells_i):
#         spikes_i_array=np.array([x for x in spikes_i[j] if x>100 and x>(i*dt-30)])
#         if len(spikes_i_array)>0:
#             Isyn_i[j]=synoutput_i(spikes_i_array,i)
#         else:
#             Isyn_i[j]=0
#     return Isyn_i


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
    #NOTE: THIS CREATES A GAUSSIAN WITH SD=SQRT(1/2) ALWAYS
    
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


# def animate(i, graph, ax, peaks, numpeaks):
#     alphas=np.power(np.linspace(.99-.9*((i+1)/numpeaks),1,i+1),4)
#     colors_temp=[]
#     edgecolors_temp=[]
#     for j in range(i):
#         color_now=list(colors[j])
#         color_now[3]=alphas[j]
#         colors_temp.append(tuple(color_now))
#         edgecolors_now=[0,0,0,np.power(1-(i-j)/i,20)]
#         edgecolors_temp.append(tuple(edgecolors_now))
    
#     graph.set_offsets(np.vstack((x[:i+1], y[:i+1])).T)
#     graph.set_facecolors(colors_temp[:i+1])
#     graph.set_edgecolors(edgecolors_temp[:i+1])
    
    
#     temptitle_str='Time=%1.0f ms' % (spikes_e_end_plot[i])
#     ax.set_title(temptitle_str)
    
#     animate_raster(spikes_e_end_plot[i], ax2)
    
#     line.set_xdata([spikes_e_end_plot[i], spikes_e_end_plot[i+1]])    
    
#     return graph, line



















def animate(i, graph, ax, peaks, numpeaks, line1, line2, line3, graph2, graph3, numISI, peaks_shorter):
    alphas=np.power(np.linspace(.99-.5*((peaks_shorter[i]+1)/len(x)),1,peaks_shorter[i]+1),4)
    colors_temp=[]
    edgecolors_temp=[]
    for j in range(peaks_shorter[i]):
        color_now=list(colors[j])
        color_now[3]=alphas[j]
        colors_temp.append(tuple(color_now))
        edgecolors_now=[0,0,0,np.power(1-(peaks_shorter[i]-j)/peaks_shorter[i],20)]
        edgecolors_temp.append(tuple(edgecolors_now))
    
    graph.set_offsets(np.vstack((x[8000:peaks_shorter[i]], y[8000:peaks_shorter[i]])).T)
    graph.set_facecolors(colors_temp[8000:peaks_shorter[i]])
    graph.set_edgecolors(edgecolors_temp[8000:peaks_shorter[i]])
    
    
    temptitle_str='Time=%1.0f ms' % (peaktimes_shorter[i])
    ax.set_title(temptitle_str)
    
    animate_firingrate(peaktimes_shorter[i], ax2)
    
    line1.set_xdata([peaktimes_shorter[i]-20, peaktimes_shorter[i]-20])    
    line2.set_xdata([peaktimes_shorter[i]+20, peaktimes_shorter[i]+20])    


        
    # # Check if we've moved past the next spike of given cell
    # trackspike=0
    # if peaktimes_shorter[i]>1250:
    #     nowspike=np.argmin(np.array(spikes_e_end_plot<peaktimes_shorter[i]+10))-1
    # else:
    #     nowspike=0
        
    # alphas2=np.power(np.linspace(.99-.5*((nowspike+1)/numISI),1,nowspike+1),4)
    # colors_temp2=[]
    # edgecolors_temp2=[]
    # for j in range(nowspike):
    #     color_now2=list(colors2[j])
    #     color_now2[3]=alphas2[j]
    #     colors_temp2.append(tuple(color_now2))
    #     edgecolors_now2=[0,0,0,np.power(1-(nowspike-j)/nowspike,20)]
    #     edgecolors_temp2.append(tuple(edgecolors_now2))

        
    # if nowspike>trackspike:
    #     graph2.set_offsets(np.vstack((x2[:nowspike+1], y2[:nowspike+1])).T)
    #     graph2.set_facecolors(colors_temp2[:nowspike+1])
    #     graph2.set_edgecolors(edgecolors_temp2[:nowspike+1])
    #     trackspike=nowspike
    #     line3.set_xdata([spikes_e_end_plot[nowspike], spikes_e_end_plot[nowspike+1]])    
    
    # temptitle_str='Time=%1.0f ms' % (peaktimes_shorter[i])
    # ax4.set_title(temptitle_str)
        
    

    
    
    return graph, graph2, graph3, line1, line2, line3


def animate_firingrate(i, ax):
    
    if i<frame:
        ax.set_xlim(0,frame)
        ax.set_xticks([0, frame/4, frame/2, 3*frame/4, frame])
        # ax.tick_params(labelbottom=False)    
    elif i>1000 and i<(T-frame):
        ax.set_xlim(i-frame/2, i+frame/2)
        ax.set_xticks([i-frame/2, i-frame/2+frame/4, i-frame/2+frame/2, i-frame/2+3*frame/4, i+frame/2])
        # ax.tick_params(labelbottom=False)    
    elif i>(T-frame):
        ax.set_xlim(T-frame,T)
        ax.set_xticks([T-frame, T-frame+frame/4, T-frame+frame/2, T-frame+3*frame/4, T])
        # ax.tick_params(labelbottom=False)    
    
    animate_raster(i,ax3)
    

def animate_raster(i,ax):
    if i<1000:
        ax.set_xlim(0,frame)
        ax.set_xticks([0, frame/4, frame/2, 3*frame/4, frame])
    elif i>1000 and i<(T-frame):
        ax.set_xlim(i-frame/2, i+frame/2)
        ax.set_xticks([i-frame/2, i-frame/2+frame/4, i-frame/2+frame/2, i-frame/2+3*frame/4, i+frame/2])
    elif i>(T-frame):
        ax.set_xlim(T-frame,T)
        ax.set_xticks([T-frame, T-frame+frame/4, T-frame+frame/2, T-frame+3*frame/4, T])
        
    animate_spectogram_e(i,ax4, graph2)
        

def animate_spectogram_e(i,ax, graph):
    if i<frame:
        ax.set_xlim(0,frame/1000)
        ax.set_xticks([0, (frame/4)/1000, (frame/2)/1000, (3*frame/4)/1000, frame/1000])
        ax.tick_params(labelbottom=False)            
        graph.set_clim(0,.5)
    elif i>1000 and i<(T-frame):
        ax.set_xlim((i-frame/2)/1000, (i+frame/2)/1000)
        ax.set_xticks([(i-frame/2)/1000, (i-frame/2+frame/4)/1000, (i-frame/2+frame/2)/1000, (i-frame/2+3*frame/4)/1000, (i+frame/2)/1000])
        ax.tick_params(labelbottom=False)
        graph.set_clim(0,np.max([.5,np.max(Sxx_e[:,(np.abs(t_e-((i-frame/2)/1000))).argmin():(np.abs(t_e-((i+frame/2)/1000))).argmin()])]))
    elif i>(T-frame):
        ax.set_xlim((T-frame)/1000,T/1000)
        ax.set_xticks([(T-frame)/1000, (T-frame+frame/4)/1000, (T-frame+frame/2)/1000, (T-frame+3*frame/4)/1000, T/1000])
        ax.tick_params(labelbottom=False)
        graph.set_clim(0,np.max([.5,np.max(Sxx_e[:,(np.abs(t_e-((T-frame)/1000))).argmin():(np.abs(t_e-(T/1000))).argmin()])]))

    animate_spectogram_i(i,ax5, graph3)    
        
    
def animate_spectogram_i(i,ax, graph):
    if i<frame:
        ax.set_xlim(0,frame/1000)
        ax.set_xticks([0, (frame/4)/1000, (frame/2)/1000, (3*frame/4)/1000, frame/1000])
        # ax.tick_params(labelbottom=False)            
        graph.set_clim(0,.25)
    elif i>1000 and i<(T-frame):
        ax.set_xlim((i-frame/2)/1000, (i+frame/2)/1000)
        ax.set_xticks([(i-frame/2)/1000, (i-frame/2+frame/4)/1000, (i-frame/2+frame/2)/1000, (i-frame/2+3*frame/4)/1000, (i+frame/2)/1000])
        # ax.tick_params(labelbottom=False)
        graph.set_clim(0,np.max([.2,np.max(Sxx_i[:,(np.abs(t_e-((i-frame/2)/1000))).argmin():(np.abs(t_e-((i+frame/2)/1000))).argmin()])]))
    elif i>(T-frame):
        ax.set_xlim((T-frame)/1000,T/1000)
        ax.set_xticks([(T-frame)/1000, (T-frame+frame/4)/1000, (T-frame+frame/2)/1000, (T-frame+3*frame/4)/1000, T/1000])
        # ax.tick_params(labelbottom=False)
        graph.set_clim(0,np.max([.2,np.max(Sxx_i[:,(np.abs(t_e-((T-frame)/1000))).argmin():(np.abs(t_e-(T/1000))).argmin()])]))



















def firingRateHistogram(spikes, window):
    windows=np.arange(0,T+dt,window)
    spikecount=np.zeros(len(windows))
    
    for i in range(len(windows)):
        for j in range(len(spikes)):
            spikecount[i]=spikecount[i]+len([1 for k in spikes[j] if (k>windows[i] and k<windows[i+1])])
    
    firingrate=spikecount/(len(spikes)*(window/1000))
                                            
    return firingrate


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
        # if np.mod(i,100)==0:
            # print('SD= ', sd, '; Cell= ', i)
        timeseries=convert_spiketimes(spikes[i], T/1000, 1000/dt/scale)
        gaussian=conv_gaussian_NEW(timeseries[1], 1000/dt/scale, sd)
        gaussian_sum=gaussian_sum+gaussian
    # gaussian_sum=gaussian_sum/len(spikes)
    
    return gaussian_sum
        
        
















    

#%% Loop over all repetitions for given noise and IExt
# for noiseamp in [0, 1, 20, 70, 200, 600, 1000, 2000]:
Iext_e_range=[25,28,30,33,35,38,40,43,45,48,50]
for noiseamp in [0, 2000]:
    nonPINGduration=np.zeros([len(Iext_e_range), 100])
    gammaEvents=np.zeros([len(Iext_e_range), 100])

    # CSV
    str1='FinalFigure4Quants\GammaEvents_NonPINGDuration_EGABA75_NoiseAmp%1.0f.csv' % (noiseamp)
         
    with open(str1, 'a', newline='') as csvfile:
        outputwriter=csv.writer(csvfile)
        
        for Iext_e_index, Iext_e in enumerate(Iext_e_range):
            print('Iext_e= ', Iext_e)
        
            for number_index, number in enumerate(np.arange(1,101,1)):
                print('Number= ', number)
                
                # str1=r'D:\OneDrive - University of Connecticut\3) Papers\1) Steve Paper\1) Redoing simulations\3) New initial conditions\CSV_Spiking\SpikingData_OUNewInitial_NoiseAmp%1.0f_Iext%1.0f_%d.csv' \
                #      % (noiseamp, Iext_e, number)

                str1=r'CSV_Spiking/SpikingData_OUNewInitial_NoiseAmp%1.0f_Iext%1.0f_%d.csv' \
                     % (noiseamp, Iext_e, number)
                     
                if os.path.exists(str1)==True:
                     
                    spiking_csv=np.genfromtxt(str1, delimiter=',')
                 
                    #Process data into spike array format        
                    spikes_e=[[] for i in range(numcells_e)]
                    spikes_i=[[] for i in range(numcells_i)]
                    
                    for i in range(len(spiking_csv)):
                        if spiking_csv[i][0]<numcells_e:
                            spikes_e[int(spiking_csv[i][0])].append(spiking_csv[i][1])
                        else:
                            spikes_i[int(spiking_csv[i][0]-numcells_e)].append(spiking_csv[i][1])
                         
                    #%% Calculate firing rate gaussians and corresponding peak measures
                    # names=dict([(0, 'shorter'), (1, 'short'), (2, 'mid'), (3, 'long')])
                    names=dict([(0, 'shorter')])
            
                    window_shorter=2
                    # window_short=5
                    # window_mid=15
                    # window_long=30
                    
                    scale=12.5
                    # scale=50
                    # scale=1
                    
                    frame=1000
                    
                    firingrate_shorter=[]               
                    peakheights_shorter=[]
                    peaktimes_shorter=[]
                    
                    firingrate_shorter_i=[]               
                    peakheights_shorter_i=[]
                    peaktimes_shorter_i=[]
                    
                    peaks_shorter=[]
            
                    
                    for i in range(len(names)):
                        vars()["firingrate_"+names[i]]=firingRateGaussian(spikes_e, vars()["window_"+names[i]], scale)
                        vars()["peaks_"+names[i]], vars()["peak_heights_"+names[i]]= \
                            find_peaks(vars()["firingrate_"+names[i]],height=5, prominence=5, distance=10)
                        vars()["peakheights_"+names[i]]=vars()["peak_heights_"+names[i]]["peak_heights"]
                        temp=vars()["peakheights_"+names[i]]
                        vars()["peakToPeak_"+names[i]]=[(temp[x+1]-temp[x]) for x in range(len(temp)-1)]
                        vars()["peaktimes_"+names[i]]=np.arange(0,T,dt*scale)[vars()["peaks_"+names[i]]]
                        
                    for i in range(len(names)):
                        vars()["firingrate_"+names[i]+"_i"]=firingRateGaussian(spikes_i, vars()["window_"+names[i]], scale)
                        vars()["peaks_"+names[i]+"_i"], vars()["peak_heights_"+names[i]+"_i"]= \
                            find_peaks(vars()["firingrate_"+names[i]+"_i"], height=.1, prominence=.1, distance=10)
                        vars()["peakheights_"+names[i]+"_i"]=vars()["peak_heights_"+names[i]+"_i"]["peak_heights"]
                        temp=vars()["peakheights_"+names[i]+"_i"]
                        vars()["peakToPeak_"+names[i]+"_i"]=[(temp[x+1]-temp[x]) for x in range(len(temp)-1)]
                        vars()["peaktimes_"+names[i]+"_i"]=np.arange(0,T,dt*scale)[vars()["peaks_"+names[i]+"_i"]]
                            
                    # firingrate_shorter_spect=firingRateGaussian(spikes_e, window_shorter, 1)
                    firingrate_shorter_i_spect=firingRateGaussian(spikes_i, window_shorter, 1)
                    
                    #%% See if full PING arises and cutoff as necessary
                    cut=np.argmax(vars()["peaktimes_"+names[0]]>2000)
                    peaktimes_use=vars()["peaktimes_"+names[0]][cut:]
                    peakheights_use=vars()["peakheights_"+names[0]][cut:]
                    
                    if np.any(peakheights_use>40):
                        syncstart=peaktimes_use[np.argmax(peakheights_use>40)]
                        
                        cut2=np.argmax(peakheights_use>40)
                        peaktimes_postsync=peaktimes_use[cut2:]
                        peakheights_postsync=peaktimes_use[cut2:]
                        
                        if np.any(peakheights_postsync<20):
                            syncstop=peaktimes_postsync[np.argmax(peakheights_postsync<20)]
                            syncduration=syncstop-syncstart
                        else:
                            syncstop=T
                            syncduration=syncstop-syncstart
                            
                    else:
                        syncstart=np.nan
                        syncstop=np.nan
                        syncduration=0
            
                    
                    #%% Calculate spectograms, cut for time, and find summed gamma power
                    cut_start=2000/1000
                    if np.isnan(syncstart)==False:
                        cut_stop=syncstart/1000
                    else:
                        cut_stop=T/1000
                    
                    # NON PING DURATION
                    nonPINGduration[Iext_e_index, number_index]=(cut_stop-cut_start)
     
                    # f_e,t_e, Sxx_e=signal.spectrogram(firingrate_shorter_spect, 1000/(dt*1), nperseg=256*25)
                    f_i,t_i, Sxx_i=signal.spectrogram(firingrate_shorter_i_spect, 1000/(dt*1), nperseg=256*25)
                    
                    #%% New IFRH Calculations
                    highIFRHpeaks=np.where(peakheights_shorter_i>7)[0]
                    gamma_grouped=[]
                    i=0
                    imax=len(highIFRHpeaks)
                    track=0
                    for i in range(imax):
                        if i<len(highIFRHpeaks)-1:
                            if highIFRHpeaks[i+1]-highIFRHpeaks[i]==1:
                                track=track+1
                            else:
                                if track>0:
                                    gamma_grouped.append(highIFRHpeaks[i-track:i+1])
                                    track=0
                        else:
                            if track>0:
                                gamma_grouped.append(highIFRHpeaks[i-track:i+1])
                                track=0
                    
                    #If gamma group is long enough, add to counter
                    for gamma in gamma_grouped:
                        if len(gamma)>3:
                            t_min=np.argmax(t_i>(peaktimes_shorter_i[gamma[0]]/1000))
                            t_max=np.argmin(t_i<(peaktimes_shorter_i[gamma[-1]]/1000))-1
                            if t_max<=t_min:
                                t_max=t_min+1
                            if (peaktimes_shorter_i[gamma[-1]]/1000)<cut_stop or np.isnan(syncstart)==True:
                                gammaEvents[Iext_e_index, number_index]+=1                              
                                
        #%% Average over 100 repetitions, output mean and std
        gammaEvents_mean=np.mean(gammaEvents,1) 
        gammaEvents_std=np.std(gammaEvents,1)
        nonPINGduration_mean=np.mean(nonPINGduration,1)
        nonPINGduration_std=np.mean(nonPINGduration,1)
        
        for Iext_e_index, Iext_e in enumerate(Iext_e_range):           
            mystr=[Iext_e, gammaEvents_mean[Iext_e_index], gammaEvents_std[Iext_e_index], nonPINGduration_mean[Iext_e_index], nonPINGduration_std[Iext_e_index]]
            outputwriter.writerow(mystr)
            csvfile.flush()

    
    
    
    
    
    
    
    
    
    
    
    

