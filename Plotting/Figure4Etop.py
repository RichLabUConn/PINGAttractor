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
import pandas as pd
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
from scipy import stats


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
        if np.mod(i,100)==0:
            print('SD= ', sd, '; Cell= ', i)
        timeseries=convert_spiketimes(spikes[i], T/1000, 1000/dt/scale)
        gaussian=conv_gaussian_NEW(timeseries[1], 1000/dt/scale, sd)
        gaussian_sum=gaussian_sum+gaussian
    # gaussian_sum=gaussian_sum/len(spikes)
    
    return gaussian_sum
        
        















#%% Read desired data
E_GABA_list=[75, 73]
# E_GABA_list=[75]
NoiseAmp_list=[0, 2000]
Iext_list=[25, 27.5, 30, 32.5, 35]
gammaEvents_mean=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
gammaEvents_std=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
nonPING_mean=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]
nonPING_std=[[] for x in range(len(E_GABA_list)*len(NoiseAmp_list))]

listindex=0
for E_GABA in E_GABA_list:
    for NoiseAmp in NoiseAmp_list:
        csvstr='FinalFigure4Quants/GammaEvents_NonPINGDuration_EGABA%d_NoiseAmp%d.csv' %(E_GABA, NoiseAmp)
        temp=np.genfromtxt(csvstr, delimiter=',')
        gammaEvents_mean[listindex]=temp[:len(Iext_list),1].tolist()
        gammaEvents_std[listindex]=temp[:len(Iext_list),2].tolist()
        nonPING_mean[listindex]=temp[:len(Iext_list),3].tolist()
        nonPING_std[listindex]=temp[:len(Iext_list),4].tolist()
        listindex +=1

gammaEvents_mean=np.array(gammaEvents_mean)/18
gammaEvents_std=np.array(gammaEvents_std)/18
nonPING_mean=np.array(nonPING_mean)/18
nonPING_std=np.array(nonPING_std)

gammaEventsfreq_mean=(gammaEvents_mean)/np.array(nonPING_mean)
gammaEventsfreq_std=(gammaEvents_std)/nonPING_std

NoiseAmp_to_VoltageSD=dict([(0,0),(1,0.023),(20,0.108), (70, 0.199), (200, 0.325), (600, 0.581), (1000, 0.750), (2000,1.087), (4000,1.439), (6000, 1.842), (8000, 2.234)])
 
        
#%% New Plotting 
cmap=cm.get_cmap('jet')

plt.rcParams['font.size']=8
fig, ax=plt.subplots()
fig.set_size_inches(2.5,2)

listindex=0
for E_GABA in E_GABA_list:
    for NoiseAmp in NoiseAmp_list:
        str1='-%d, %1.3f' % (E_GABA, NoiseAmp_to_VoltageSD[NoiseAmp])
        ax.plot(Iext_list, gammaEventsfreq_mean[listindex, :len(Iext_list)], color=cmap(listindex/(len(gammaEvents_mean)-1)), label=str1, lw=2)
        # ax.fill_between(Iext_list, nonPING_mean[listindex, :len(Iext_list)]+nonPING_std[listindex, :len(Iext_list)],
        #                 nonPING_mean[listindex, :len(Iext_list)]-nonPING_std[listindex, :len(Iext_list)], color=cmap(listindex/(len(gammaEvents_mean)-1)), alpha=.3)
        
        listindex +=1
        
ax.set_ylabel('Gamma event rate during \n  non-PING activity (Hz)')

ax.set_xlabel(r'$I_{ext}$ ($\mu$ A)')
# ax.set_xticklabels([])

ax.spines[['right', 'top']].set_visible(False)
ax.legend(title=r'$E_{\text{GABA}}$ (mV), $SD_V$ (mV)', loc='right', bbox_to_anchor=(1.8, .5),ncol=1)


filenamestr='PaperFigures/Figure4/GammaRateNormalized.png'
plt.savefig(filenamestr, dpi=600, format='png', bbox_inches='tight')



















