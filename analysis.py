#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:16:51 2018

@author: hjoca
"""
import peakutils
import numpy as np 
import matplotlib.pyplot as plt

# Function to analyze single contractions:
#   Use the sarcomere length trace (sl) with its derivate (dsl).
#   Also include the sampling rate (sampling).
#   Add contraction id for better identification (cont) - Optional
def analysis_sl(sl,dsl,sampling,cont=1):
    bl = np.mean(sl[0:10])  # 10 points after the trigger for baseline
    # Detect peaks (inverted) - Only detect upward peaks
    pkt = peakutils.indexes(-sl, thres=0.5, min_dist=50)

    # Stop the analysis if more than 1 peak is detected
    if not len(pkt) == 1:
        print('Invalid Contraction #', cont)
        output = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        return output

    pkt = int(pkt)
    peak = sl[pkt]      # Get the peak value
    amp = bl - peak     # Amplitude
    fs = (amp/bl)*100   # Fractional Shortning
    cpeak = dsl.min()   # Get the contraction peak from the derivate data
    rpeak = dsl.max()   # Get the relaxation peak from the derivate data

    r10 = np.array(np.where(sl <= (bl-(amp*0.1))))  # Find 10% Rise
    r90 = np.array(np.where(sl <= (bl-(amp*0.9))))  # Find 90% Rise

    # Calcutate Rise Time (10-90%)
    if r10.size and r90.size:
        r10 = int(r10[0, 0])
        r90 = int(r90[0, 0])
        rt = ((r90-r10)*(1/sampling))*1000
        if rt < 10:
            rt = np.nan
    else:
        rt = np.nan

    i50 = np.array(np.where(sl[pkt:] >= (peak+(amp*0.5))))  # Find 50% Decay
    i90 = np.array(np.where(sl[pkt:] >= (peak+(amp*0.9))))  # Find 90% Decay

    # Calculate 50% Decay time
    if i50.size:
        i50 = int(i50[0, 0])
        d50 = (i50*(1/sampling))*1000  # in ms
        if d50 < 10:
            d50 = np.nan
    else:
        d50 = np.nan
        
    # Calculate 90% Decay time
    if i90.size:
        i90 = int(i90[0, 0])
        d90 = (i90*(1/sampling))*1000  # in ms
        if d90 < 50:
            d90 = np.nan
    else:
        d90 = np.nan

    # Create output array and return it
    output = np.array([bl,peak,fs,rt,d50,d90,cpeak,rpeak])
    return output

# Function to analyze single calcium transients:
#   Use the calcium signal trace (ca) with its sampling rate (sampling).
#   Add contraction id for better identification (cont) - Optional
#   Use mode = 'single' (Default) or 'ratio' for
#   Single wavelength or Ratiometric transient, respectively.
def analysis_ca(ca, sampling, cont=1, mode='single'):
    if mode == 'ratio':
        cathres = 0.9
    elif mode == 'single':
        cathres = 0.8
    else:
        print('Invalid mode selected! Using Single wavelength parameters!')
        cathres = 0.8

    bl = np.mean(ca[0:10])  # 10 points after the trigger for baseline
    pkt = peakutils.indexes(ca, thres=cathres, min_dist=100)  # Detect peaks

    # Stop the analysis if more than 1 peak is detected
    if not len(pkt) == 1:
        print('Invalid Transient #', cont)
        output = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        return output

    pkt = int(pkt)
    peak = ca[pkt]      # Get the peak value
    amp = peak - bl     # Amplitude

    r10 = np.array(np.where(ca >= (bl+(amp*0.1))))  # Find 10% Rise
    r90 = np.array(np.where(ca >= (bl+(amp*0.9))))  # Find 90% Rise
    
    # Calcutate Rise Time (10-90%)
    if r10.size and r90.size:
        r10 = int(r10[0, 0])
        r90 = int(r90[0, 0])
        rt = ((r90-r10)*(1/sampling))*1000
        if rt < 10:
            rt = np.nan
    else:
        rt = np.nan

    i50 = np.array(np.where(ca[pkt:] <= (peak-(amp*0.5))))  # Find 50% Decay
    i90 = np.array(np.where(ca[pkt:] <= (peak-(amp*0.9))))  # Find 90% Decay

    # Calculate 50% Decay time
    if i50.size:
        i50 = int(i50[0, 0])
        d50 = (i50*(1/sampling))*1000  # in ms
        if d50 < 10:
            d50 = np.nan
    else:
        d50 = np.nan
     
    # Calculate 90% Decay time
    if i90.size:
        i90 = int(i90[0, 0])
        d90 = (i90*(1/sampling))*1000  # in ms
        if d90 < 50:
            d90 = np.nan
    else:
        d90 = np.nan

    # Create output array and return it.
    output = np.array([bl,peak,amp,rt,d50,d90])
    return output

# Function to find and analyze calcium transients:
#   Use the signal trace (signal) with its sampling rate (sampling).
#   Include linescan (ls) image for delay and syncrony analysis.
#   Use mode = 'single' (Default) or 'ratio' for
#   Single wavelength or Ratiometric transient, respectively.
def find_ca(ca, sampling, ls=False, mode='single'):
    if mode == 'ratio':
        cathres = 0.6
    elif mode == 'single':
        cathres = 0.45
    else:
        print('Invalid mode selected! Using Single wavelength parameters!')
        cathres = 0.45

    pkts = peakutils.indexes(ca, thres=cathres, min_dist=200)  # Detect peaks
    n = len(pkts)
    # Stop the analysis if more than 1 peak is detected
    if n < 1:
        print('No transient found')
        return
    elif n > 1:
        remove_last = True  # Skip last transient if have more than 1
        if ls is False:
            data = np.zeros((pkts.size-1, 9))
        else:
            data = np.zeros((pkts.size-1, 12))
            delay_profile = []
        n = n - 1
    else:
        remove_last = False
        if ls is False:
            data = np.zeros((1, 9))
        else:
            data = np.zeros((1, 12))
            delay_profile = []
    
    for l in range(pkts.size):
        if pkts[l] < 50:  # if the peak occurs in less than 50 points
            data[l,:] = np.nan
            continue
        elif l >= n:
            if remove_last: continue  # Skip last transient
        
        if l < n and n > 1:
            freq = round(sampling / (pkts[l+1] - pkts[l]))
            if freq < 1: freq = 1 
        else:
            freq = 1

        duration = int(((1000 / freq) * 0.8) / (1/sampling*1000))
        crop = ca[pkts[l]-50:pkts[l]+duration]
        data[l, 0] = pkts[l]-50
        data[l, 1] = pkts[l]+duration
        if ls is False:
            data[l, 2] = freq
            data[l, 3:9] = analysis_ca(crop, sampling, l, mode)       
        else:
            crop_ls = ls[:,int(data[l, 0]):int(data[l, 1])]
            data[l, 2] = freq
            data[l, 3:12],profile = analysis_ca_sync(crop, sampling, crop_ls)
            delay_profile.append(profile)
    # Return the data
    if ls is False:
        return data
    else:
        return data,np.asarray(delay_profile)

# Function to analyze single calcium transients and syncrony:
#   Use the calcium signal trace (ca) with its sampling rate (sampling).
#   Include linescan (ls) image for delay and syncrony analysis.
#   ampd - Amplitude % which delay will be calculated (default - 50%) 
#   Only for Single wavelength
def analysis_ca_sync(ca, sampling, ls, ampd = 50):
    cathres = 0.9 #  Threshold for calcium peak detection
    bl = np.mean(ca[0:10])  # 10 points after the trigger for baseline
    pkt = peakutils.indexes(ca, thres=cathres, min_dist=100)  # Detect peaks
    
    # Stop the analysis if more than 1 peak is detected
    if not len(pkt) == 1:
        print('Invalid Transient')
        output = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
        return output

    pkt = int(pkt)
    peak = ca[pkt]      # Get the peak value
    amp = peak - bl     # Amplitude

    r10 = np.array(np.where(ca >= (bl+(amp*0.1))))  # Find 10% Rise
    r90 = np.array(np.where(ca >= (bl+(amp*0.9))))  # Find 90% Rise

    # Calcutate Rise Time (10-90%)
    if r10.size and r90.size:
        r10 = int(r10[0, 0])
        r90 = int(r90[0, 0])
        rt = ((r90-r10)*(1/sampling))*1000
        if rt < 10:
            rt = np.nan
    else:
        rt = np.nan

    i50 = np.array(np.where(ca[pkt:] <= (peak-(amp*0.5))))  # Find 50% Decay
    i90 = np.array(np.where(ca[pkt:] <= (peak-(amp*0.9))))  # Find 90% Decay

    # Calculate 50% Decay time
    if i50.size:
        i50 = int(i50[0, 0])
        d50 = (i50*(1/sampling))*1000  # in ms
        if d50 < 10:
            d50 = np.nan
    else:
        d50 = np.nan
     
    # Calculate 90% Decay time
    if i90.size:
        i90 = int(i90[0, 0])
        d90 = (i90*(1/sampling))*1000  # in ms
        if d90 < 50:
            d90 = np.nan
    else:
        d90 = np.nan
    
    # Syncrony/Delay analysis
    sy_max = np.max(ls, 1)  #Max intensity for each row
    sy_bl = np.mean(ls[:, 0:10], 1)  #  Baseline for each row
    sy_amp = sy_max - sy_bl  #  Amplitude for each row
    sy_thres = sy_max - sy_amp*((100-ampd)/100)  # Amplitude for delay calculations
    delay_times = np.zeros((ls.shape[0]))
    sy_lineprofile = np.zeros((ls.shape[0]))
    
    # Calculate delay for each row
    for r in range(ls.shape[0]):
        delay = np.array(np.where(ls[r] >= sy_thres[r]))
        if delay.size:
            delay = int(delay[0, 0])
            delay_times[r] = (delay*(1/sampling))*1000  # in ms
            if delay_times[r] < 10:
                delay_times[r] = np.nan
        else:
            delay_times[r] = np.nan
    
    # Calculate synchrony index
    delay_mean = np.nanmean(delay_times)
    delay_std = np.nanstd(delay_times)
    SI = delay_std / delay_mean 

    # Create output array and return it.
    output = np.array([bl,peak,amp,rt,d50,d90,delay_mean,delay_std,SI])
    return output,delay_times