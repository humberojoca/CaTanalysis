# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:29:53 2022

@author: hjoca
"""

import sys
import pims
import tkinter as tk
import matplotlib.pyplot as plt
import skimage.filters as filters
import skimage.morphology as morphology
import numpy as np
from tkinter import filedialog
from os import path
from pp_style import pps_xy
from scipy import signal

from analysis import find_ca  # analysis.py


# Image Parameters
fluo_index = 0  # Calcium fluorescence channel (start at 0)
linescan_speed = 1.87  # Scan time for each line (in ms)
ls_sampling = 1/linescan_speed*1000  # Calculate sampling rate
filter_kernelsize = 5  # NxN square for median filter
max_ff0 = 8  # Max F/F0 value for images
show_images = True  # True or False
export_csv = True  # True or False
analyze_synchrony = True  # True or False

# Select file window
root = tk.Tk()
filename = filedialog.askopenfilenames(parent=root,
                                       filetypes=[('TIFF Image', '*.*')],
                                       title='Choose an TIFF Image')
root.withdraw()

if filename is None:
    sys.exit('No file selected!')

for fn in filename:
    # Open file and Select channels
    image = pims.Bioformats(fn)
    ca_fluo = np.transpose(image[fluo_index])  # Calcium fluorescense channel
    meta = image.metadata  # Metadata from file
    voxel = meta.PixelsPhysicalSizeX(0)  # pixel size (um)
    # Median Filter
    ca_filtered = filters.median(ca_fluo, morphology.square(filter_kernelsize))

    # Select cell limits within the linescan image
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    plt.imshow(ca_filtered, cmap='inferno', vmin=0,
               vmax=np.mean(ca_filtered)*10, aspect='auto')
    plt.xlim(0, 2000)
    plt.title("Select cell limits (2 clicks)")
    ylimits = plt.ginput(n=2, timeout=60)  # Save two click with the limits
    pps_xy()
    plt.close(fig)
    # Crop array using points selected (y-axis)
    ca_corrected = ca_filtered[int(ylimits[0][1]):int(ylimits[1][1])]

    # Select baseline region for F/F0 normalization
    ca_flat = np.mean(ca_corrected, 0)  # Create a flat vector x-axis mean
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    plt.plot(ca_flat)
    plt.xlim(0, 2000)
    plt.title("Select baseline region (2 clicks)")
    xlimits = plt.ginput(n=2, timeout=60)  # Save two click with the baseline
    pps_xy()
    plt.close(fig)

    # F/F0 normalization - Average of baseline region
    ca_bkg = np.mean(ca_flat[int(xlimits[0][0]):int(xlimits[1][0])])
    ca_signal = ca_flat/ca_bkg  # Normalize flat vector
    ca_signal = signal.savgol_filter(ca_signal, 21, 3)  # Smooth vector
    ca_norm = ca_corrected/ca_bkg  # Normalize 2D image

    # Create time/space axis (Time in ms / Space in um)
    taxis = np.linspace(0, linescan_speed*np.size(ca_signal), np.size(ca_signal))
    xaxis = np.linspace(0,voxel*np.size(ca_norm[:,0]),np.size(ca_norm[:,0])) 

    # Use find_ca function to find and analyze peaks
    if analyze_synchrony:
        ca_data,sy_profile = find_ca(ca_signal, ls_sampling, ca_norm)
        sy_mean = np.array([xaxis,np.nanmean(sy_profile,0)]).T
    else:
        ca_data = find_ca(ca_signal, ls_sampling)

    # Show Images (if enable)
    if show_images:
        plt.figure(figsize=(16, 5), constrained_layout=True)
        plt.plot(taxis, ca_signal)
        plt.plot(taxis[np.int32(ca_data[1:, 0])],
                 ca_signal[np.int32(ca_data[1:, 0])], label='Begin', marker='o',
                 color='r', fillstyle='none', linestyle='none')
        plt.plot(taxis[np.int32(ca_data[1:, 1])],
                 ca_signal[np.int32(ca_data[1:, 1])], label='End', marker='o',
                 color='g', fillstyle='none', linestyle='none')
        plt.xlim(0, 3000)
        plt.xlabel('Time (ms)')
        plt.ylabel('Ca Signal (F/F0)')
        pps_xy()
        plt.figure(figsize=(16, 5), constrained_layout=True)
        plt.imshow(ca_norm, cmap='inferno', vmin=0, vmax=max_ff0, aspect='auto',
                   extent=[0,linescan_speed*np.size(ca_signal),0,voxel*np.size(ca_norm[:,0])])
        plt.xlabel('Time (ms)')
        plt.ylabel('Distance (um)')
        plt.xlim(0, 3000)
        cbar = plt.colorbar(orientation='horizontal', shrink=0.6)
        cbar.set_label('F/F0')
        pps_xy()

    if export_csv:
        # Export analyzed data to .csv and .npy files
        csvheader = ['Begin', 'End', 'Freq (Hz)', 'Ca Baseline', 'Ca Peak']
        csvheader = csvheader + ['Ca Amplitude', 'Ca Rise Time 10-90%(ms)']
        csvheader = csvheader + ['Ca Decay time 50%(ms)', 'Ca Decay time 90%(ms)']
        if analyze_synchrony: 
            csvheader = csvheader + ['Delay(ms)', 'SD(ms)', 'SI']

        csvheader = ','.join(csvheader)
        csvpath = path.dirname(fn)
        csvfile = path.basename(fn)
        csvfile = path.splitext(csvfile)
        
        if analyze_synchrony:
            sy_file = csvfile[0] + '_sync.csv'
        npzfile = csvfile[0] + '.npz'
        csvfile = csvfile[0] + '.csv'
        
        if analyze_synchrony:
            sy_file = csvpath + '/' + sy_file     
        csvfile = csvpath + '/' + csvfile
        npzfile = csvpath + '/' + npzfile
      
        np.savez_compressed(npzfile,signal=ca_signal,data=ca_data,sampling=ls_sampling,csvheader=csvheader)
        np.savetxt(csvfile,ca_data,delimiter=',',header=csvheader,fmt='%1.3f')
        np.savetxt(sy_file,sy_mean,delimiter=',',header='Distance (um),Time (ms)',fmt='%1.3f')
