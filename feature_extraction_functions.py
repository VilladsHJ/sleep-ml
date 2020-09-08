# -*- coding: utf-8 -*-
"""
This script contains the functions for feature extraction

@author: Villads

"""


import numpy as np
from scipy.signal import cwt, ricker
from scipy.fftpack import fft




def correlation_feat(x, y):
    """
    Function that calculates the linear correlation between two signals

    Inputs:
        x           (First signal array for correlation)
        y           (Second signal array for correlation)

    Output:
        cor_feat    (Pearsons correlation constant)


    """
    cor_feat = np.corrcoef(x, y)
    # cor_feat = np.correlate(x, y)
    cor_feat = cor_feat[0,1]
    return cor_feat

def fourier_peak(signal, sampling_rate, N):
    """
    Function the finds the dominant frequency in a signal
    
    Input:
        Signal          (Signal for computations)
        Sampling rate   (The sampling frequency fs)
        N               (Length of the signal)
        
    Output:
        xf              (x-axis datapoints the fourier transformed data)
        y_axis          (the real-values of the fourier transform)
    """
    # Function that returns the dominant frequency in the data via fft

    # Number of sample points

    # sample spacing
    
    y = signal
    yf = fft(y, N)
    xf = np.linspace(0, sampling_rate/2, int(N/2))

    y_axis = 2.0/N * np.abs(yf[:N//2])

    return xf, y_axis

def spectral_density_feature(normal_segment, segment):
    """
    Calculates the ratior between the sum of one power spectral density and
    another power spectral density
    
    Input:
        normal_segment  (The normal segment for feature extraction)
        segment         (The segment that is currently being investigated)
    
    Output:
        summed_feature  (The feature called rPSD)
        peak_f_norm     (The peak frequency of the normal segment) 
        peak_f_segment  (The peak frequency of the segment being investigated)
    """
    
    sampling_freq = 10
    nyquist = sampling_freq/2
    x_dat, fourier_p = fourier_peak(normal_segment, sampling_freq, 2400)
    
    for r in range(len(fourier_p)):
        if fourier_p[r] == fourier_p[10:len(fourier_p)].max():
            spot = r
        
    peak_f_norm = nyquist * spot / len(fourier_p)
    
    left_side_norm = (peak_f_norm-0.1)*len(fourier_p)/nyquist
    right_side_norm = (peak_f_norm+0.1)*len(fourier_p)/nyquist
    
    summed_feature_norm = sum(fourier_p[int(left_side_norm):int(right_side_norm)])
    
    sampling_freq = 10
    nyquist = sampling_freq/2
    x_dat, fourier_p = fourier_peak(segment, sampling_freq, len(normal_segment))
    
    for r in range(len(fourier_p)):
        if fourier_p[r] == fourier_p[10:len(fourier_p)].max():
            spot = r
        
    peak_f_segment = nyquist * spot / len(fourier_p)
    
    if summed_feature_norm == 0:
        summed_feature = 0
    else:
        summed_feature = sum(fourier_p[int(left_side_norm):int(right_side_norm)])/summed_feature_norm
            
    return summed_feature, peak_f_norm, peak_f_segment


def arclen_function(abs_derived_signal):
    
    """
    Input:
        abs_derived_signal  (the absolute of the derivate 
                             of the signal |f'(x)|)
    Output:
        arclength           (The arc length of the signal)
    """
    
    arclength = np.trapz(abs_derived_signal)
    
    return arclength

def wavelet_feature(envelope_data, start_index, stop_index):
    
    """
    Wavelet feature
    """
    sig  = envelope_data
    widths = np.arange(1, 601)
    cwtmatr = cwt(sig, ricker, widths)
    
    # Wavelet feature
    # integration_array = []
    # wavelet_inters = []
    # for i in range(len(start_index)):
    #     maxi = []
    #     for k in range(len(cwtmatr[:,int(start_index[i]):int(stop_index[i])])):
    #         maxi =  np.append(maxi, np.max(cwtmatr[k,int(start_index[i]):int(stop_index[i])]))
    #     row = np.where(max(maxi) == cwtmatr[:,int(start_index[i]):int(stop_index[i])])[0]
    #     len_div = len(cwtmatr[row[0],int(start_index[i]):int(stop_index[i])])
    #     int_val = np.trapz(cwtmatr[row[0], int(start_index[i]):int(stop_index[i])],
    #                        np.arange(1,len(cwtmatr[row[0],int(start_index[i]):int(stop_index[i])])+1))/len_div
    #     integration_array = np.append(integration_array, row)
    #     wavelet_inters = np.append(wavelet_inters, int_val)
    
    row_dict = {}
    rows = [10,20,30,40,50]
    for q in rows:
        wavelet_inters = []
        for i in range(len(start_index)):
    
            len_div = len(cwtmatr[q,int(start_index[i]):int(stop_index[i])])
    
            int_val = np.trapz(cwtmatr[q, int(start_index[i]):int(stop_index[i])],
                               np.arange(1,len(cwtmatr[q,int(start_index[i]):int(stop_index[i])])+1))/len_div
    
            # integration_array = np.append(integration_array, q)
    
            wavelet_inters = np.append(wavelet_inters, int_val)
    
        row_update = {q:wavelet_inters}
        row_dict.update(row_update)
        
    return row_dict #wavelet_inters
