# -*- coding: utf-8 -*-
"""
Main functions for preprocessing.
Input files required: SHHS database {recordingname}.edf
                      SHHS annotation sheet {recordingname}.xml
@author: Villads
"""


import numpy as np

from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks
from feature_extraction_functions import fourier_peak
import matplotlib.pyplot as plt
# This script contains all functions recarding prepocessing

# Signal processing:

# Function of the backwards forward butterworth bandpass filter.

def frequency_criteria(data_array, ranger, plotting=False):
    
    """
    The frequency criteria the functions finds the dominant frequency in a 
    data array.
    
    Input:
        data_array  (data to be processed fx. nasal airflow signal)
        ranger      (array containing all sample points in wake segments)
        plotting    (for plotting purposes)
        
    Output:
        peak_f      (the peak frequency)
    """
    array_fft = data_array.copy()
    
    for r in range(len(array_fft)):
        if r in ranger:
            array_fft[r] = 0

    sampling_freq = 10
    nyquist = sampling_freq/2
    x_dat, fourier_p = fourier_peak(data_array, sampling_freq, len(data_array))

    for r in range(len(fourier_p)):
        if fourier_p[r] == fourier_p[10:len(fourier_p)].max():
            spot = r
        
    peak_f = nyquist * spot / len(fourier_p)

    
    if plotting:
        fig, ax = plt.subplots()
        ax.axvline(x=peak_f - 0.1, color='r')
        ax.axvline(x=peak_f + 0.1, color='r')
        ax.plot(x_dat, fourier_p)
        plt.show()
    return peak_f

def butter_bandpass(lowcut, highcut, fs, order=5):
    
    """
    Inputs: 
            low cut
            high cut
            sampling frequency (fs) 
            Filter order
            
     Output: Coefficients for the filter 
     """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], output='sos', btype='band')
    
    return sos

def butter_bandpass_filter_filt(data, lowcut, highcut, fs, order=5):
    
    """ 
    Inputs:
            Data array
            low cut
            high cut
            sampling frequency (fs)
            filter type, default: bandpass
            
    Imported functions: butter_bandpass for filter coefficients
    
    Output: Filtered data
    """
    
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    
    y = sosfiltfilt(sos, data)
    
    # from scipy import signal
    
    # w, h = signal.sosfreqz(sos)
    
    # db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    # plt.plot(w, db)
    # plt.ylim(-75, 5)
    # plt.grid(True)
    # plt.yticks([0, -20, -40, -60])
    # plt.ylabel('Gain [dB]')
    # plt.title('Frequency Response')
    # plt.xlabel('Frequency [Hz]')
    return y


def outlier_removal(signal, percentile_f):

    """This function replaces outliers outside a set percentile
    
    Inputs:
        Nasal airflow signal
        Percentile to replace.
        
    Output:
        Signal with values outside set percentile replaced
    """

    # Finding the set percentile
    
    percentile_n = np.percentile(signal, percentile_f, interpolation='nearest')

    # Replacing outliers 
    counter = 0
    for i in range(len(signal)):
        if signal[i] > percentile_n:
            counter +=1
            signal[i] = percentile_n
        elif signal[i] < -percentile_n:
            counter +=1
            signal[i] = -percentile_n
    print('Samples outside of percentile: {}'.format(counter))
    return signal


def preprocessor(data_array):    
    
    """
    This functions replace values outside the 99.5th percentile 
    and then filter the signal by a 5th order backwards forward
    butter worth bandpass filter
    
    Input: 
        Signal (For example nasal airflow signal)
    
    Output:
        Preprocessed signal
    """
    
    percentile = 99.5

    data_processed = outlier_removal(data_array, percentile)
    low_cut = 0.01
    high_cut = 2
    data_bandpass = butter_bandpass_filter_filt(data_processed, low_cut, high_cut, 10, 5)  # bandpass filtering

    
    return data_bandpass


def HR_SA02_preprocessor(signal, lower_bound, upper_bound):
    
    """
    This function is made for the HR and Sa02 signal.
    The functions replaces values outside the lower and
    the upper band and then smooth the signal using the 
    Savitzky Golay filter.
    
    Input:
        Signal
        lower bound
        upper bound
    
    Output:
        Preprocessed and filtered signal
    
    """
    
    
    for i in range(1,len(signal)):
        if signal[i] < lower_bound  or signal[i] > upper_bound:
            signal[i] = signal[i-1]
        
    signal = savgol_filter(signal, 31, 2)
        
    return signal

def standardizer(data):
    
    """
    Function that standardize the data it is given by first subtracting the 
    mean and the dividing by the data std.
    
    Input:
        signal
    
    Output:
        Standardized signal
    """
    
    new_data = (data-np.mean(data))/np.std(data)
    return new_data

def segment_normalizer(data_processed, wake_start, wake_stop):
    
    """
    Segment standardizing function that standardize the wake segments of the
    nasal airflow signal and the segments between wake segments.
    
    Input: 
        Preprocessed nasal airflow signal
        Start index values for the wake segments
        Stop index values for the wake segments
        
    Output:
        Piece-wise standardized nasal airflow signal
    """
    
    # for i in range(len(wake_start)-1):
    #     data_processed[int(wake_stop[i]):int(wake_start[i+1])] = standardizer(data_processed[int(wake_stop[i]):int(wake_start[i+1])])
    #     data_processed[int(wake_start[i]):int(wake_stop[i])] = standardizer(data_processed[int(wake_start[i]):int(wake_stop[i])])

    # data_processed[int(wake_start[-1]):int(len(data_processed))] = standardizer(data_processed[int(wake_start[-1]):int(len(data_processed))])
    # return data_processed
    
    for i in range(len(wake_start)-1):
        data_processed[int(wake_stop[i]):int(wake_start[i+1])] = standardizer(data_processed[int(wake_stop[i]):int(wake_start[i+1])])
        data_processed[int(wake_start[i]):int(wake_stop[i])] = standardizer(data_processed[int(wake_start[i]):int(wake_stop[i])])

    data_processed[int(wake_start[-1]):int(wake_stop[-1])] = standardizer(data_processed[int(wake_start[-1]):int(wake_stop[-1])])
    
    return data_processed



def movement_artifacts(data_array, number):
    
    """
    Movement artifact finder. This functions finds high frequency noise in
    belt signals.
    Input:
        data_array  (belt signals)
        number      (the recording number from the SHHS database)
    
    Output:
        wake_start  (updated wake start indicies 
                     that contain movement segments)
        wake_stop   (updated wake stop indices 
                     that contain movement artifacts)
    """    
    
    from Data_loader import annotation_finder

    low_cut = 3
    high_cut = 4.9
    band = butter_bandpass_filter_filt(data_array, low_cut, high_cut, 10, 5)  # bandpass filtering
    
    
    wake_start, wake_stop = annotation_finder(number, parameter='wake')
    

    ranger = []
    for k in range(len(wake_start)):
        ranger = np.append(ranger, range(int(wake_start[k]), int(wake_stop[k])))
        
    peaks, props = find_peaks(band, height=0.25, distance=400)
    for i in peaks:
        if i-100 not in ranger and i+200 not in ranger:
            if i + 200 > len(data_array):
                wake_start = np.append(wake_start, i-100)
                wake_stop = np.append(wake_stop, len(data_array))
            else: 
                wake_start = np.append(wake_start, i-100)
                wake_stop = np.append(wake_stop, i+200)


            
    for t in range(1,len(wake_start)-1):
        if wake_start[t] < wake_start[t+1] < wake_stop[t]:
            print('found one start: '+str(t+1)+' '+str(wake_start[t+1]))
        if wake_start[t] < wake_stop[t-1] < wake_stop[t]:
            print('found one stop: '+str(t-1)+' '+str(wake_stop[t-1]))
            
    return wake_start, wake_stop