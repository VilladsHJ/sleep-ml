# -*- coding: utf-8 -*-
"""

This script contains all functions for the segmentation algorithm and 
oxygen saturation delay




@author: Villads
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# Cubic spline interpolation

def cubic_spline_int(array_x, data, new_length):
    
    """
    This functions generates a cubic spline interpolation
    
    Input:
        array_x         (array of indices on the x-axis)
        data            (data array corresponding to the values of the y-axis)
        new_length      (The desired interpolation lenght)
    """
    
    x_data = array_x
    cubic_inter = interp1d(x_data, data, kind='cubic')
    new_x = np.arange(0, new_length)
    new_cubic = cubic_inter(new_x)
    return new_cubic


def trend_line_function(array_of_interest, width, height):
    
    """
    This function generates a trendline over the signal
    The trendline corresponds that corresponds an envelope on the fine
    structure of the signal
    
    Input:
        array_of_interest   (signal to generate trendline on)
        width               (parameter that sets the how fine the structure
                             of the envelope should be)
        height              (paramter that sets a minimum of peak height in
                             the fine structure)
    """

    local_idx, properties = find_peaks(array_of_interest, width=width, height=height)
    local_max = properties['peak_heights']
    # Inserting starting value for interpolation
    if local_idx[0] != 0:
        local_max = np.insert(local_max, 0, array_of_interest[0])
        local_idx = np.insert(local_idx, 0, 0)

    # adding ending value for interpolation
    local_max = np.append(local_max, array_of_interest[-1])
    local_idx = np.append(local_idx, len(array_of_interest))
    
    trend_line = cubic_spline_int(local_idx, local_max, len(array_of_interest))
    
    return trend_line



def enveloper(feature_sample_output):
    
    """
    This functions generates the envelope of the input
    
    Input: 
        feauture_sample_output (signal to generate envelope from)
        
    output:
        smooth_new_trend       (Envelope of the signal)
    """

    new_data = feature_sample_output.copy()
    trend_line = trend_line_function(new_data, 2, 0)
    new_trend = savgol_filter(trend_line, 301, 2)
    smooth_new_trend = savgol_filter(new_trend.copy(), 201, 2)

    
    for i in range(len(smooth_new_trend)):
        if smooth_new_trend[i] < 0:
            smooth_new_trend[i] == 0
                
    
    return smooth_new_trend


def segmentation_function(smooth_new_trend, wake_start, wake_stop, data_processed):
    """
    
    Input:
        smooth_new_trend   (Envelope of nasal airflow signal)
        wake_start         (Array of start indices of wake segments)
        wake_stop          (Array of stop indices of wake segments)
        data_processed     (Preprocessed nasal airflow signal)
    
    Output:
        segments_start      (Start indices of segments)
        segments_stop       (Stop indices of segments)
        
            
    """
    # peaks in this signal are downward slopes
    diff_trend_raw = np.diff(smooth_new_trend,1)*-1
    
    
    diff_trend = abs(np.diff(smooth_new_trend,1))
    diff_trend = np.append(diff_trend, 0)
    
    
    diff_trend_feature = np.diff(smooth_new_trend,1)
    diff_trend_feature = (diff_trend_feature-np.mean(diff_trend_feature))/np.std(diff_trend_feature)
    
    
    peak_x, properties = find_peaks(diff_trend, width=80, height=0) 
    
    peak_x_half, _ =  find_peaks(diff_trend_raw, width=80, height=0) 

    
    if len(peak_x) % 2 != 0:
        peak_x = np.insert(peak_x, 0, 1)

    for k in range(len(peak_x)):
        if k != 0:
            if peak_x[k] in wake_start:
                print('moved one (start)')
                peak_x[k] = peak_x[k]-1
            elif peak_x[k] in wake_stop:
                peak_x[k] = peak_x[k]+1
                print('moved one (stop)')
    
    for k in range(len(peak_x_half)):
        if k != 0:
            if peak_x_half[k] in wake_start:
                # print('moved one (start)')
                peak_x_half[k] = peak_x_half[k]-1
            elif peak_x_half[k] in wake_stop:
                peak_x_half[k] = peak_x_half[k]+1
                # print('moved one (stop)')
    
    peak_x = np.append(peak_x, wake_start)
    
    peak_x = np.append(peak_x, wake_stop)
    
    peak_x = np.append(peak_x, wake_start[1:len(wake_start)]-1)
    
    peak_x = np.append(peak_x, wake_stop[0:len(wake_stop)-1]+1)
    
    event_idx = np.sort(np.unique(peak_x))

    segments_start = []
    segments_stop = []
    for i in range(0,len(event_idx)-1):
            if event_idx[i+1] - event_idx[i] < 80:
                continue
            else:
                segments_start = np.append(segments_start, event_idx[i])
                segments_stop = np.append(segments_stop, event_idx[i+1])
    
    
    segments_all =[]
    segments_all = np.append(segments_all, segments_start)
    segments_all = np.append(segments_all, segments_stop)
    segments_all = np.unique(segments_all)
    segments_half = []
    for i in range(len(peak_x_half)):
        if peak_x_half[i] in segments_all:
            segments_half = np.append(segments_half, peak_x_half[i])
    
    return segments_start, segments_stop, segments_half



# def piecewise_lin_oxy_delay(sa02, HR, start_index, stop_index):
    
    
#     """
#     This function finds the piece-wise linear oxygen delay.
    
#     Input:
#         sa02            (oxygen saturation signal)
#         HR              (heart rate signal)
#         start_index     (array of start indices from 
#                          segmentation algorithm)
#         stop_index      (array of stop indicies from 
#                          segmentation algortihm)
        
#     output:
#         start_sat_index (array of start indices from 
#                          segmentation algorithm + delay)
#         stop_sat_index  (array of start indices from 
#                          segmentation algorithm + delay)
#         mover           (array of all delays to each entry in start_index)
#     """
    
#     # Finding peaks to find overlap for oxygen saturation
#     peaks_top, peaks_ys = find_peaks(sa02, distance=10, height=0)
#     # Finding peaks to find overlap for heart rate
#     peaks_toph, peaks_ysh = find_peaks(HR, distance=10, height=0, prominence = 2)

#     range_num = 10 # How many segments should the recording be split into
#     delay_array = [150,0,0] # Intializing the delay array
#     actual_delay = [] # Array that holds the actual delay of each 10th of a section.
#     range1 = 8 # The the beginning delay for overlap
#     range2 = 56 # The end delay for overlap
#     for k in range(0,range_num):
#         # create fraction of sa02 signal
#         start_frac, stop_frac = k*len(peaks_top)//range_num, (k+1)*len(peaks_top)//range_num 
#         # create fraction of HR signal
#         start_frach, stop_frach = k*len(peaks_toph)//range_num, (k+1)*len(peaks_toph)//range_num 
#         # print(start_frac, stop_frac)
#         # print(start_frach, stop_frach)
#         delay_overlap_2 = [0,0]
#         for i in range(range1,range2):
#             # Finding overlap between peaks in sa02 and HR.
#             delay_overlap_2 = np.vstack([delay_overlap_2, 
#                                          [i, len(np.intersect1d(peaks_top[start_frac:stop_frac], 
#                                                                                  peaks_toph[start_frach:stop_frach]+i))]]) 
#         # Saves the delay for the biggest overlap.
#         actual_delay = np.append(actual_delay, 
#                                  int(10*(np.where(delay_overlap_2[:,1]==max(delay_overlap_2[:,1]))[0][0]+range1))) 
#         if actual_delay[-1] < 80:
#             actual_delay[-1] = actual_delay[0] # saves default delay if no overlap is found
#         # All delays in each 10th of a section of the signal 
#         #(All indexes are multiplied by 10 to apply for nasal airflow signal that has 10 times more samples)
#         delay_array = np.vstack([delay_array, [int(actual_delay[-1]), 
#                                                peaks_toph[start_frach]*10, 
#                                                peaks_toph[stop_frach-1]*10]]) 
            
#     delay_array = np.vstack([delay_array, [int(actual_delay[-1]),
#                                            peaks_toph[stop_frach-1]*10,
#                                            len(HR)*10]]) # save the last values
    
#     print('\nDelay array: \nActual delay, time start, time stop\n')
#     print(delay_array)
#     print('\n')
    
    
    
#     # Generating saturation indices from the adaptive delay
#     start_sat_index = []
#     stop_sat_index = []
#     mover = []
#     for i in range(len(start_index)):
#          for k in range(len(delay_array[:,2])-1):
#             if start_index[i] in range(delay_array[:,2][k],delay_array[:,2][k+1]):
#                 start_sat_index = np.append(start_sat_index, int(start_index[i]) + int(delay_array[:,0][k+1]))
#                 mover = np.append(mover, int(delay_array[:,0][k+1]))
#             if stop_index[i] in range(delay_array[:,2][k],delay_array[:,2][k+1]):
#                 stop_sat_index = np.append(stop_sat_index, int(stop_index[i]) + int(delay_array[:,0][k+1]))
    
#     if len(mover) < len(start_index):
#         mover = np.append(mover, mover[-1])
        
#     return start_sat_index, stop_sat_index, mover

def normal_finder(sa02, data_processed, envelope_data, lower_envelope, ranger, duration):
    # Proccessed nasal pressure data
    # Envelope data from the upper envelope
    # Envelope data from the lower envelope
    # Duration in whole minutes
    sa02 = sa02.copy()
    
    time_duration = int(duration*600)
    new_index = []
    new_std = []
    new_diff_mean = []
    new_diff_mean_low = []
    amp = []
    sa02_std = []
    for i in range(len(envelope_data)):
        if i in ranger:
            continue
        if i == 0 or i % 300 == 0:
            new_index = np.append(new_index, i)
            new_std = np.append(new_std, np.std(envelope_data[i:i+time_duration]))
            new_diff_mean = np.append(new_diff_mean, np.mean(abs(np.diff(envelope_data[i:i+time_duration]))))
            new_diff_mean_low = np.append(new_diff_mean_low, np.mean(abs(np.diff(lower_envelope[i:i+time_duration]))))
            amp = np.append(amp, np.mean(envelope_data[i:i+time_duration]))
            sa02_std = np.append(sa02_std, np.std(sa02[int(i/10+14):int((i+time_duration)/10+14)]))

    new_std = new_std/new_std.max()
    new_diff_mean = new_diff_mean/new_diff_mean.max()
    new_diff_mean_low = new_diff_mean_low/new_diff_mean_low.max()
    # plt.plot(new_index, new_std)
    # plt.plot(new_index, new_diff_mean)
    # plt.plot(new_index, new_diff_mean_low)
    new_derp = sa02_std*(new_diff_mean+new_diff_mean_low)/2
    difference_array = abs(new_std+new_derp)+1/amp
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x_axis = np.linspace(0, len(data_processed) / 600, len(data_processed))
    ax.plot(x_axis, data_processed)
    ax.plot(x_axis, envelope_data)
    ax.plot(x_axis, lower_envelope)
    # ax.plot(new_index/600,difference_array,'o')
    
    epic = new_index[np.where(difference_array == difference_array.min())][0]
    # ax.plot(epic/600, difference_array.min(),'o')
    epic_stop = epic+time_duration
    
    normal_segment = data_processed[int(epic):int(epic)+time_duration]
    
    norm_up = np.mean(envelope_data[int(epic):int(epic)+time_duration])
    
    norm_down = np.mean(lower_envelope[int(epic):int(epic)+time_duration])
    
    ax.axhline(y=norm_up, color='red')
    ax.axhline(y=norm_down, color='red')
    
    left, bottom, width, height = (epic/600, -3, duration, 6)
    rect = plt.Rectangle((left, bottom),
                          width,
                          height,
                          facecolor="yellow",
                          alpha=0.7,
                          edgecolor='black',
                          linewidth=2)
    ax.add_patch(rect)
    
    plt.title('Nasal Airflow (Normal Finder)')
    plt.ylabel('Amplitude [ab]')
    plt.xlabel('Time [minutes]')
    
    
    

    return normal_segment, epic, epic_stop



def piecewise_lin_oxy_delay(sa02, envelope_data_nasal, start_index, stop_index):
    
    
    """
    This function finds the piece-wise linear oxygen delay.
    
    Input:
        sa02            (oxygen saturation signal)
        HR              (heart rate signal)
        start_index     (array of start indices from 
                         segmentation algorithm)
        stop_index      (array of stop indicies from 
                         segmentation algortihm)
        
    output:
        start_sat_index (array of start indices from 
                         segmentation algorithm + delay)
        stop_sat_index  (array of start indices from 
                         segmentation algorithm + delay)
        mover           (array of all delays to each entry in start_index)
    """
    
    
    sa02_diff = np.diff(sa02)
    sa02_diff = np.append(sa02_diff,0)
    x_dat = np.linspace(0,len(sa02_diff)*10,num=len(sa02_diff))
    new_sa02 = cubic_spline_int(x_dat, sa02_diff, len(sa02_diff)*10)
    # plt.plot(new_sa02)
    new_sa02 = savgol_filter(new_sa02, 201,2)
    # plt.plot(envelope_data_nasal)
    # plt.plot(new_sa02)
    
    delay_array = [150,0,0]
    
    if len(new_sa02) % 200 == 0:
        splitter = 200
    elif len(new_sa02) % 300 == 0:
        splitter = 300
    elif len(new_sa02) % 400 == 0:
        splitter = 400
    else:
        splitter = 100
    
    for i in range(splitter):
        a_i = int((i)*len(new_sa02)//splitter)
        print('a '+ str(a_i))
        b_i = int((i+1)*len(new_sa02)//splitter)
        print('b ' + str(b_i))
        
        corr = abs(np.argmax(np.correlate(new_sa02[a_i:b_i], envelope_data_nasal[a_i:b_i],'full'))-len(envelope_data_nasal[a_i:b_i]))
        if 80 <= corr <= 600:
            delay_array = np.vstack([delay_array, [corr, int(a_i), int(b_i)]])
        else:
            delay_array = np.vstack([delay_array, [150, int(a_i), int(b_i)]])
    
    print('\nDelay array: \nActual delay, time start, time stop\n')
    print(delay_array)
    print('\n')
    print(len(envelope_data_nasal))
    
    
    # Generating saturation indices from the adaptive delay
    start_sat_index = []
    stop_sat_index = []
    mover = []
    for k in range(len(delay_array[:,2])-1):
        for i in range(len(start_index)):
            if start_index[i] in range(int(delay_array[:,2][k]),int(delay_array[:,2][k+1])):
                # print(start_index[i], range(int(delay_array[:,2][k]),int(delay_array[:,2][k+1])))
                start_sat_index = np.append(start_sat_index, int(start_index[i]) + int(delay_array[:,0][k+1]))
                stop_sat_index = np.append(stop_sat_index, int(stop_index[i]) + int(delay_array[:,0][k+1]))
                mover = np.append(mover, int(delay_array[:,0][k+1]))
                
            # if stop_index[i] in range(int(delay_array[:,2][k]),int(delay_array[:,2][k+1])):
                
    
    print(len(start_index), len(start_sat_index), len(stop_sat_index))
    
    if len(mover) < len(start_index):
        mover = np.append(mover, mover[-1])
        
    return start_sat_index, stop_sat_index, mover, delay_array