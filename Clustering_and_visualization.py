# -*- coding: utf-8 -*-
"""
Clustering and visualization
This script contains the clustering and the visualization algorithm

@author: Villads
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import scipy.cluster.hierarchy as sch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# Import from scripts
from Data_loader import annotation_finder
from Segmentation_and_feature_extraction import cluster_function


#%%

"""
The section below calculates all variables 
needed for clustering and visualization

"""

time_start = time.time()
recording = 200070

start = recording-200001
stop = recording-200000
annotated=False

channel_dict = {0: 'nasal',
                1: 'abdo',
                2: 'thor'}
for channels in range(len(channel_dict.keys())):
    
    channel = channel_dict[channels]
    print('Working on the '+channel+' channel')
    print((time.time()-time_start)/60)

    if channel == 'nasal':
        (number, 
         start_index, 
         stop_index, 
         start_sat_index, 
         stop_sat_index, 
         mover, data_processed, 
         wake_start, wake_stop, 
         envelope_data, 
         lower_envelope, 
         clustering_array, 
         nas_final_clustering_array, 
         peak_f, events_in_recording, 
         sa02, 
         time_stamp_start, 
         time_stamp_stop,
         delay_array) = cluster_function(start,stop, channel, loading=False, annotated=False)
        data_processed_nasal, envelope_data_nasal, lower_envelope_nasal = data_processed, envelope_data, lower_envelope
    if channel == 'abdo':
        # continue
        (number, 
         start_index, 
         stop_index, 
         start_sat_index, 
         stop_sat_index, 
         mover, data_processed, 
         wake_start, wake_stop, 
         envelope_data, 
         lower_envelope, 
         clustering_array, 
         abdo_final_clustering_array, 
         peak_f, events_in_recording, 
         sa02, 
         time_stamp_start, 
         time_stamp_stop,
         delay_array) = cluster_function(start,stop, channel, loading=False, annotated=False)
        data_processed_abdo, envelope_data_abdo, lower_envelope_abdo = data_processed, envelope_data, lower_envelope
    if channel == 'thor':
        # continue
        (number, 
         start_index, 
         stop_index, 
         start_sat_index, 
         stop_sat_index, 
         mover, data_processed, 
         wake_start, wake_stop, 
         envelope_data, 
         lower_envelope, 
         clustering_array, 
         thor_final_clustering_array, 
         peak_f, events_in_recording, 
         sa02, 
         time_stamp_start, 
         time_stamp_stop,
         delay_array) = cluster_function(start,stop, channel, loading=False, annotated=False)
        data_processed_thor, envelope_data_thor, lower_envelope_thor = data_processed, envelope_data, lower_envelope
    plt.close('all')

final_clustering_array = np.zeros(shape=(nas_final_clustering_array.shape[0],nas_final_clustering_array.shape[1]*3-8))
for i in range(0,len(channel_dict.keys())):
    for k in range(0,nas_final_clustering_array.shape[1]-4):
        if i == 0:
            final_clustering_array[:,[i*k+k]] = nas_final_clustering_array[:,[k]]
        elif i == 1:
            final_clustering_array[:,[i*(nas_final_clustering_array.shape[1]-4)+k]] = abdo_final_clustering_array[:,[k]]
        elif i == 2:
            final_clustering_array[:,[i*(nas_final_clustering_array.shape[1]-4)+k]] = thor_final_clustering_array[:,[k]]

final_clustering_array[:,[-4]] = nas_final_clustering_array[:,[-4]] # adding start of event
final_clustering_array[:,[-3]] = nas_final_clustering_array[:,[-3]] # adding stop of event
final_clustering_array[:,[-2]] = nas_final_clustering_array[:,[-2]] # adding start of event
final_clustering_array[:,[-1]] = nas_final_clustering_array[:,[-1]] # adding stop of event

hyp_0, hyp_1 = annotation_finder(number, parameter='hypopneas')
obs_0, obs_1 = annotation_finder(number, parameter='obstructive')
cen_0, cen_1 = annotation_finder(number, parameter='central')
aro_0, aro_1 = annotation_finder(number, parameter='arousals')
pure_wake_0, pure_wake_1 = annotation_finder(number, parameter='wake')

annotation_dict = {'hypopneas': [hyp_0, hyp_1],
                    'obstructive': [obs_0, obs_1],
                    'central': [cen_0, cen_1],
                    'wake': [wake_start, wake_stop],
                    'arousal': [aro_0, aro_1]}

ranger = []
for k in range(len(wake_start)):
    ranger = np.append(ranger, range(int(wake_start[k]), int(wake_stop[k])))


color_dict = {0: 'green',
              1: 'red',
              2: 'cyan',
              3: 'purple',
              4: 'yellow',
              5: 'black',
              6: 'pink',
              7: 'orange',
              -1: 'grey',
              9: 'blue'}

dict_color = {'green': 0,
              'red': 1,
              'cyan': 2,
              'purple': 3,
              'yellow': 4,
              'black': 5,
              'pink': 6,
              'orange': 7,
              'grey': -1,
              'blue': 9}

for i in range(2,len(final_clustering_array[0,:])-4): # Removing the mean from each feature
    final_clustering_array[:,i] = final_clustering_array[:,i]-np.mean(final_clustering_array[:,i])

#%%
final_clust_save = final_clustering_array.copy()
for i in range(2,len(final_clustering_array[0,:])-4):
    final_clustering_array[:,i]=final_clustering_array[:,i]/np.std(final_clustering_array[:,i])

#%%

label_dict = {  # Names of features:
                0: 'AHI Predict',
                1: 'SatC',      #'Combined sat feat',
                2: 'SatC',#'BED',       #'Baseline up/low envelope difference',
                3: 'BED',#'rPSD',      #'Ratio of PSD',
                4: 'CorE',      #'Upper/Lower envelope correlation',
                5: 'Envelope Arc Length',
                6: 'HeartC',
                7: 'Wavelet feature',
                8: 'Hypo capture',
                9: 'Min/Max UpE and LowE',
                # Abdominal belt signal features:
                10: 'AHI Predict',
                11: 'SatC',      #'Combined sat feat',
                12: 'BED',       #'Baseline up/low envelope difference',
                13: 'rPSD',      #'Ratio of PSD',
                14: 'CorE',      #'Upper/Lower envelope correlation',
                15: 'Envelope Arc Length',
                16: 'Arc len sat',
                17: 'Wavelet feature',
                18: 'Hypo capture',
                19: 'Min/Max UpE and LowE',
                # Thorax belt signal features:
                20: 'AHI Predict',
                21: 'SatC',      #'Combined sat feat',
                22: 'BED',       #'Baseline up/low envelope difference',
                23: 'rPSD',      #'Ratio of PSD',
                24: 'CorE',      #'Upper/Lower envelope correlation',
                25: 'Envelope Arc Length',
                26: 'Arc len sat',
                27: 'Wavelet feature',
                28: 'Hypo capture',
                29: 'Min/Max UpE and LowE',
                30: 'Start saturation',
                31: 'Stop saturation',
                32: 'Start',
                33: 'Stop',
                }
    

# interest_array = [6,7,5,15,25,17,27]
# interest_array = [3,2,4,5,19,34]
interest_array = [13,2,4,5,8,9,10,11,12,3,19,34]

def label_function(final_clustering_array, interest_array, recording, sub_clust, clusters=3, threshold=18):
    
    """
    Hierachical clustering to generate segment labels
    
    Input:
        final_clustering_array      (The feature array)
        interest_array              (the specific features to look at)
        recording                   (recording number for saving plots)
        sub_clust (type = bool)     (for saving plots)
        clusters                    (the amount for clusters being found)
        threshold                   (threshold)
        
    Output:
        labels                      (array of labels)
    """
    
    
    og_thresh = threshold
    
    X = final_clustering_array[:,interest_array]
    Z = sch.linkage(X, 'ward')
    labels = sch.fcluster(Z, threshold, criterion='distance')
    # labels = sch.fcluster(Z, threshold, criterion='maxclust', maxclust=clusters)
    iteration = 0
    while len(np.unique(labels)) != clusters:
        iteration += 1
        if iteration == 200:
            break
        if len(np.unique(labels)) < clusters:
            threshold -= 0.5
            labels = sch.fcluster(Z, threshold, criterion='distance')
        else:
            threshold += 0.5
            labels = sch.fcluster(Z, threshold, criterion='distance')
    if threshold != og_thresh:
        print('Threshold corrected to {}'.format(threshold))
        
    labels = labels-1
    
    fig, ax = plt.subplots(figsize=(10,10))
    dendrogram = sch.dendrogram(Z, color_threshold=threshold)
    plt.axhline(y=threshold, color='black', linewidth=2)
    plt.show()
    plt.title('Dendrogram')
    plt.xlabel('Linked Clusters')
    plt.ylabel("Distance [Wards' distance measure]")
    
    if sub_clust == True:
        filename = str(recording)+'_sub_dendro'
    else:
        filename = str(recording)+'_dendro'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    fig.savefig(path+filename)
    
    return labels


def validator(final_clustering_array, annotation_dict, labels, ranger, event_type='hypopneas'):
    
    
    """
    This functions finds the overlap of annotations and the corresponding 
    color of the segment with the biggest overlap
    
    Input:
        final_clustering_array  (feature array)
        annotation_dict         (dictionary containing all annotations for 
                                 events)
        labels                  (labels found in the hierarchical clustering)
        ranger                  (array that consist of all datapoints indices
                                 that are in wake/movement segments)
        event_type              (the event type to look into, default 
                                 hypopneas)
        
    Output:
        percentages             (array of percentages of the how much the 
                                 event_type overlaps each color/cluster)
        amount,                 (how many of event_type that overlaps which
                                 color)
        val_idx,                (matrix containing 
                                 5 columns:
                                     column 0: the value of the maximum
                                               overlap for each event in 
                                               event_type.
                                     column 1: The row in the feature array
                                               that indicates in which segment
                                               the biggest overlap is.
                                     column 2: the label of the correspondig
                                               segment where the biggest
                                               overlap is.
                                     column 3: start of event_type.
                                     column 4: stop of event_type.
                                               
        event_type
        
        
    """
    
    valid_array_start = np.zeros(shape=(len(final_clustering_array),len(annotation_dict[event_type][0])))
    
    annotations_not_in_use = []
    amount_not_in_use = 0
    for e in range(len(annotation_dict[event_type][0])):
        if annotation_dict[event_type][0][e] in ranger and annotation_dict[event_type][1][e] in ranger:
            amount_not_in_use += 1
            annotations_not_in_use = np.append(annotations_not_in_use, [annotation_dict[event_type][0][e],
                                                                        annotation_dict[event_type][1][e]])
            continue
        for i in range(len(final_clustering_array)):
            num1 = int(final_clustering_array[i,-1])-int(final_clustering_array[i,-2])
            num2 = int(annotation_dict[event_type][1][e])-int(annotation_dict[event_type][0][e])
            valid_array_start[i,e] = len(np.intersect1d(np.linspace(int(final_clustering_array[i,-2]),
                                                                    int(final_clustering_array[i,-1])-1, 
                                                                    num=num1, 
                                                                    dtype='int64'),
                                                        np.linspace(int(annotation_dict[event_type][0][e]),
                                                                    int(annotation_dict[event_type][1][e])-1, 
                                                                    num=num2, 
                                                                    dtype='int64')))
    
   
    print('\n{} ({}) are in wake/movement segments.\n'.format(amount_not_in_use, event_type))
    
    val_idx = np.zeros(shape=(len(annotation_dict[event_type][0]),5))
    count_current = 0
    count_70 = 0
    for i in range(len(annotation_dict[event_type][0])):
        val_idx[i,0] = valid_array_start[:,i].max()
        val_idx[i,1] = np.where(valid_array_start[:,i].max() == valid_array_start[:,i])[0][0]
        val_idx[i,2] = labels[int(np.where(valid_array_start[:,i] == valid_array_start[:,i].max())[0][0])]
        val_idx[i,3] = annotation_dict[event_type][0][i]
        val_idx[i,4] = annotation_dict[event_type][1][i]
        
        if final_clustering_array[int(np.where(valid_array_start[:,i] == valid_array_start[:,i].max())[0][0]),0]==1:
           count_current +=1
        if final_clustering_array[int(np.where(valid_array_start[:,i] == valid_array_start[:,i].max())[0][0]),1]==1:
           count_70 +=1 
    
    print('Current rules prediction overlap: {}: {}'.format(event_type,count_current))
    print('Old rules prediction overlap: {}: {}'.format(event_type,count_70))
    
    for i in range(len(val_idx[:,0])-1):
        if val_idx[i,1] > 0 and val_idx[i,1] == val_idx[i+1,1]:
            val_idx[i,2] = -1
        if val_idx[i,4] in annotations_not_in_use:
            val_idx[i,2] = -1
    
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    f = 0
    g = 0
    h = 0
    j = 0
    k = 0
    for i in val_idx[:,2]:
        if i == 0:
            a += 1
        elif i == 1:
            b += 1
        elif i == 2:
            c += 1
        elif i == 3:
            d += 1
        elif i == 4:
            e += 1
        elif i == 5:
            f += 1
        elif i == 6:
            g += 1
        elif i == 7:
            h += 1
        elif i == 8:
            j += 1
        elif i == 9:
            k += 1
            
            
    
    percentages_init = [np.round(100*a/len(val_idx),1), 
                   np.round(100*b/len(val_idx),1), 
                   np.round(100*c/len(val_idx),1), 
                   np.round(100*d/len(val_idx),1), 
                   np.round(100*e/len(val_idx),1), 
                   np.round(100*f/len(val_idx),1), 
                   np.round(100*g/len(val_idx),1), 
                   np.round(100*h/len(val_idx),1), 
                   np.round(100*j/len(val_idx),1), 
                   np.round(100*k/len(val_idx),1)]
    amount_init = [a,b,c,d,e,f,g,h,j,k]
    percentages = []
    amount = []
    for i in range(len(np.unique(labels))):
        percentages = np.append(percentages, percentages_init[i])
        amount = np.append(amount, int(amount_init[i]))
    return percentages, amount, val_idx, event_type


def clustering_plot_function(final_clustering_array, 
                             interest_array, 
                             data_processed, 
                             sa02, 
                             envelope_data, 
                             lower_envelope, 
                             recording, 
                             time_stamp_start, 
                             time_stamp_stop, 
                             labels, 
                             annotation_dict, 
                             color_dict, 
                             coloring, ano_plot, 
                             plot_only_clustering, 
                             sub_clust, 
                             predict, 
                             plot_env):
    
    """
    Plotting function that plots the nasal airflow signal and oxygen 
    saturation signal of the processed recording. The function uses

    Input:
        final_clustering_array      (the feature array)
        interest_array              (the chosen features in the feature array)
        data_processed              (the signal to plot in the first plot)
        sa02                        (the processed oxygen saturation signal)
        envelope_data               (the upper envelope of the signal to plot
                                     in the first plot)
        lower_envelope              (the lower envelope of the signal to plot
                                     in the first plot)
        recording                   (the number of the recording being plotted)
        time_stamp_start            (the start index for the normal segment)
        time_stamp_stop,            (the stop index for the normal segment)
        labels                      (the labels found with the hierarchical
                                     clustering)
        annotation_dict             (dictionary containing all annotations for 
                                     events)
        color_dict                  (dictionary of integer keys and their
                                     corresponding color) 
        coloring                    (if True the plots are colored with 
                                     segments colored according to cluster
                                     if False the plots are not colored)
        ano_plot,                   (if True annotations are plotted)
        plot_only_clustering        (whether the clustering plotted agains two
                                     features should be plotted alone)
        sub_clust                   (only True if a subclustering is being 
                                     made)
        predict                     (only True if a seudo predictoin should be
                                     plotted)
        plot_env                    (if True the upper and lower envelopes are
                                     displayed in the plot)
    Output:
        None - the functions opens all the plots.
        
    """
    
    
    X = final_clustering_array[:,interest_array]
    
    # clust = MeanShift(bandwidth=1.5).fit(X)
    # labels = clust.labels_
    
    number = recording
    
    # fig, ax = plt.subplots(2)
    n_c = np.zeros(shape=(final_clustering_array.shape[0],final_clustering_array.shape[1]+1))
    n_c[:,0:len(n_c[0])-1] = final_clustering_array
    n_c[:,-1] = labels
    
    # print(numbers)
    
    
    wake_start, wake_stop = annotation_dict['wake'][0], annotation_dict['wake'][1]
    
    
    nan_array_0 = np.zeros(len(data_processed)) * np.nan
    nan_array_1 = np.zeros(len(data_processed)) * np.nan
    nan_array_2 = np.zeros(len(data_processed)) * np.nan
    nan_array_3 = np.zeros(len(data_processed)) * np.nan
    nan_array_4 = np.zeros(len(data_processed)) * np.nan
    nan_array_5 = np.zeros(len(data_processed)) * np.nan
    nan_array_6 = np.zeros(len(data_processed)) * np.nan
    nan_array_7 = np.zeros(len(data_processed)) * np.nan
    nan_array_8 = np.zeros(len(data_processed)) * np.nan
    nan_wake = np.zeros(len(data_processed)) * np.nan
    
    
    
    for i in range(len(final_clustering_array)):
    
        if labels[i] == 0:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_0[k] = data_processed[k]
        elif  labels[i] == 1:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_1[k] = data_processed[k]
        elif  labels[i] == 2:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_2[k] = data_processed[k]
        elif  labels[i] == 3:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_3[k] = data_processed[k]
        elif  labels[i] == 4:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_4[k] = data_processed[k]
        elif  labels[i] == 5:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_5[k] = data_processed[k]
        elif  labels[i] == 6:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_6[k] = data_processed[k]
        elif  labels[i] == 7:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_7[k] = data_processed[k]
        elif  labels[i] == 8:
            for k in range(int(n_c[:,len(n_c[0])-3][i]),int(n_c[:,len(n_c[0])-2][i])):
                nan_array_8[k] = data_processed[k]
                
                

    
    fig, ax = plt.subplots(2, sharex=True, figsize=(10,10))
    
    x_axis = np.linspace(0, len(data_processed) / 600, len(data_processed))
    
    ax[0].plot(x_axis, data_processed)
    
    
    
    if coloring==True:
        ax[0].plot(x_axis, nan_array_0, color='green')
        ax[0].plot(x_axis, nan_array_1, color='red')
        ax[0].plot(x_axis, nan_array_2, color='cyan') 
        ax[0].plot(x_axis, nan_array_3, color='purple')
        ax[0].plot(x_axis, nan_array_4, color='yellow')
        ax[0].plot(x_axis, nan_array_5, color='black')
        ax[0].plot(x_axis, nan_array_6, color='pink')
        ax[0].plot(x_axis, nan_array_7, color='yellow')
        ax[0].plot(x_axis, nan_array_8, color='grey')
        
    for i in final_clustering_array[:,-2]:
        ax[0].axvline(x=i/600, ymin=0.95, ymax=1, color='blue')
        ax[0].axvline(x=i/600, ymin=0.25, ymax=0.95, alpha=0.1, color='blue')
        ax[0].axvline(x=i/600, ymin=0.20, ymax=0.25, color='blue')
    
    ax[0].set(title=('Normalized Nasal Airflow'.format(recording)))
    ax[0].set(ylabel='Normalized Amplitude [ab]')
    ax[0].set(xlabel='Time [minutes]')
    
    legend_obj = []
    legend_names =  []
    
    legend_obj = np.append(legend_obj, ax[0].axvline(x=final_clustering_array[:,-1][0]/600,
                                                     ymin=0.95,
                                                     ymax=1,
                                                     color='blue'))
    legend_names = np.append(legend_names, 'Algorithm segments')
    
    if coloring==True:
        left, bottom, width, height = (time_stamp_start/600,
                                       -3,
                                       (time_stamp_stop-time_stamp_start)/600,
                                       6)
        rect = plt.Rectangle((left, bottom),
                             width,
                             height,
                             facecolor="yellow",
                             alpha=0.7,
                             edgecolor='red',
                             linewidth=2)
        ax[0].add_patch(rect)
        legend_obj = np.append(legend_obj, ax[0].add_patch(rect))
        legend_names = np.append(legend_names, 'Algorithm normal segment')
    
    
    
    if ano_plot == True:
        # hyp_0, hyp_1 = annotation_finder(number, parameter='hypopneas')
        if type(annotation_dict['hypopneas'][0]) != int:
            
            event_type='hypopneas'
            print('Number of hypopnea events: {}'.format(len(annotation_dict[event_type][0])))
            _, _, val_idx_hyp, _ = validator(final_clustering_array, 
                                             annotation_dict, 
                                             labels, 
                                             ranger, 
                                             event_type
                                             )
            
            first = True
            for q in range(len(annotation_dict[event_type][0])):
                left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                               -3,
                                               (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                               6)
                rect = plt.Rectangle((left, bottom),
                                     width,
                                     height,
                                     facecolor="grey",
                                     alpha=0.7,
                                     edgecolor=color_dict[val_idx_hyp[q,2]],
                                     linewidth=2)
                ax[0].add_patch(rect)
                if val_idx_hyp[q,2] == -1 and first==True:
                    left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                                   -3,
                                                   (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                                   6)
                    rect = plt.Rectangle((left, bottom),
                                         width,
                                         height,
                                         facecolor="grey",
                                         alpha=0.7,
                                         edgecolor=None,
                                         linewidth=2)
                    legend_obj = np.append(legend_obj, ax[0].add_patch(rect))
                    legend_names = np.append(legend_names, 'Annotated hypopnea events')
                    first = False
        else:
            print('No hypopnea events')
            
        # obs_0, obs_1 = annotation_finder(number, parameter='obstructive')
        if type(annotation_dict['obstructive'][0]) != int:
            event_type='obstructive'
            print('Number of obstructive events: {}'.format(len(annotation_dict[event_type][0])))
            _, _, val_idx_obs, _ = validator(final_clustering_array, 
                                             annotation_dict, 
                                             labels, 
                                             ranger, 
                                             event_type
                                             )
            first = True
            for q in range(len(annotation_dict[event_type][0])):
                left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                               -3,
                                               (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                               6)
                rect = plt.Rectangle((left, bottom),
                                     width,
                                     height,
                                     facecolor="brown",
                                     alpha=0.7,
                                     edgecolor=color_dict[val_idx_obs[q,2]],
                                     linewidth=2)
                ax[0].add_patch(rect)
                if val_idx_hyp[q,2] == -1 and first==True:
                    left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                                   -3,
                                                   (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                                   6)
                    rect = plt.Rectangle((left, bottom),
                                         width,
                                         height,
                                         facecolor="brown",
                                         alpha=0.7,
                                         edgecolor=None,
                                         linewidth=2)
                    legend_obj = np.append(legend_obj, ax[0].add_patch(rect))
                    legend_names = np.append(legend_names, 'Annotated obstructive events')
                    first = False
        else:
            print('No obstructive events')
        
        # cen_0, cen_1 = annotation_finder(number, parameter='central')
        if type(annotation_dict['central'][0]) != int:
            print('Number of central events: {}'.format(len(cen_0)))
            event_type = 'central'
            first = True
            for q in range(len(cen_0)):
                left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                               -3, (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                               6)
                rect = plt.Rectangle((left, bottom),
                                     width,
                                     height,
                                     facecolor="blue",
                                     alpha=0.7,
                                     edgecolor='orange',
                                     linewidth=2)
                ax[0].add_patch(rect)
                if first==True:
                    left, bottom, width, height = (annotation_dict[event_type][0][q]/600,
                                                   -3, (annotation_dict[event_type][1][q]-annotation_dict[event_type][0][q])/600,
                                                   6)
                    rect = plt.Rectangle((left, bottom),
                                         width,
                                         height,
                                         facecolor="blue",
                                         alpha=0.7,
                                         edgecolor='orange',
                                         linewidth=2)
                    legend_obj = np.append(legend_obj, ax[0].add_patch(rect))
                    legend_names = np.append(legend_names, 'Annotated central events')
                    first = False
        else:
            print('No central events')
    else:
        print('Not showing annotations')
    
    
        
    #     ax[1].plot((start_sat_index[i]/600,stop_sat_index[i]/600),(101,101), color='black')
    #     ax[1].arrow(start_sat_index[i]/600,101,0,-2, width = 0.01, color='black')
    #     ax[1].arrow(stop_sat_index[i]/600,101,0,-2, width = 0.01, color='black')
   
    
    nan_array_0_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_1_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_2_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_3_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_4_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_5_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_6_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_7_sa02 = np.zeros(len(sa02)) * np.nan
    nan_array_8_sa02 = np.zeros(len(sa02)) * np.nan
    nan_wake_sa02 = np.zeros(len(sa02)) * np.nan
    
    
    for i in range(len(final_clustering_array)):
    
        if labels[i] == 0:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_0_sa02[k] = sa02[k]
        elif  labels[i] == 1:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_1_sa02[k] = sa02[k]
        elif  labels[i] == 2:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_2_sa02[k] = sa02[k]
        elif  labels[i] == 3:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_3_sa02[k] = sa02[k]
        elif  labels[i] == 4:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_4_sa02[k] = sa02[k]
        elif  labels[i] == 5:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_5_sa02[k] = sa02[k]
        elif  labels[i] == 6:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_6_sa02[k] = sa02[k]
        elif  labels[i] == 7:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_7_sa02[k] = sa02[k]
        elif  labels[i] == 8:
            for k in range(int(np.round((n_c[:,len(n_c[0])-5][i])/10)),
                           int(np.round((n_c[:,len(n_c[0])-4][i])/10))):
                nan_array_8_sa02[k] = sa02[k]
                
    x_axis = np.linspace(0, len(sa02)/60, len(sa02))
    
    ax[1].plot(x_axis, sa02)
    
    if coloring==True:
        ax[1].plot(x_axis, nan_array_0_sa02, color='green', linewidth=3)
        ax[1].plot(x_axis, nan_array_1_sa02, color='red', linewidth=3)
        ax[1].plot(x_axis, nan_array_2_sa02, color='cyan', linewidth=3)
        ax[1].plot(x_axis, nan_array_3_sa02, color='purple', linewidth=3)
        ax[1].plot(x_axis, nan_array_4_sa02, color='yellow', linewidth=3)
        ax[1].plot(x_axis, nan_array_5_sa02, color='black', linewidth=3)
        ax[1].plot(x_axis, nan_array_6_sa02, color='pink', linewidth=3)
        ax[1].plot(x_axis, nan_array_7_sa02, color='orange', linewidth=3)
        ax[1].plot(x_axis, nan_array_8_sa02, color='grey', linewidth=3)
    
    ax[1].set(ylabel='Oxygen Saturation [% saturation]')
    ax[1].set(xlabel='Time [minutes]')
    
    for i in final_clustering_array[:,-4]:
        ax[1].axvline(x=i/600, ymin=0.95, ymax=1, color='blue')
        ax[1].axvline(x=i/600, ymin=0.15, ymax=0.95, alpha=0.1, color='blue')
        ax[1].axvline(x=i/600, ymin=0.10, ymax=0.15, color='blue')
    
    
    
    if plot_env == True:
        x_axis = np.linspace(0, len(data_processed) / 600, len(data_processed))
        legend_line_colors = ax[0].plot(x_axis, envelope_data_nasal, color='purple')
        legend_line_names = 'The signal envelope'
        ax[0].plot(x_axis, envelope_data_nasal, color='purple')
        ax[0].plot(x_axis, lower_envelope_nasal, color='purple')
        legend_obj = np.append(legend_obj, legend_line_colors)
        legend_names = np.append(legend_names, legend_line_names)
    
    if predict == True:
        for i in range(len(final_clustering_array[:,0])):
            if final_clustering_array[i,0] == 1:
                ax[0].plot((final_clustering_array[i,-2]/600,
                            final_clustering_array[i,-1]/600),(6,6),
                           color='black')
                ax[0].arrow(final_clustering_array[i,-2]/600,6,0,-2, width = 0.01, color='black')
                ax[0].arrow(final_clustering_array[i,-1]/600,6,0,-2, width = 0.01, color='black')
        legend_line_colors1 = plt.plot([0,1],[0,1],color='black')
        legend_line_names1 = 'Predictions'
        legend_obj = np.append(legend_obj, legend_line_colors1)
        legend_names = np.append(legend_names, legend_line_names1)
    
    # Adding legend
    ax[0].legend(legend_obj, legend_names, fontsize=12, facecolor='white', bbox_to_anchor=(0.38, 0.2))
    
    if coloring == True:
        a = plt.axes([.65, .38, .25, .25], facecolor='w')
        plt.title('Clustering', backgroundcolor= 'white')
        a.title.set_position([0.5, 1.018])
        plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='green')
        plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='red')
        plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='cyan')
        plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
        plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='yellow')
        plt.scatter(X[labels==5, 0], X[labels==5, 1], s=50, marker='o', color='black')
        plt.scatter(X[labels==6, 0], X[labels==6, 1], s=50, marker='o', color='pink')
        plt.scatter(X[labels==7, 0], X[labels==7, 1], s=50, marker='o', color='orange')
        plt.scatter(X[labels==8, 0], X[labels==8, 1], s=50, marker='o', color='grey')
        plt.scatter(X[labels==9, 0], X[labels==9, 1], s=50, marker='o', color='blue')
        plt.xlabel(label_dict[interest_array[0]], backgroundcolor= 'white')
        plt.ylabel(label_dict[interest_array[1]], rotation=0, backgroundcolor= 'white')
        
        a.tick_params(axis='y',
                      which='both',
                      direction='in',
                      pad=-20,
                      right=True,
                      left=False,
                      labelleft=False,
                      labelright=True)
        a.tick_params(axis='x',
                      which='both',
                      direction='in',
                      pad=-20,
                      top=True,
                      bottom=False,
                      labelbottom=False,
                      labeltop=True)
        a.yaxis.set_label_position("right")
        a.xaxis.set_label_position("top")
        a.xaxis.set_label_coords(0.5,0.90)
        a.yaxis.set_label_coords(0.90,0.5)
        
    # plt.yticks(rotation=90)
    
    if plot_only_clustering == True:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='green')
        ax.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='red')
        ax.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='cyan')
        ax.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
        ax.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='yellow')
        ax.scatter(X[labels==5, 0], X[labels==5, 1], s=50, marker='o', color='black')
        ax.scatter(X[labels==6, 0], X[labels==6, 1], s=50, marker='o', color='pink')
        ax.scatter(X[labels==7, 0], X[labels==7, 1], s=50, marker='o', color='orange')
        ax.scatter(X[labels==8, 0], X[labels==8, 1], s=50, marker='o', color='grey')
        ax.scatter(X[labels==9, 0], X[labels==9, 1], s=50, marker='o', color='blue')
        ax.set_title('Clustering', backgroundcolor= 'white')
        ax.set_xlabel(label_dict[interest_array[0]])
        ax.set_ylabel(label_dict[interest_array[1]])
        if sub_clust == True:
            filename = str(recording)+'_sub_cluster'
        else:
            filename = str(recording)+'_cluster'
        path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
        fig.savefig(path+filename)


def PCA_plotter(final_clustering_array, interest_array, labels, color_dict, recording, sub_clust):
    
    """
    This functions makes the PCA analysis and makes a visualization of the 
    PCA.
    
    Input:
        final_clustering_array  (the feature array)
        interest_array          (the feature chosen)
        labels                  (labels from the hierarchical clustering)
        color_dict              (dictionary of integer keys and their
                                 corresponding color)
        recording               (the number of the recording being plotted)
        sub_clust               (True if a subclustering is made, otherwise
                                 False)
    Output:
        None - generates and saves plots
        
    """
    ## Cannot save 3d plot automatically
    
    X = final_clustering_array.copy()

    new_labels = []
    for i in range(len(labels)):
        new_labels = np.append(new_labels, color_dict[labels[i]])
        
    df = pd.DataFrame(X)
    n_components=3
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X[:,interest_array])
    
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    if n_components>=3:
        df['pca-three'] = pca_result[:,2]
    if n_components>= 4:
        df['pca-four'] = pca_result[:,3]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    var_expl = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.bar([0,1,2], var_expl, align='center', alpha=0.6)
    for index, value in enumerate(var_expl):
        plt.text(index, value, str(np.round(value,3)))
    plt.xticks(np.arange(len(var_expl)), ['PC1', 'PC2', 'PC3'])
    plt.xlabel('Principal Components',fontsize=24)
    plt.ylabel('Variance explained by Principal Component',fontsize=24)
    plt.title('Histogram of PCA Variance Explained',fontsize=24)
    if sub_clust == True:
        filename = str(recording)+'_sub_hist_PCA'
    else:
        filename = str(recording)+'_hist_PCA'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    fig.savefig(path+filename)
    
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    
    ax = plt.figure(figsize=(10,10))
    plt.scatter(
        df["pca-one"], 
        df["pca-two"], 
        c=new_labels
    )
    plt.xlabel('pca-one',fontsize=24)
    plt.ylabel('pca-two',fontsize=24)
    plt.title('PCA Clustering',fontsize=24)
    if sub_clust == True:
        filename = str(recording)+'_sub_1_2_PCA'
    else:
        filename = str(recording)+'_1_2_PCA'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    ax.savefig(path+filename)
    
    
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x="pca-one", y="pca-three",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    
    ax = plt.figure(figsize=(10,10))
    plt.scatter(
        df["pca-one"], 
        df["pca-three"], 
        c=new_labels
    )
    plt.xlabel('pca-one',fontsize=24)
    plt.ylabel('pca-three',fontsize=24)
    plt.title('PCA Clustering',fontsize=24)
    
    if sub_clust == True:
        filename = str(recording)+'_sub_1_3_PCA'
    else:
        filename = str(recording)+'_1_3_PCA'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    ax.savefig(path+filename)
    
    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x="pca-two", y="pca-three",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )
    
    ax = plt.figure(figsize=(10,10))
    plt.scatter(
        df["pca-two"], 
        df["pca-three"], 
        c=new_labels
    )
    plt.xlabel('pca-two',fontsize=24)
    plt.ylabel('pca-three',fontsize=24)
    plt.title('PCA Clustering',fontsize=24)
    
    if sub_clust == True:
        filename = str(recording)+'_sub_2_3_PCA'
    else:
        filename = str(recording)+'_2_3_PCA'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    ax.savefig(path+filename)
    
    
    
    ax = plt.figure(figsize=(10,10)).gca(projection='3d')
    ax.scatter(
        xs=df["pca-one"], 
        ys=df["pca-two"], 
        zs=df["pca-three"], 
        c=new_labels
    )
    
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    ax.set_title('PCA Clustering')
    
    # if sub_clust == True:
    #     filename = str(recording)+'_sub_3d_PCA'
    # else:
    #     filename = str(recording)+'_3d_PCA'
    # path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    # fig.savefig(path+filename)
    plt.show()




#%%

### SUB CLUSTERING
def sub_cluster_function(dict_color, val_idx, final_clustering_array, labels_in, color='cyan'):
    
    """
    Function to create new labels if a subclustering of a specific cluster 
    from the hierarchical clustering is wanted.
    
    Input:
        dict_color              (Dictionary with string colors as keys and 
                                 the corresponding integer for that color)
        val_idx                 (matrix containing 
                                 5 columns:
                                     column 0: the value of the maximum
                                               overlap for each event in 
                                               event_type.
                                     column 1: The row in the feature array
                                               that indicates in which segment
                                               the biggest overlap is.
                                     column 2: the label of the correspondig
                                               segment where the biggest
                                               overlap is.
                                     column 3: start of event_type.
                                     column 4: stop of event_type.
        final_clustering_array  (the feature array)
        labels_in               (the labels from the hierarchical clustering)
        color                   (the color of the clustering being
                                 investigated with subclustering)
    Output:
        sub_clustering_array    (same number of columns as feature array, but
                                 contains only the rows that are labeled 
                                 according to the color being investigated)
        sub_cluster_dict        (dictionary containing start and stop 
                                 indices of the segments of belonging to the 
                                 cluster being subclustered)
    
    """
    
    annotations_in_sub = []
    og_start_idx = []
    og_stop_idx = []
    for i in range(len(val_idx[:,2])):
        if val_idx[i,2] == dict_color[color]:
            annotations_in_sub = np.append(annotations_in_sub, val_idx[i,1])
            og_start_idx = np.append(og_start_idx, val_idx[i,3])
            og_stop_idx = np.append(og_stop_idx, val_idx[i,4])
    
    sub_clustering_array = []
    
    for i in range(len(labels_in)):
        if labels_in[i] == dict_color[color]:
            if sub_clustering_array == []:
                sub_clustering_array = final_clustering_array[i,:]
            else:
                sub_clustering_array = np.vstack([sub_clustering_array, final_clustering_array[i,:]])
    
    sub_cluster_dict = {'sub': [og_start_idx, og_stop_idx]}
    
    return sub_clustering_array, sub_cluster_dict


#%%

#%%
AHI_predict_current = sum(final_clustering_array[:,0])/(((len(data_processed_nasal)-len(ranger))/600)/60)
AHI_predict_70 = sum(final_clustering_array[:,1])/(((len(data_processed_nasal)-len(ranger))/600)/60)

#%%
# for i in interest_array:
#     print('Variance of {}: {}'.format(label_dict[i],np.var(final_clustering_array[:,i])))

### USING FUNCTIONS
### OVERALL CLUSTERING

print("Amount of movement artifacts found: {}".format(len(wake_start)-len(pure_wake_0)))

labels = label_function(final_clustering_array,  
                        interest_array,
                        recording,
                        sub_clust=False,
                        clusters=3, 
                        threshold=18)
# c = 0
# for i in range(len(labels)):
#     if labels[i] == 1:
#         c+=1
    # elif labels[i] == 1:
    #     labels[i] = 0

# labels = labels + 1
# 
#         
clustering_plot_function(final_clustering_array, 
                         interest_array, 
                         data_processed_nasal, 
                         sa02, envelope_data, 
                         lower_envelope, recording, 
                         time_stamp_start,
                         time_stamp_stop,
                         labels,
                         annotation_dict,
                         color_dict,
                         coloring=True,
                         ano_plot=False,
                         plot_only_clustering=False,
                         sub_clust=False,
                         predict=False,
                         plot_env=True
                         )
#%%
new_count = 0
for i in range(len(labels)):
    if labels[i] == 1:
        new_count += 1
print(new_count)
#%%
pred_current = []
pred_70 = []
for i in range(len(final_clustering_array[:,0])):
    if labels[i] == 1 and final_clustering_array[i,0] == 1:
        pred_current = np.append(pred_current, 1)
    elif labels[i] == 1 and final_clustering_array[i,0] != 1:
        pred_current = np.append(pred_current, 0)

for i in range(len(final_clustering_array[:,0])):
    if labels[i] == 1 and final_clustering_array[i,1] == 1:
        pred_70 = np.append(pred_70, 1)
    elif labels[i] == 1 and final_clustering_array[i,1] != 1:
        pred_70 = np.append(pred_70, 0)

#%%

PCA_plotter(final_clustering_array, interest_array, labels, color_dict, recording, sub_clust=False)
#%%
### Distribution of labels:

def label_annotation_distribution(labels, color_dict, final_clustering_array, annotation_dict, ranger):
    

    """
    This function takes information from the segments color and their 
    placement and compare to the annotations placement to determine which 
    segments the annotations overlaps the most. It then plots all this 
    information in pie charts to show the distribution of segments compared
    to annotions type and the distribution of annotation type compared to 
    segment color.
    
    Input:
        labels                  (labels of the clustering)
        color_dict              (dictionary of integer keys and their
                                 corresponding color) 
        final_clustering_array  (the feature array)
        annotation_dict         (dictionary containing all annotations for 
                                 events)
        ranger                  (array that consist of all datapoints indices
                                 that are in wake/movement segments)
    
    Output:
        val_idx_hyp             (matrix containing
                                 5 columns:
                                     column 0: the value of the maximum
                                               overlap for each hypopnea.
                                     column 1: The row in the feature array
                                               that indicates in which segment
                                               the biggest overlap is.
                                     column 2: the label of the correspondig
                                               segment where the biggest
                                               overlap is.
                                     column 3: start indices of hypopneas.
                                     column 4: stop indices of hypopneas.)
        val_idx_obs             (matrix containing
                                 5 columns:
                                     column 0: the value of the maximum
                                               overlap for each hypopnea.
                                     column 1: The row in the feature array
                                               that indicates in which segment
                                               the biggest overlap is.
                                     column 2: the label of the correspondig
                                               segment where the biggest
                                               overlap is.
                                     column 3: start indices of hypopneas.
                                     column 4: stop indices of hypopneas.)
        segments_label          (array that contains the names of each color
                                 in the clustering)
        names                   (array containing a string for plotting 
                                 purposes)
    """
    
    n_clusters = len(np.unique(labels))
    n_segments = len(labels)
    print('Number of clusters {} \nNumber of segments: {}\n'.format(n_clusters, 
                                                                       n_segments))
    
    segment_dict = {}
    for e in range(len(np.unique(labels))):
        update_dict = {e: 0}
        segment_dict.update(update_dict)
    
    for q in range(len(labels)):
        for w in range(0,n_clusters):
            if labels[q] == w and w == np.unique(labels)[w]:
                segment_dict[w] = segment_dict[w] + 1
    
    
    for s in range(len(np.unique(labels))):
        print('Amount of segments in {} cluster: {} ({}%)'.format(color_dict[s],
                                                                  segment_dict[s], 
                                                                  np.round(100*segment_dict[s]/len(labels),1)))
    
    
    ## Pie chart
    
    fig, ax = plt.subplots(figsize=(12,10))
    segments_unique, segments_count = np.unique(labels,return_counts=True)
    # segments_count = segments_count*100/sum(segments_count)
    segments_label = []
    for i in segments_unique:
        segments_label = np.append(segments_label, color_dict[segments_unique[i]])
    ax.pie(segments_count,
           autopct='%1.1f%%',
           colors=segments_label,
           shadow=True,
           startangle=45)
    ax.set(title=('Distribution of segments by cluster labels\n(Relative size of each cluster)'))
    ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
    
    filename = str(recording)+'_dist_clusters'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    fig.savefig(path+filename)
    ### VALIDATION OF ANNOTATIONS:
    names=[]
    for i in range(len(segments_unique)):    
        names = np.append(names,'')
    percentages, amount_hyp, val_idx_hyp, event_type_hyp = validator(final_clustering_array, 
                                                                     annotation_dict, 
                                                                     labels, 
                                                                     ranger, 
                                                                     event_type='hypopneas')
    
    print('\nRecording: {0}\nEvent type: {1}'.format(recording, 
                                                     event_type_hyp))
    for s in range(len(np.unique(labels))):
        print('Amount of {} in {} cluster: {} ({}%)'.format(event_type_hyp, 
                                                             color_dict[s],
                                                             int(amount_hyp[s]),
                                                             percentages[s]))
    
    
    percentages = np.append(percentages, 100-sum(percentages))
    segments_label = np.append(segments_label, 'grey')
    names = np.append(names, 'Annotated events \nin wake/movement \nsegments (grey)')
    
    explosion = np.ones(len(percentages))*0.3
    fig, ax = plt.subplots(figsize=(12,10))
    ax.pie(percentages,
           labels=names,
           explode=explosion,
           autopct='%1.1f%%',
           colors=segments_label,
           shadow=True,
           startangle=45)
    ax.set(title=('Distribution of annotated \nhypopnea events'))
    ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
    
    
    filename = str(recording)+'_dist_hypopneas'
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
    fig.savefig(path+filename)
    
    print(' ')
    segment_hyp_ratio = []
    for s in range(len(np.unique(labels))):
        segment_hyp_ratio = np.append(segment_hyp_ratio,
                                      np.round(100*amount_hyp[s]/(segment_dict[s]+0.000000001),1))
        seg_hyp_ratio = np.round(100*amount_hyp[s]/(segment_dict[s]+0.000000001),1)
        print('Segments in {} cluster annotated as {}: {}/{} ({}%)'.format(color_dict[s], 
                                                                            event_type_hyp, 
                                                                            int(amount_hyp[s]), 
                                                                            segment_dict[s],
                                                                            seg_hyp_ratio))
        if type(obs_0) == int:
            fig, ax = plt.subplots(figsize=(12,10))
            ax.pie([100-seg_hyp_ratio, seg_hyp_ratio],
                   labels=['With no\nannotations \n({}/{})'.format(segment_dict[s]-int(amount_hyp[s]),
                                                                   segment_dict[s]),
                           'With {}\n({}/{})'.format(event_type_hyp,
                                                     int(amount_hyp[s]),
                                                     segment_dict[s])],
                   autopct='%1.1f%%',
                   colors=['grey',color_dict[s]],
                   shadow=True,
                   startangle=45)
            ax.set(title=('The distribution of {0} segments').format(color_dict[s]))
            filename = str(recording)+'_dist_'+color_dict[s]
            path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
            fig.savefig(path+filename)
    
    # Obstructive events
    print(type(obs_0))
    if type(obs_0) != int:
        percentages, amount_obs, val_idx_obs, event_type_obs = validator(final_clustering_array, 
                                                                         annotation_dict, 
                                                                         labels, 
                                                                         ranger, 
                                                                         event_type='obstructive')
    
        print('\nEvent type: {}'.format(event_type_obs))
        for s in range(len(np.unique(labels))):
            print('Amount of {} in {} cluster: {} ({}%)'.format(event_type_obs, 
                                                          color_dict[s],
                                                          int(amount_obs[s]),
                                                          percentages[s]))
       
        percentages = np.append(percentages, 100-sum(percentages))
        explosion = np.ones(len(percentages))*0.3
        fig, ax = plt.subplots(figsize=(12,10))
        ax.pie(percentages,
               labels=names,
               explode=explosion,
               autopct='%1.1f%%',
               colors=segments_label,
               shadow=True,
               startangle=45)
        ax.set(title=('Distribution of annotated \nobstructive events'))
        ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
        
        filename = str(recording)+'_dist_obstructive'
        path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
        fig.savefig(path+filename)
        
        
        print(' ')
        for s in range(len(np.unique(labels))):
            segment_obs_ratio = np.append(segment_hyp_ratio,
                                          np.round(100*amount_obs[s]/(segment_dict[s]+0.000000001),1))
            seg_obs_ratio = np.round(100*amount_obs[s]/(segment_dict[s]+0.000000001),1)
            print('Segments in {} cluster annotated as {}: {}/{} ({}%)'.format(color_dict[s], 
                                                                               event_type_obs, 
                                                                               int(amount_obs[s]), 
                                                                               segment_dict[s],
                                                                               seg_obs_ratio))
        
        
        print('\n')
        segment_hyp_obs_ratio = []
        a_pie_ca = []
        for s in range(len(np.unique(labels))):
            segment_hyp_obs_ratio = np.append(segment_hyp_obs_ratio,
                                              np.round(100*(amount_hyp[s]+amount_obs[s])/(segment_dict[s]+0.000000001),1))
            seg_hyp_ratio = np.round(100*amount_hyp[s]/(segment_dict[s]+0.000000001),1)
            seg_obs_ratio = np.round(100*amount_obs[s]/(segment_dict[s]+0.000000001),1)
            seg_hyp_obs_ratio = np.round(100*(amount_hyp[s]+amount_obs[s])/(segment_dict[s]+0.000000001),1)
            print('Segments in {} cluster annotated as {} or {}: {}/{} ({}%)'.format(color_dict[s], 
                                                                                     event_type_hyp,
                                                                                     event_type_obs,
                                                                                     int(amount_hyp[s]+amount_obs[s]), 
                                                                                     segment_dict[s],
                                                                                     seg_hyp_obs_ratio))
            
            fig, ax = plt.subplots(figsize=(12,10))
            ax.pie([100-seg_hyp_obs_ratio, seg_hyp_ratio, seg_obs_ratio], 
                   labels=['With no\nannotations \n({}/{})'.format(segment_dict[s]-int(amount_hyp[s]+amount_obs[s]),
                                                                   segment_dict[s]),
                           'With {}\n({}/{})'.format(event_type_hyp,
                                                     int(amount_hyp[s]),
                                                     segment_dict[s]),
                           'With {}\n({}/{})'.format(event_type_obs,
                                                     int(amount_obs[s]),
                                                     segment_dict[s])], 
                   explode=[0.3,0,0.075], 
                   autopct='%1.1f%%', 
                   colors=['grey', color_dict[s], 
                           color_dict[s]], shadow=True, startangle=225)
            a_pie_ca = np.append(a_pie_ca, np.array([np.round(100-seg_hyp_obs_ratio,1), 
                                                     np.round(seg_hyp_ratio,1),
                                                     np.round(seg_obs_ratio,1)]))
            ax.set(title=('The distribution of {0} segments').format(color_dict[s]))
            ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
            
            filename = str(recording)+'_dist_'+color_dict[s]
            path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/plots/Results/'
            fig.savefig(path+filename)
    
    fig, ax = plt.subplots(figsize=(12,10))
    segments_unique, segments_count = np.unique(labels,return_counts=True)
    # segments_count = segments_count*100/sum(segments_count)
    segments_label = []
    for i in segments_unique:
        segments_label = np.append(segments_label, color_dict[segments_unique[i]])
    ax.pie(segments_count, autopct='%1.1f%%', colors=segments_label, shadow=True, startangle=45)
    ax.set(title=('Distribution of segments by cluster labels\n(Relative size of each cluster)'))
    ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
    if type(obs_0) != int:
        fig, ax = plt.subplots(figsize=(12,10))
        # a_pie_ca[0]
        size = 0.2
        vals = np.array([[a_pie_ca[0]*segments_count[0]/sum(segments_count), 
                          a_pie_ca[1]*segments_count[0]/sum(segments_count),
                          a_pie_ca[2]*segments_count[0]/sum(segments_count)], 
                         [a_pie_ca[3]*segments_count[1]/sum(segments_count),
                          a_pie_ca[4]*segments_count[1]/sum(segments_count),
                          a_pie_ca[5]*segments_count[1]/sum(segments_count)], 
                         [a_pie_ca[6]*segments_count[2]/sum(segments_count),
                          a_pie_ca[7]*segments_count[2]/sum(segments_count),
                          a_pie_ca[8]*segments_count[2]/sum(segments_count)]])
        
        cmap = plt.get_cmap("nipy_spectral")
        outer_colors = segments_label #  cmap([120,225,90])
        inner_colors = cmap(np.array([130, 140, 150, 195, 205, 215, 65, 75, 85]))
        #('grey', color_dict[0], 'olive', 'grey', color_dict[1], 'pink', 'grey', color_dict[2], 'blue')
        
        ax.pie(segments_count, radius=1, colors=outer_colors,
               wedgeprops=dict(width=0.4, edgecolor='w'),startangle=130)#, autopct='%1.1f%%')
        new_labels=['w','o','h','w','o','h','w','o','h']
        wedges, texts = ax.pie(vals.flatten(), radius=1-0.38, colors=inner_colors,
               wedgeprops=dict(width=0.6, edgecolor='w'), startangle=130)
        
        
        plt.legend(['All green segments \n('+str(np.round(100*segments_count[0]/sum(segments_count),1))+'%) (outer ring)',
                    'All red segments \n('+str(np.round(100*segments_count[1]/sum(segments_count),1))+'%) (outer ring)',
                    'All cyan segments \n('+str(np.round(100*segments_count[2]/sum(segments_count),1))+'%) (outer ring)\n',
                    'green no annotations \n('+str(np.round(a_pie_ca[0]*segments_count[0]/sum(segments_count),1))+'%) (inner ring)',
                    'green hypopneas \n('+str(np.round(a_pie_ca[1]*segments_count[0]/sum(segments_count),1))+'%) (inner ring)',
                    'green obstructive \n('+str(np.round(a_pie_ca[2]*segments_count[0]/sum(segments_count),1))+'%) (inner ring)',
                    'red no annotations \n('+str(np.round(a_pie_ca[3]*segments_count[1]/sum(segments_count),1))+'%) (inner ring)',
                    'red hypopneas \n('+str(np.round(a_pie_ca[4]*segments_count[1]/sum(segments_count),1))+'%) (inner ring)',
                    'red obstructive \n('+str(np.round(a_pie_ca[5]*segments_count[1]/sum(segments_count),1))+'%) (inner ring)',
                    'cyan no annotations \n('+str(np.round(a_pie_ca[6]*segments_count[2]/sum(segments_count),1))+'%) (inner ring)',
                    'cyan hypopneas \n('+str(np.round(a_pie_ca[7]*segments_count[2]/sum(segments_count),1))+'%) (inner ring)',
                    'cyan obstructive \n('+str(np.round(a_pie_ca[8]*segments_count[2]/sum(segments_count),1))+'%) (inner ring)'],
                   fontsize=16,
                   loc=4,
                   bbox_to_anchor=(1.2, 0.0))
        
        ax.set(title=("Distribution of each cluster (outer ring) \nSub-distribution of each cluster (inner ring)"))
        ax.text(-1.5,-1,'Recording {}'.format(recording),size=8)
    
    if type(obs_0) == int:
        val_idx_obs = [0,0,0,0,0]
        
    return val_idx_hyp, val_idx_obs, segments_label, names

val_idx_hyp , val_idx_obs, segments_label, names = label_annotation_distribution(labels, 
                                                          color_dict, 
                                                          final_clustering_array, 
                                                          annotation_dict, 
                                                          ranger)

amount_not_hyp = 0
amount_not_obs = 0  
for i in range(len(val_idx_hyp)):
    if val_idx_hyp[:,2][i] == -1:
        amount_not_hyp +=1
if type(obs_0) != int:
    for i in range(len(val_idx_obs)):
        if val_idx_obs[:,2][i] == -1:
            amount_not_obs +=1

ahi_hyp_obs = 0
if type(annotation_dict['central'][0]) == int:
    centrals = 0
else:
    centrals = len(annotation_dict['central'][0])
if type(annotation_dict['hypopneas'][0]) == int:
    hypops = 0
else:
    hypops = len(annotation_dict['hypopneas'][0])-amount_not_hyp
if type(annotation_dict['obstructive'][0]) == int:
    obstr = 0
else:
    obstr = len(annotation_dict['obstructive'][0])-amount_not_obs

AHI_predict_rules = (hypops + obstr) / (((len(data_processed_nasal)-len(ranger))/600)/60)
print('\n\nAHI by annotations: {}'.format(AHI_predict_rules))
 
#%%
sub_cluster_visual = val_idx_hyp
look_at = 'hypopneas'
color_look = 'cyan'


sub_clustering_array, sub_cluster_dict = sub_cluster_function(dict_color, 
                                                              sub_cluster_visual, 
                                                              final_clustering_array, 
                                                              labels, 
                                                              color=color_look)

labels_sub = label_function(sub_clustering_array, 
                            interest_array,
                            recording,
                            sub_clust=True,
                            clusters=2, 
                            threshold=18)

clustering_plot_function(sub_clustering_array, 
                         interest_array,
                         data_processed_nasal, 
                         sa02, envelope_data, 
                         lower_envelope, recording, 
                         time_stamp_start,
                         time_stamp_stop,
                         labels_sub,
                         annotation_dict,
                         color_dict,
                         coloring=True,
                         ano_plot=False,
                         plot_only_clustering=False,
                         sub_clust=True,
                         predict=False,
                         plot_env=False
                         )
PCA_plotter(sub_clustering_array, interest_array, labels_sub, color_dict, recording, sub_clust=True)
#%%
from scipy.signal import savgol_filter
plt.plot(sa02-90)
sav_differ = savgol_filter(np.diff(sa02),31,2)
plt.plot(savgol_filter(np.diff(sa02),31,2))
for i in start_sat_index:
    plt.axvline(x=i/10, color='green')
    
#%%
fig, ax = plt.subplots()
x_axis = np.linspace(0, len(sa02) / 60, len(sa02))
ax.plot(x_axis,HR)

ax.set(title=('Smoothed Heart Rate Signal'.format(recording)))
ax.set(ylabel='Heart Rate [bpm]')
# ax.set(ylabel='Normalized Amplitude [ab]')
ax.set(xlabel='Time [minutes]')    
#%%

