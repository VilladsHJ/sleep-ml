import numpy as np
import pickle
from Data_loader import EDF_function, annotation_finder
from Preprocessing_script import HR_SA02_preprocessor, movement_artifacts, segment_normalizer, preprocessor, frequency_criteria
from segmentation_functions import enveloper, normal_finder, piecewise_lin_oxy_delay, segmentation_function
from feature_extraction_functions import correlation_feat, spectral_density_feature, arclen_function, wavelet_feature



def cluster_function(start, stop, channel, loading=True, annotated=False):

    
    """
    
    
    Input:
        Start       (integer from 0 to 5799 that indicates which recording 
                     from the SHHS database the algorithm should start 
                     processing)
        stop        (integer that is > start that indicates which recording
                     from the SHHS database where the algorithm should stop 
                     processing)
        channel     ('nasal', 'abdo' or 'thor' relates to which signal should 
                     be processed. Note that the nasal airflow is always 
                     processed)
        loading     (If the feature array has already been saved this can be
                     loaded if loading=True, otherwise loading=False if no 
                     feature array has been saved before)
        annotated   (If annoated=true then only the annotated segments are 
                     being used for feature extraction, otherwise all segments 
                     found by the segmentation algorithm is processed)
    
    The cluster function is split into sections.
    First section contains implementation of exclusion criteria set up
    to ensure data quality of the recording being processed.
    
    Second section contains the segmentation algorithm and a variety of data
    extracted from this step for visualization and feature extraction purposes.
    
    Third section contains feature extraction and construction of a feature
    array used for clustering.
    
    Output:
        number,                 (the number of the recording)
        start_index,            (array of start indices)
        stop_index,             (array of stop indices)
        start_sat_index,        (array of saturation start indices)
        stop_sat_index,         (array of saturation stop indices)
        mover,                  (array of delays)
        data_processed,         (processed signal data)
        wake_start,             (array of wake start indices)
        wake_stop,              (array of wake stop indices)
        envelope_data,          (upper envelope data for processed signal)
        lower_envelope,         (lower envelope data for processed signal)
        clustering_array,       (feature array of current recording)
        final_clustering_array, (same as clustering_array if only one
                                 recording is being processed)
        peak_f,                 (the peak frequency of the recording)
        events_in_recording,    (number of events in recording)
        sa02,                   (the processed oxygen saturation signal)
        time_stamp_start,       (the start index of the normal segment)
        time_stamp_stop         (the stop index of the normal segment)
        
    
    """
    
    for i in range(start, stop):
        
        # the numbers of the SHHS database recordings
        number = int(200001+i)
        
            
        print("Processing recording number {}".format(number))
        _, sa02, HR, _, _, _, _, _, _, thor_resp, abdo_resp, _, _, airflow, nasal_pressure, ox_stat = EDF_function(number)
        
        """
        EXCLUSION CRITERIA BEGIN:
            
        """
        
        # Checks if the channels are integers which means that the data loader
        # did not find a recording
        if type(sa02) == int or type(thor_resp) == int or type(abdo_resp) == int:
            print('One of the sa02, thor or abdo signals are integers which indicate that the recording does not exist')
            continue
        
        
        # Check for correct nasal airflow array length    
        _, recording_length = annotation_finder(number, parameter='length')
        

        if int(recording_length) == len(nasal_pressure):
            data_array_nasal = np.array(nasal_pressure, dtype=float)
        elif int(recording_length) == len(airflow):
            data_array_nasal = np.array(airflow, dtype=float)
        elif int(recording_length) == len(ox_stat):
            data_array_nasal = np.array(ox_stat, dtype=float)
                 
        data_array = []
        if channel == 'nasal':
            if int(recording_length) == len(nasal_pressure):
                data_array = np.array(nasal_pressure, dtype=float)
                data_processed = preprocessor(data_array)
            elif int(recording_length) == len(airflow):
                data_array = np.array(airflow, dtype=float)
                data_processed = preprocessor(data_array)
            elif int(recording_length) == len(ox_stat):
                data_array = np.array(ox_stat, dtype=float)
                data_processed = preprocessor(data_array)
            else:
                print("No array with the correct length. This recording is skipped.")
                if number == 200001+stop:
                    break
                else:
                    continue
                
        """
        CHECK FOR CORRUPT BELT SIGNALS
        """
        
        diff_abdo = np.diff(abdo_resp)
        diff_abdo = np.append(diff_abdo, 0.1)
        flatliner_abdo = 100*len(np.where(diff_abdo == 0)[0])/len(abdo_resp)
        
        
        diff_thor = np.diff(thor_resp)
        diff_thor = np.append(diff_thor, 0.1)
        flatliner_thor = 100*len(np.where(diff_thor == 0)[0])/len(thor_resp)
                  
        
        if flatliner_abdo > 50:
            #print('{} is skipped because the abdominal belt signal is corrupt ({}% flatline)'.format(number, flatliner_abdo))
            continue
        if flatliner_thor > 50:
            #print('{} is skipped because the thorax belt signal is corrupt ({}% flatline)'.format(number, flatliner_thor))
            continue

        elif channel == 'thor':
            data_array = np.array(thor_resp, dtype=float)
            data_processed = data_array
        elif channel == 'abdo':
            data_array = np.array(abdo_resp, dtype=float)
            data_processed = data_array        


        """ 
        Finding wake and movement artifacts
        """
        wake_start, wake_stop = movement_artifacts(abdo_resp, number)        
        ranger = []
        for k in range(len(wake_start)):
            ranger = np.append(ranger, range(int(wake_start[k]), int(wake_stop[k])))
        
        
        """
        Checking for dominant frequency in data arry
        """
        
        peak_f = frequency_criteria(data_array, ranger)
        # Skipping recording if peak frequency in nasal airflow is too high or too low (faulty recordings).
        if channel == 'nasal':
            if 2 < peak_f or peak_f < 0.1:
                print("The dominant frequency is: {} thus recording {} is skipped".format(peak_f, number))
                continue
        
        
        """
        Preprocessing oxygen saturation signal and heart rate signal
        """
        
        sa02 = HR_SA02_preprocessor(sa02,65,100)
        HR = HR_SA02_preprocessor(HR, 20,200)
        
        
        diff_HR = np.diff(HR)
        diff_HR = np.append(diff_HR, 0)
        

        
        # Nasal airflow specific:
        data_processed_nasal = preprocessor(data_array_nasal)
        data_processed_nasal = segment_normalizer(data_processed_nasal, wake_start, wake_stop)
        
        # Nasal airflow or belts
        data_processed = segment_normalizer(data_processed, wake_start, wake_stop)


        """
        EXLUSION AND PREPROCESSING COMPLETE
        """
        
        ###################################
        
        """
        FEATURE EXTRACTION BEGIN
        """
        
        if annotated==False: # If not(!) looking at annotated segments specifically.
            """
            Finding start and stop indices for segmentation function
            """
            envelope_data_idx = enveloper(data_processed_nasal.copy()**2)
            start_index, stop_index, segments_half = segmentation_function(envelope_data_idx,
                                                                           wake_start,
                                                                           wake_stop,
                                                                           data_processed_nasal.copy()**2)

        else: 
            """
            This is only if looking at annotations as segments specifically
            """
            start_index, stop_index = annotation_finder(number, parameter='hypopneas')#
            start_index_ob, stop_index_ob = annotation_finder(number, parameter='obstructive')
            start_index = np.append(start_index, start_index_ob)
            stop_index = np.append(stop_index, stop_index_ob)
        
        
        # Extracting the upper and lower envelope of the nasal airflow signal
        envelope_data_nasal = enveloper(data_processed_nasal)
        envelope_data_nasal_diff = np.diff(envelope_data_nasal)
        envelope_data_nasal_diff = np.append(envelope_data_nasal_diff, envelope_data_nasal_diff[-1])
        lower_envelope_nasal = enveloper(data_processed_nasal.copy() * -1)
        lower_envelope_nasal = lower_envelope_nasal*-1

            
        # Extracting upper envelope for visualization and feature extraction:
        envelope_data = enveloper(data_processed) 
        # Extracting upper envelope for visualization and feature extraction:
        lower_envelope = enveloper(data_processed.copy() * -1)
        lower_envelope = lower_envelope*-1
        
        # Difference between upper and lower envelopes for feature extraction:
        envelope_difference = envelope_data.copy()-lower_envelope.copy()
        

        # Extracting 'normal segment':
        normal_segment, time_stamp_start, time_stamp_stop = normal_finder(sa02,
                                                                          data_processed_nasal,
                                                                          envelope_data_nasal,
                                                                          lower_envelope_nasal,
                                                                          ranger,
                                                                          duration=4)
        
        # Ensuring correct datatype for time stamps:
        time_stamp_start, time_stamp_stop = int(time_stamp_start), int(time_stamp_stop)
        
        # Feature exraction; mean of normal env:
        normal_env_mean = np.mean(envelope_data_nasal[time_stamp_start:time_stamp_stop])
        
        # Feature extraction;   difference between the mean of the upper
        #                       envelope and the mean of the lower envelope
        normal_env_mean_difference = np.mean(envelope_difference[time_stamp_start:time_stamp_stop])
        
        
        # finding
        sa02_diff = np.diff(sa02, 1)
        sa02_diff = np.append(sa02_diff, sa02_diff[-1])


        # start_sat_index, stop_sat_index, mover = piecewise_lin_oxy_delay(HR, sa02, start_index, stop_index)
        start_sat_index, stop_sat_index, mover, delay_array = piecewise_lin_oxy_delay(sa02, envelope_data_nasal, start_index, stop_index)
        # from scipy.signal import find_peaks, savgol_filter
        # peaks_top, peaks_ys = find_peaks(savgol_filter(sa02, 31, 2), distance=10, height=0)
        
        # new_mover = []
        # peaks_used = []
        # index_used = []
        # for i in range(len(segments_half)-1):
        #     condi = False
        #     q=5
        #     while condi == False:
        #         if segments_half[i]//10 + q in peaks_top:
        #             if segments_half[i]//10 + q in peaks_used:
        #                 condi = True
        #             elif segments_half[i]//10 + q >= segments_half[i+1]//10:
        #                 condi = True
        #             else:
        #                 new_mover = np.append(new_mover, q)
        #                 current_peak = segments_half[i]//10 + q
        #                 peaks_used = np.append(peaks_used, current_peak)
        #                 index_used = np.append(index_used, segments_half[i])
        #                 condi = True
        #                 continue
        #         q+=1
        #         if q == 45:
        #             condi = True
        #             new_mover = np.append(new_mover, 15)
        #             current_peak = segments_half[i]//10 + 15
        #             peaks_used = np.append(peaks_used, current_peak)
        #             index_used = np.append(index_used, segments_half[i])
        #     continue
        
        # mover = []
        # k = 0
        # for i in range(len(start_index)):
        #     if i < k:
        #         continue
        #     if start_index[i] in index_used:
        #         place = np.where(index_used == start_index[i])[0][0]
        #         k = i+2
        #         mover = np.append(mover, new_mover[place]*10)
        #         mover = np.append(mover, new_mover[place]*10)
        #     else:
        #         mover = np.append(mover, 150)
        #         mover = np.append(mover, 150)
        #         k = i+2
        
        # if len(mover) > len(start_index):
        #     mover = np.delete(mover,-1)
        
        # start_sat_index = start_index + mover
        
        # stop_sat_index = stop_index + mover
        # for i in range(len(stop_sat_index)):
        #     if stop_sat_index[i] > len(data_processed):
        #         stop_sat_index[i] = len(data_processed)
                
        # print("Extracting Wavelet Feature")
        

        wavelet_inters = wavelet_feature(envelope_data, start_index, stop_index)
        
        
        clustering_array = []
        print("Clustering array created")
        print("Appending values from recording {}".format(number))
        print("Number of events: {}".format(len(start_index)))
        
        # BEGIN FEATURE EXTRACTION:
        hypo_70 = 0
        for idx in range(len(start_index)):
            if  start_index[idx] in ranger: # No feature extraction in wake segments
                continue
            elif stop_index[idx] in ranger: # No feature extraction in wake segments
                continue
            else:
                segment_for_computations = data_processed[int(start_index[idx]):int(stop_index[idx])]
                segment_saturation_diff = sa02_diff[int(start_sat_index[idx]/10):int(stop_sat_index[idx]/10)]
                
                segment_saturation = sa02[int(start_sat_index[idx]/10):int(stop_sat_index[idx]/10)]
                segment_heart = HR[int(start_index[idx]/10):int(stop_index[idx]/10)]
                segment_heart_diff = diff_HR[int(start_index[idx]/10):int(stop_index[idx]/10)]
            if segment_saturation.size == 0 or segment_saturation_diff.size  == 0 or segment_for_computations.size == 0:
                continue
            elif len(segment_for_computations) < 80:
                continue
            else:
                envelope_segment = envelope_data[int(start_index[idx]):int(stop_index[idx])]
                envelope_segment_diff = envelope_data_nasal_diff[int(start_index[idx]):int(stop_index[idx])]
                lower_envelope_segment = lower_envelope[int(start_index[idx]):int(stop_index[idx])]
                
                correlation_feature = correlation_feat(envelope_segment, lower_envelope_segment)
                
                saturation_feature_diff = np.mean(segment_saturation_diff) 
                saturation_feature = (np.max(segment_saturation)-np.min(segment_saturation))/(np.max(segment_saturation)+0.001)
                saturation_feature_raw = np.max(segment_saturation)-np.min(segment_saturation)
                sign_sat = (saturation_feature_diff/(abs(saturation_feature_diff)+0.000001))
                satc_inter = np.trapz(segment_saturation_diff)
                
                HR_feature_diff = np.mean(segment_heart_diff)
                heart_feat_raw = np.max(segment_heart)-np.min(segment_heart)
                sign_HR = (HR_feature_diff/(abs(HR_feature_diff)+0.000001))
                combined_heart_feat = sign_HR*heart_feat_raw
                combined_sat_feat = saturation_feature_raw*sign_sat
                segment_env_mean_difference = np.mean(envelope_difference[int(start_index[idx]):int(stop_index[idx])])
                arc_len_sat = (sign_sat*arclen_function(abs(segment_saturation_diff)))#/(len(segment_saturation_diff)/60)
                
                sign = sum(envelope_segment_diff[0:int(len(envelope_segment_diff)//2)])
                sign_use = sign/(abs(sign)+0.00000000001)
                if sign_use <= 0:
                    min_max_envelopes = min(envelope_segment)-max(lower_envelope_segment)
                else:
                    min_max_envelopes = max(lower_envelope_segment)-min(envelope_segment)
                
                avg_max_val = min(envelope_segment)/max(envelope_segment)
                arc_len = sign_use*arclen_function(abs(envelope_segment_diff))#/(len(envelope_segment)/600)
                
                if avg_max_val < 0.70 and sign_use < 0:
                    # max_min_ratio_env = sign_use*abs(max(envelope_segment)-min(envelope_segment))
                    hypo_70 = 1
                else:
                    hypo_70 = 0
                    # max_min_ratio_env = 0
                max_min_ratio_env = sign_use*abs(max(envelope_segment)-min(envelope_segment))
                # difference between  normal and segment of the mean of the upper envelope
                baseline_up_env_difference = normal_env_mean - np.mean(envelope_segment) 
                
                # difference between normal and segment of the difference between the upper and lower envelope for the normal
                baseline_env_mean_difference = normal_env_mean_difference-segment_env_mean_difference 
                 
 
                psd_feature, peak_norm, peak_seg = spectral_density_feature(normal_segment, segment_for_computations)

                # Setting up AASM rules for prediction
                if combined_sat_feat <= -2:
                    sat_score = 1
                else:
                    sat_score = 0
                
                if min(envelope_segment)/normal_env_mean <= 0.70 or avg_max_val <= 0.70:
                    sig_drop = 1
                else: 
                    sig_drop = 0
                
                if sat_score == 1 and sig_drop == 1:
                    hypo = 1
                else:
                    hypo = 0
                
                if min(envelope_segment)/normal_env_mean <= 0.1:
                    hypo = 1
                

                if str(correlation_feature) == 'nan':
                    correlation_feature = 0
                    print('Replaced a "nan" at idx = {}'.format(idx))

                # Initializing feature array
                if np.size(clustering_array) == 0:
                    clustering_array = [hypo,
                                        hypo_70,
                                        combined_sat_feat,              # Oxygen saturation feature (SatC)
                                        baseline_env_mean_difference,   # Envelope feature (BED)
                                        psd_feature,                    # Ratio of power spectral density (rPSD)
                                        correlation_feature,            # Correlation of upper and lower envelope (CorE)
                                        arc_len,
                                        combined_heart_feat,
                                        wavelet_inters[10][idx],
                                        wavelet_inters[20][idx],
                                        wavelet_inters[30][idx],
                                        wavelet_inters[40][idx],
                                        wavelet_inters[50][idx],
                                        max_min_ratio_env, 
                                        #min_max_envelopes,
                                        satc_inter,
                                        int(start_sat_index[idx]),
                                        int(stop_sat_index[idx]),
                                        start_index[idx],
                                        stop_index[idx],
                                        ]
                    print("Clustering array initialized")
                else: # Stacking features for each segment
                    clustering_array_values = [hypo,
                                               hypo_70,
                                               combined_sat_feat,              # Oxygen saturation feature (SatC)
                                               baseline_env_mean_difference,   # Envelope feature (BED)
                                               psd_feature,                    # Ratio of power spectral density (rPSD)
                                               correlation_feature,            # Correlation of upper and lower envelope (CorE)
                                               arc_len,
                                               combined_heart_feat,
                                               wavelet_inters[10][idx],
                                               wavelet_inters[20][idx],
                                               wavelet_inters[30][idx],
                                               wavelet_inters[40][idx],
                                               wavelet_inters[50][idx],
                                               max_min_ratio_env, 
                                               #min_max_envelopes,
                                               satc_inter,
                                               int(start_sat_index[idx]),
                                               int(stop_sat_index[idx]),
                                               start_index[idx],
                                               stop_index[idx],
                                               ]
                    
                    clustering_array = np.vstack([clustering_array, 
                                                  clustering_array_values])

                
        if number == 200001+start:
            shape_0 = []
            shape_0 = np.append(shape_0, int(clustering_array.shape[0]))
            new_s = []
            new_s = np.append(new_s, 0)
            startstop = np.array([new_s[-1], shape_0[-1],number])
            events_in_recording = startstop
        else:
            shape_0 = np.append(shape_0, int(clustering_array.shape[0]))
            new_s = np.append(new_s, int(new_s[-1]+shape_0[-2]))
            events_in_recording = np.vstack([events_in_recording, 
                                             [new_s[-1], 
                                              shape_0[-1]+1,
                                              number]])
            
    
        if number == 200001+start:
            final_clustering_array = clustering_array
        else:
            final_clustering_array = np.vstack([final_clustering_array, clustering_array])
        
        if not loading:
            print("Pickling clustering array for recording {}".format(number))
            if annotated==True:
                file_cluster = 'ano_feature_array_'+str(channel)+'_'+str(number)
            else:
                file_cluster = 'seg_feature_array_'+str(channel)+'_'+str(number)
            cluster_file = open(file_cluster, 'wb')
            pickle.dump(clustering_array, cluster_file)
            cluster_file.close()
            print("Pickle file saved")
    
     
    return (number,
            start_index,
            stop_index,
            start_sat_index,
            stop_sat_index,
            mover,
            data_processed,
            wake_start,
            wake_stop,
            envelope_data,
            lower_envelope,
            clustering_array,
            final_clustering_array,
            peak_f,
            events_in_recording,
            sa02,
            time_stamp_start,
            time_stamp_stop,
            delay_array)