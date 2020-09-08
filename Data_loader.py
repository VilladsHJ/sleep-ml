# -*- coding: utf-8 -*-
"""
Data loader script
The script utilizes the package pedflib to load .edf files and the package
xml to load the annotation sheet xml files

Files required: SHHS database edf file with number starting from 200001 and up.

@author: Villads

"""

# EDF Loader

import pyedflib
import xml.etree.ElementTree as ET
import numpy as np
import os


def EDF_function(number):
    
    """ 
    This loads edf files from the SHHS database by utilizing the numeration
    of the data files. It checks if the file exist in the directory.
    If the file exists the data is loaded, if not each channel becomes 0.
    
    Input:
        The number of the recording to load
    
    Output:
        EDF_file        (The edf file containing all the signals below) 
        sa02            (The oxygen saturation signal)
        HR              (The heart rate signal)
        EEG_sec         (Electroencephalography signal)
        ECG             (Electrocardiography signal)
        EMG             (Electromyography signal) 
        EOG_L           (Electroocculography signal (left eye)) 
        EOG_R           (Electroocculography signal (left eye))
        EEG             (Electroencephalography signal)
        thor_resp       (Thoracic belt signal)
        abdo_resp       (Abdominal belt signal)
        position        (Body position signal) 
        light           (Lights on/off) 
        airflow         (Airflow signal)*
        nasal_pressure  (Airflow signal)* 
        ox_stat         (Airflow signal)*
                                            *(airflow is recorded in different
                                              channels, but only one channel 
                                              is used at any time)
    
    """
    
    
    
    if os.path.exists(r'C:\Users\Villads\OneDrive - Danmarks Tekniske Universitet\Stanford\Lab Work\Data\SHHS\shhs1-'+ str(number) +'.edf'):
        EDF_file = r'C:\Users\Villads\OneDrive - Danmarks Tekniske Universitet\Stanford\Lab Work\Data\SHHS\shhs1-'+ str(number) +'.edf'
        f = pyedflib.EdfReader(EDF_file)
        sa02 = f.readSignal(0)
        HR = f.readSignal(1)
        EEG_sec = f.readSignal(2)
        ECG = f.readSignal(3)
        EMG = f.readSignal(4)
        EOG_L = f.readSignal(5)
        EOG_R = f.readSignal(6)
        EEG = f.readSignal(7)
        thor_resp = f.readSignal(8)
        abdo_resp = f.readSignal(9)
        position =  f.readSignal(9)
        light = f.readSignal(10)
        airflow =  f.readSignal(11)
        nasal_pressure = f.readSignal(12)
        ox_stat =  f.readSignal(13)
    else:
        EDF_file = 0
        sa02 = 0
        HR = 0
        EEG_sec = 0
        ECG = 0
        EMG = 0
        EOG_L = 0
        EOG_R = 0
        EEG = 0
        thor_resp = 0
        abdo_resp = 0
        position =  0
        light = 0
        airflow =  0
        nasal_pressure = 0
        ox_stat =  0
    return EDF_file, sa02, HR, EEG_sec, ECG, EMG, EOG_L, EOG_R, EEG, thor_resp, abdo_resp, position, light, airflow, nasal_pressure, ox_stat

def annotation_finder(number, parameter='wake'):
    
    """
    Input:
        Number      (The recording number of the SHHS datafile)
        Paramenter  (Event type: 'wake', 'hypopneas', 'obstructive', arousals,
                                 'cental', length)
        
    Output:
        start       (array of start indicices of each event)
        stop        (array of stop indicidies of each event)
    """
    
    # This functions finds the annotated  events
    # The default event parameter is 'wake'.
    # Otherwise it can find the annotated arousals if parameter = 'arousals'.
    
    # The functions outputs the start sample and stop sample of the chosen event type.
    path = 'C:/Users/Villads/OneDrive - Danmarks Tekniske Universitet/Stanford/Lab Work/Data/SHHS/'
    
    file_name = 'shhs1-'+str(number)+'-nsrr.xml'
    
    tree = ET.parse(path+file_name)
    
    root = tree.getroot()
    
    if str(parameter) == 'wake':
        start = []
        stop = []
        
        for child in root[2]:
            if child[1].text == 'Wake|0': #and float(child[3].text) >= 90.0:
                start = np.append(start, float(child[2].text)*10)
                stop_value = float(child[2].text)*10 + float(child[3].text)*10
                stop = np.append(stop, stop_value)
    
    if str(parameter) == 'arousals':
        start = []
        stop = []
        
        for child in root[2]:
            if child[0].text == 'Arousals|Arousals':
                start = np.append(start, float(child[2].text)*10)
                stop_value = float(child[2].text)*10 + float(child[3].text)*10
                stop = np.append(stop, stop_value)

    if str(parameter) == 'hypopneas':
        start = []
        stop = []
        
        for child in root[2]:
            if child[1].text == 'Hypopnea|Hypopnea':
                start = np.append(start, float(child[2].text)*10)
                stop_value = float(child[2].text)*10 + float(child[3].text)*10
                stop = np.append(stop, stop_value)
                
    if str(parameter) == 'obstructive':
        start = []
        stop = []
    
        for child in root[2]:
            if child[1].text == 'Obstructive apnea|Obstructive Apnea':
                start = np.append(start, float(child[2].text)*10)
                stop_value = float(child[2].text)*10 + float(child[3].text)*10
                stop = np.append(stop, stop_value)
    
    if str(parameter) == 'central':
        start = []
        stop = []
    
        for child in root[2]:
            if child[1].text == 'Central apnea|Central Apnea':
                start = np.append(start, float(child[2].text)*10)
                stop_value = float(child[2].text)*10 + float(child[3].text)*10
                stop = np.append(stop, stop_value)
            else:
                continue
    

    if str(parameter) == 'length':
        for child in root[2]:
            if child[1].text == 'Recording Start Time': #and float(child[3].text) >= 90.0:
                start = 0
                stop = float(child[3].text)*10
    
    if start == [] or stop == []:
        print('No {} events found'.format(parameter))
        start = 0
        stop = 1
    
    return start, stop