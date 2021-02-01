# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:00:08 2020

@author: erfan pakdamanian
"""
# STEP1----------------- # Importing the libraries------------
#-------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.signal as ss
import wfdb
import csv
import sklearn






# STEP1------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
# Taking care of dataset
    
directory = os.chdir("\\ML4TakeOver\\Data\\RawData")   
file = "RawData"
dataframe_All = pd.read_csv("RawData.csv")        
  

 

## 
# STEP2 =========================== Removing Unnecessary columns========================
#========================================================================================

# 1) Removing the columns with more than 50% NAs
half_count = len(dataframe) / 2
dataframe = dataframe.dropna(thresh=half_count, axis=1) # Drop any column with more than 50% missing values
ColumnList =list(dataframe.columns)  

# These columns are not useful for our purposes
dataframe = dataframe.drop(['StudyName','ExportDate','Age','MediaTime',
                            'Internal ADC A13 PPG RAW (no units) (GSR)',
                            'VSenseBatt RAW (no units) (GSR)', #Battery status
                            'VSenseBatt CAL (mVolts) (GSR)',
                            'ValidityLeft',
                            'ValidityRight',
                            'Wide Range Accelerometer X CAL (m/(sec^2)) (GSR)',
                            'Wide Range Accelerometer X RAW (no units) (GSR)',
                            'Wide Range Accelerometer Y CAL (m/(sec^2)) (GSR)',
                            'Wide Range Accelerometer Y RAW (no units) (GSR)',
                            'Wide Range Accelerometer Z CAL (m/(sec^2)) (GSR)',
                            'Wide Range Accelerometer Z RAW (no units) (GSR)',
                            'Trust (0.0)',
                            'Gaze3DX', 'GazeX', 'GazeY',
                            'Gaze3DY',
                            'Gaze3DZ', 'AccX',
                            'AccY','AccZ'],axis=1) 


# 2) Removing columns with no distinct values
for col in dataframe.columns:
    if len(dataframe[col].unique()) == 1:
        dataframe.drop(col,inplace=True,axis=1)



# 3) Check which columns have the same values
dataframe['check'] = np.where((dataframe['ManualBrake (0.0)'] == dataframe['CurrentBrake (0.0)']), 
                      'TRUE', 'False')
dataframe['check'].value_counts()



# 4) Change some of the variables name
dataframe = dataframe.rename(columns={'Unnamed: 0': 'ID',
                                      'Mode (0.0)' : 'Mode',
                                      'Alarm (0.0)' : 'Alarm',
                                      'AutoBrake (0.0)':'AutoBrake',
                                      'AutoGear (0.0)':'AutoGear',
                                      'AutoThrottle (0.0)':'AutoThrottle',
                                      'AutoWheel (0.0)': 'AutoWheel',
                                      'CurrentBrake (0.0)':'CurrentBrake',
                                      'CurrentGear (0.0)':'CurrentGear', 
                                      'CurrentThrottle (0.0)':'CurrentThrottle', 
                                      'CurrentWheel (0.0)':'CurrentWheel',
                                      'EventN (0.0)': 'EventN',  
                                      'EventW (0.0)':'EventW',
                                      'MPH (0.0)':'MPH',
                                      'ManualBrake (0.0)': 'ManualBrake', 
                                      'ManualGear (0.0)':'ManualGear',
                                      'ManualThrottle (0.0)': 'ManualThrottle',
                                      'ManualWheel (0.0)':'ManualWheel',
                                      'PassBy (0.0)':'PassBy',
                                      'RangeN (0.0)':'RangeN', 
                                      'RangeW (0.0)':'RangeW',
                                      'RightLaneDist (0.0)':'RightLaneDist', 
                                      'RightLaneType (0.0)':'RightLaneType',
                                      'LeftLaneDist (0.0)': 'LeftLaneDist',
                                      'LeftLaneType (0.0)': 'LeftLaneType',
                                      'Speed (0.0)':'Speed'})


## 
# STEP3 ================== Making Data Sets for Each type of Data ==============================
#===============================================================================================

# EyeTracking Data

EyeDataFrame = dataframe[['Timestamp','ID','Name','FixationDuration', 
                          'FixationSeq', 
                          'FixationStart', 'FixationX','FixationY',
                          'GazeDirectionLeftX','GazeDirectionLeftY', 'GazeDirectionLeftZ', 
                          'GazeDirectionRightX','GazeDirectionRightY', 'GazeDirectionRightZ', 
                          'GazeLeftx', 'GazeLefty','PupilLeft','PupilRight',
                          'GazeRightx', 'GazeRighty', 'GazeVelocityAngle', 
                          'InterpolatedGazeX','InterpolatedGazeY',
                          'EventN', 'EventSource', 'EventW',
                          'Alarm', 'Mode']]


CarDataFrame = dataframe[['AutoBrake', 'AutoGear',
                          'AutoThrottle', 'AutoWheel', 'CurrentBrake',
                          'CurrentGear', 'CurrentThrottle', 'CurrentWheel',
                          'Distance3D',
                          'EventN', 'EventSource', 'EventW',
                          'MPH', 'ManualBrake', 'ManualGear',
                          'ManualThrottle', 'ManualWheel',
                          'PassBy','RangeN', 'RangeW',
                          'RightLaneDist', 'RightLaneType', 'LeftLaneDist',
                          'LeftLaneType',
                          'Speed']]


GsrDataFrame = dataframe[['GSR CAL (kOhms) (GSR)', 
                          'GSR CAL (µSiemens) (GSR)',
                          'GSR Quality (GSR)', 
                          #'GSR RAW (no units) (GSR)', Not important
                          'Heart Rate PPG  (Beats/min) (GSR)', 
                          'IBI PPG  (mSecs) (GSR)',
                          'Internal ADC A13 PPG CAL (mVolts) (GSR)',
                          'Packet reception rate RAW (no units) (GSR)',
                          'System Timestamp CAL (mSecs) (GSR)']]





## 
# STEP4================ Creating Individual dataset for furthur analysis =================================
#=========================================================================================================

# Concatinating the EyeData with CarData 
Eye_Car_DataFrame = pd.concat([EyeDataFrame, CarDataFrame], axis=1, sort=False)
Eye_Car_DataFrame.to_csv("Eye_Car_Data" + '.csv')




