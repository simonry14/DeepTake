# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:56:08 2020

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
import csv
import sklearn
from quilt.data.ResidentMario import missingno_data
import missingno as msno 
import seaborn as sns
from sklearn.impute import SimpleImputer 



# STEP2------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
os.chdir("\\ML4TakeOver\\Data\\RawData")
directory = os.getcwd()
dataFrame_takeover = pd.read_csv('takeover_Alarm_Eye_Car_Data_10sec.csv')


dataFrame_takeover = dataFrame_takeover.drop(['Unnamed: 0','Unnamed: 0.1',
                                              'CurrentGear','GazeVelocityAngle','GazeRightx', 'GazeRighty',
                                              'AutoGear','AutoBrake','GazeLeftx', 'GazeLefty'], axis=1)

## CHANGING FALSE ALARM TO TRUE ALARM FOR FIRST few participants CHECK IF THEY SHOULD HA
searchforSeries = ['004','005','006','007','008']
dataFrame_takeover.loc[(dataFrame_takeover['Name'].str.contains('|'.join(searchforSeries))), 'Coming_AlarmType'] = 'TA'


# STEP5============================ Adding NoneDriving Task column ======================
#========================================================================================
### creat Task column
# map task to the alarm
TaskAlarm = {'Reading' : [16,84,103,339],
             'Cell': [5, 259, 284, 323],
             'Talk': [137, 178, 185, 332],
             'Question': [213, 254, 191]}

dataFrame_takeover['NDTask'] = 'XXX'
dataFrame_takeover.loc[dataFrame_takeover['Coming_Alarm'].isin(TaskAlarm['Reading']), 'NDTask'] = 'Reading'    # reading task
dataFrame_takeover.loc[dataFrame_takeover['Coming_Alarm'].isin(TaskAlarm['Cell']), 'NDTask'] = 'Cell'    # cell task
dataFrame_takeover.loc[dataFrame_takeover['Coming_Alarm'].isin(TaskAlarm['Talk']), 'NDTask'] = 'Talk'    # talk task
dataFrame_takeover.loc[dataFrame_takeover['Coming_Alarm'].isin(TaskAlarm['Question']), 'NDTask'] = 'Question'    # question task


#================= Visualizing "TakeOver/Not-takeover" for each Alarm type ========================
#==========================================================================================================

# Remove 000 from data for visualization

# we don't have zero alarm type anymore
dataFrame_Alarm = dataFrame_takeover

# check the number of user's per alarm
tmp_result = pd.DataFrame(dataFrame_Alarm.groupby(['Coming_Alarm']).agg({'Name': 'unique'}).reset_index()) 
[len(a) for a in tmp_result['Name']]

tmp2 = pd.DataFrame(dataFrame_Alarm.groupby(['Name']).agg({'Coming_Alarm': 'unique'}).reset_index())
[len(a) for a in tmp2['Coming_Alarm']]

# How many takeover and not-takeover per alarm?
dataFrame_Alarm.groupby(['Coming_AlarmType','Takeover']).size().plot(kind = 'barh', legend = False)    # Frequency Based
plt.show()
dataFrame_Alarm.groupby(['Coming_AlarmType','Takeover']).agg({"Name": lambda x: x.nunique()}).plot(kind = 'barh', legend = False)



# Takeover frequency per individuals
tmp_dataframe = pd.DataFrame(dataFrame_Alarm.groupby(['Name', 'Coming_AlarmType','Takeover']).agg({"Coming_Alarm": lambda x: x.nunique()}))
tmp_dataframe.to_csv("UserComingAlarmType"+'.csv')

dataFrame_Alarm.groupby(['Name', 'Coming_AlarmType']).agg({"Takeover": lambda x: x.nunique()})
dataFrame_Alarm.groupby(['Name', 'Coming_AlarmType','Takeover']).size().unstack().plot(kind = 'bar', stacked = True)
dataFrame_AlarmIndividual = pd.DataFrame(dataFrame_Alarm.groupby(['Name', 'Coming_AlarmType','Takeover']).size().reset_index(name = 'frequency'))

pd.DataFrame(tmp_dataframe).transpose().to_csv("UserComingAlarmType_2"+'.csv')
dataframe_tmp = pd.DataFrame(tmp_dataframe).transpose()
tmp_dataframe = tmp_dataframe.reset_index()
sns.barplot(x='Name', y = 'Coming_Alarm', hue ='Coming_AlarmType', data = tmp_dataframe)
sns.barplot(x='Name', y = 'Coming_Alarm', hue ='Takeover', data = tmp_dataframe)


## Number of Takeovers
## Counting the number of value changes in AlarmType
np.count_nonzero(np.diff(dataFrame_Alarm['Coming_Alarm']))     #655


# Takeover rate per each Alarm
dataFrame_Alarm.groupby(['Alarm', 'Coming_AlarmType','Takeover']).size().unstack().plot(kind = 'barh', stacked = True)  # Frequency Based

takeover_falseAlarm_frequency = dataFrame_Alarm.groupby(['Coming_Alarm', 'Coming_AlarmType','Takeover']).agg({"Name": lambda x: x.nunique()})
#takeover_falseAlarm_frequency.to_csv("takeover_falseAlarm_frequency.csv")
dataFrame_Alarm.groupby(['Coming_Alarm', 'Coming_AlarmType','Takeover']).agg({"Name": lambda x: x.nunique()}).unstack().plot(kind = 'barh', stacked = True) 




#================= statistical analysis and Visualization for "REACTION TIME" =======================================
#====================================================================================================================

###Step1. STD and Mean
reactionTime_mean = np.mean(dataFrame_takeover[dataFrame_takeover.ReactionTime != 0]['ReactionTime'])
reactionTime_std = np.std(dataFrame_takeover[dataFrame_takeover.ReactionTime != 0]['ReactionTime'])

plt.scatter(range(len(dataFrame_takeover.ReactionTime.unique())), dataFrame_takeover.ReactionTime.unique())
plt.xlabel('index')
plt.ylabel('Reaction Time')

dataFrame_takeover['TOT_Class']= 000
# consider the mean of the reaction time as a threshold to classify the users (more > mean ==> 1) and (less < mean ==> 0)
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'] > reactionTime_mean, 'TOT_Class'] = 1  # Slower reaction Times




#================= making a column for 3 classes of Reaction Time =======================================
#====================================================================================================================

###Step1. STD and Mean
reactionTime_mean = np.mean(dataFrame_takeover[dataFrame_takeover.ReactionTime != 0]['ReactionTime'])
reactionTime_std = np.std(dataFrame_takeover[dataFrame_takeover.ReactionTime != 0]['ReactionTime'])

plt.scatter(range(len(dataFrame_takeover.ReactionTime.unique())), dataFrame_takeover.ReactionTime.unique())
plt.xlabel('index')
plt.ylabel('Reaction Time')

dataFrame_takeover['TOT_Three_Class']= 000
# consider the mean of the reaction time as a threshold to classify the users (more > mean ==> 1) and (less < mean ==> 0)
lower_bound = reactionTime_mean-reactionTime_std
upper_bound = reactionTime_mean + reactionTime_std
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'].between(lower_bound, upper_bound , inclusive = True), 
                       'TOT_Three_Class'] = 1  # midum reaction Times
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'] > upper_bound, 
                       'TOT_Three_Class'] = 2  # Slowest
                       
                       
#================= making a column for 5 classes of Reaction Time =======================================
#====================================================================================================================

dataFrame_takeover['TOT_Five_Class']= 000
# consider the mean of the reaction time as a threshold to classify the users (more > mean ==> 1) and (less < mean ==> 0)
lowest_bound    =  reactionTime_mean - 2*reactionTime_std
lower_bound     =  reactionTime_mean - reactionTime_std
upper_bound     =  reactionTime_mean + reactionTime_std
uppermost_bound =  reactionTime_mean + 2*reactionTime_std

# the 'Fastest' Reaction Time will be Zero
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'].between(lowest_bound, lower_bound , inclusive = True), 
                       'TOT_Five_Class'] = 1  # Fast reaction Times           
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'].between(lower_bound, upper_bound , inclusive = True), 
                       'TOT_Five_Class'] = 2  # midum reaction Times
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'].between(upper_bound , uppermost_bound , inclusive = True), 
                       'TOT_Five_Class'] = 3  # Slow reaction Times                   
dataFrame_takeover.loc[dataFrame_takeover['ReactionTime'] > uppermost_bound, 
                       'TOT_Five_Class'] = 4  # Slowest reaction Times  
                       
tmp = dataFrame_takeover[dataFrame_takeover.Takeover_Quality != 0]
tmp.groupby(['TOT_Three_Class', 'Takeover_Quality']).agg({"Name": lambda x: x.nunique()}).unstack().plot(kind = 'barh', stacked = False) 

#dataFrame_takeover.to_csv("QualityTakeover_4ML"+".csv")
 
                      
#================= Defining Threshold for Reaction Time =============================================================
# How many datapoints are higher than Threshold
TOT_unique = dataFrame_takeover.ReactionTime.unique()
high_RT = 0
for element in TOT_unique:
    if element > 4000:
        high_RT = high_RT + 1
high_RT/len(TOT_unique)  #
sns.scatterplot(x=range(len(dataFrame_takeover.ReactionTime.unique())), y=dataFrame_takeover.ReactionTime.unique(), data=dataFrame_takeover)


###-------
#dataFrame_takeover.to_csv("quality_takeover_4ml.csv")
# Save for ML analysis
dataFrame_takeover.to_csv("takeover" + '4ML' + '.csv')

