#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:47:10 2019

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
from sklearn.impute import SimpleImputer 


# STEP2------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
os.chdir("\\ML4TakeOver\\Data\\RawData")
directory = os.getcwd()

#folderPath = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
dataFrame_Eye_Car = pd.read_csv('Eye_Car_Data.csv')



# STEP3====================== Imputation of DataSet ======================================
#========================================================================================

dataFrame_Eye_Car = dataFrame_Eye_Car.drop(['Unnamed: 0','EventN.1', 'EventSource.1', 'EventW.1'], axis=1)

# Select duplicate rows except first occurrence based on all columns
duplicateRowsDF = dataFrame_Eye_Car[dataFrame_Eye_Car.duplicated()]
print(duplicateRowsDF)

# Select all duplicate rows based on multiple column names in list
duplicateRowsDF = dataFrame_Eye_Car[dataFrame_Eye_Car.duplicated(['Timestamp',
                                                                  'FixationDuration', 'FixationSeq', 'FixationStart', 'FixationX',
                                                                  'FixationY', 'GazeDirectionLeftX', 'GazeDirectionLeftY',
                                                                  'GazeDirectionLeftZ', 'GazeDirectionRightX', 'GazeDirectionRightY',
                                                                  'GazeDirectionRightZ', 'GazeLeftx', 'GazeLefty', 'PupilLeft',
                                                                  'PupilRight', 'GazeRightx', 'GazeRighty', 'GazeVelocityAngle'], keep='first')]
print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')




# DROPPING all duplicate values except the first ones
dataFrame_Eye_Car.drop_duplicates(subset=['Timestamp','FixationDuration', 'FixationSeq', 'FixationStart', 'FixationX',
                                          'FixationY', 'GazeDirectionLeftX', 'GazeDirectionLeftY','GazeDirectionLeftZ', 
                                          'GazeDirectionRightX', 'GazeDirectionRightY',
                                          'GazeDirectionRightZ', 'GazeLeftx', 'GazeLefty', 'PupilLeft',
                                          'PupilRight', 'GazeRightx', 'GazeRighty', 'GazeVelocityAngle'], keep='first',inplace=True)


# Removing the Rows with NAN value which started before the actual session (iMotions always start earlier)
dataFrame_Eye_Car = dataFrame_Eye_Car[np.logical_not(dataFrame_Eye_Car['Mode'].isnull())]
dataFrame_Eye_Car = dataFrame_Eye_Car[np.logical_not(dataFrame_Eye_Car['Speed'].isnull())]

# Checking how many features have NaN values
pd.isnull(dataFrame_Eye_Car).sum()
# Almost all Car data DON'T have NaN value

 
###############################################
tmpDataFrame = dataFrame_Eye_Car
msno.matrix(tmpDataFrame) 

# Keep those columns that have at least 20 percent data
atLeast80Data = msno.nullity_filter(tmpDataFrame, filter='top', p = 0.20)
#msno.matrix(atLeast80Data, labels=True, fontsize=10)

# show the  value range of each column
sample_m = atLeast80Data
sample_m.describe().T.loc[:, ('count', 'mean', 'std', 'min', 'max')]
impute_cols = set(sample_m.select_dtypes(include='number').columns)
impute_cols.remove('Alarm')
impute_cols.remove('Mode')
imputer=SimpleImputer(missing_values=np.nan,strategy= "mean")

for col in impute_cols:
    imputed = imputer.fit_transform(np.array(sample_m[col]).reshape(-1, 1))
    sample_m[col] = imputed
    
#msno.matrix(sample_m, labels=True, fontsize=10)

sample_m['Alarm'] = atLeast80Data['Alarm']
sample_m['Mode'] = atLeast80Data['Mode']
sample_m.to_csv("imputed-" + 'Eye_Car_Data' + '.csv')

    

# STEP3================= Preparing columns for Calculating Reaction Time ======================
# =============================================================================================
# 
# Creating needed columns for analyzing reaction time
sample_m = pd.read_csv("imputed-Eye_Car_Data.csv") 
dataFrame_Eye_Car= sample_m
dataFrame_Eye_Car['ReactionTime'] = 0
dataFrame_Eye_Car['Takeover'] = 'NT'
dataFrame_Eye_Car['AlarmDuration'] = 0

# Check if the timestamp of scenarios get reset for each and they're not continuous
dataFrame_Eye_Car.groupby('Name')['Timestamp'].agg(['min', 'max'])



# Remove the baseline from the data - starting with b/B
mask = dataFrame_Eye_Car['Name'].str.contains('b')
dataFrame_noBasic1 = dataFrame_Eye_Car[~mask]
dataFrame_noBasic1 = dataFrame_noBasic1.reset_index(drop=True)
mask2 = dataFrame_noBasic1['Name'].str.contains('B')
dataFrame_noBasic2 = dataFrame_noBasic1[~mask2]
dataFrame_noBasic2 = dataFrame_noBasic2.reset_index(drop=True)

dataFrame = dataFrame_noBasic2

#check the number of baseline rows 
dataFrame_Eye_Car.shape[0] - dataFrame_noBasic2.shape[0]

#check if  any null value exist in Alarm column
print("before deleting null value of Alarm column", dataFrame.shape)
print("number of rows with null Alarm value", dataFrame.Alarm.isna().sum())
# dataFrame = dataFrame[~dataFrame.Alarm.isna()].reset_index(drop=True)
# print("after deleting null value of Alarm column", dataFrame.shape)



# STEP4====================== Preprocessing the Alarm column ============================
#========================================================================================
# Alarm column is supposed to have values in a way that gets value when its "ON", other no value when it's off
# However, some of these values are repettitve due to sensor failure for detection. Here we fix them

#find change in a sequence 
#when the Alarm goes on and off
Alarm_tmp = dataFrame[['Name','Alarm']]
Alarm_chng1 = [(Alarm_tmp['Name'][i],Alarm_tmp['Alarm'][i],i) for i in range(1,len(Alarm_tmp)) if Alarm_tmp['Alarm'][i-1] != Alarm_tmp['Alarm'][i]] 

# creating a dataframe for alarm changes - the index column shows the row number of that entry
AlarmIndex = pd.DataFrame(Alarm_chng1, columns = ['Name', 'Alarm', 'index'])

# adding a status column which shows an alarm is on or off - initial value = 's'
AlarmIndex['status'] = 's'


# for each alarm the first time that goes off for each person means the alarm is on
for alarm in AlarmIndex.Alarm.unique():
    if alarm != 0:
        alarm_on = AlarmIndex[AlarmIndex.Alarm == alarm].groupby('Name')['index'].agg('min').tolist()
        #assign on value for those index
        AlarmIndex.loc[AlarmIndex['index'].isin(alarm_on),'status'] = 'on'


print(AlarmIndex)

# AlarmIndex[['Alarm']][0::2]; every other value of this list should change (alarm on - alarm off). 
# Due to some issue that we had some alarm goes off repeatedly
alarmRep = AlarmIndex[['Alarm']][0::2].reset_index(drop = True)['Alarm'].tolist()
# Here we're checking which alarm when goes off more than one times for each person - get the index of this repetition
alarmRepIndex = [(i*2) for i in range(1,len(alarmRep)) if alarmRep[i-1] == alarmRep[i]] 

# repeated on-off alarm is fixed here 
for idx in range(len(alarmRepIndex)):
    AlarmIndex.loc[alarmRepIndex[idx]-2:alarmRepIndex[idx], 'Alarm'] = AlarmIndex.loc[alarmRepIndex[idx]-2:alarmRepIndex[idx]]['Alarm'][alarmRepIndex[idx]-2]

print(AlarmIndex)

dataFrame_bc = dataFrame.copy()
#assign the value of alarm when the alarm is on after fixing on-off repetition
for alarm in AlarmIndex.Alarm.unique():
    if alarm != 0:
        alarm_tmp = pd.DataFrame(AlarmIndex[AlarmIndex.Alarm == alarm].groupby('Name')['index'].agg(['min', 'max'])).reset_index(drop = False)
        for name in alarm_tmp.Name.unique():
            minIndx = alarm_tmp[alarm_tmp.Name == name]['min'].values[0]
            maxIndx = alarm_tmp[alarm_tmp.Name == name]['max'].values[0]
            if minIndx != maxIndx:
                dataFrame.loc[minIndx:maxIndx,'Alarm'] = alarm
        
        
#find change in a sequence 
#when the Alarm goes on and off
Alarm_tmp = dataFrame[['Name','Alarm']]
Alarm_chng = [(Alarm_tmp['Name'][i],Alarm_tmp['Alarm'][i],i) for i in range(1,len(Alarm_tmp)) if Alarm_tmp['Alarm'][i-1] != Alarm_tmp['Alarm'][i]] 


AlarmIndex = pd.DataFrame(Alarm_chng, columns = ['Name', 'Alarm', 'index'])
AlarmIndex

# reseting the status of alarm
AlarmIndex['status'] = 's'
for alarm in AlarmIndex.Alarm.unique():
    if alarm != 0:
        alarm_on = AlarmIndex[AlarmIndex.Alarm == alarm].groupby('Name')['index'].agg('min').tolist()
        #assing on value for those index
        AlarmIndex.loc[AlarmIndex['index'].isin(alarm_on),'status'] = 'on'
        
#when the alarm is off the value of alarm is 0        
AlarmIndex.loc[AlarmIndex.Alarm == 0, 'status'] = 'off'

print(AlarmIndex)


# alarm is on (even index) and alarm off (odd index)
Alarm_on = AlarmIndex['index'][0::2].reset_index(drop = True)
Alarm_off = AlarmIndex['index'][1::2].reset_index(drop = True)
# # Evaluating whether the "ALARM" is changed
dataFrame.loc[AlarmIndex['index'][0],'Alarm'] #on
dataFrame.loc[AlarmIndex['index'][1],'Alarm'] #0ff




# STEP5============================ creating Mode change ======================
#========================================================================================

#reading alarm Type data
#alamTaskData = pd.read_csv("Alarm Type.csv")



# when the "MODE" have been changed - +1 Auto/ -1 Manual
Mode_tmp = dataFrame[['Name', 'Mode']]
Mode_chng = [(Mode_tmp['Name'][i],Mode_tmp['Mode'][i],i) for i in range(1,len(Mode_tmp)) if Mode_tmp['Mode'][i-1] != Mode_tmp['Mode'][i]] 

ModeIndex = pd.DataFrame(Mode_chng, columns = ['Name', 'Mode', 'index'])
ModeIndex

# the take over identificatio is when Mode Change from +1 (Auto) to -1 (Manual)
Man_switch = ModeIndex['index'][0::2].reset_index(drop = True)
# the Autonomous identificatio is when Mode Change from -1 (Manual) to +1 (Auto)
Car_switch = ModeIndex['index'][1::2].reset_index(drop = True)

#  Evaluating whether the "MODE" is changed
dataFrame.loc[ModeIndex['index'][0],'Mode']
dataFrame.loc[ModeIndex['index'][1],'Mode']








# Step6 ================= Calculating Reaction time and Creating Label for TakeOver ========================
#==========================================================================================================

# Goal in here is adding the exact index of mode changing from Man_switch variable to the Alarm_index columns
# To understand where the takeover happened

# First we initialize the indicatorTakeover with a value -1
#indicator whether a user took over
AlarmIndex['indicatorTakeover'] = -1 #when takeover happened the value would be updated to +1
# to insert index of takeover
#initialize the manSwitch_index with 00000
AlarmIndex['manSwitch_index'] = 00000 # when takeover happened the value would be updated to the index of the relevant Man_switch


import time
start_time = time.time()

# Fill the culumn of indicatorTakeover and manSwithc_index based on AlarmIndex data
# Takeover is happening when man_switch is btw alarm_on and alarm_off
for j in range(len(Alarm_on)):
    for i in range(len(Man_switch)):
        if Man_switch[i] in range(Alarm_on[j], Alarm_off[j]):
            AlarmIndex.loc[AlarmIndex['index'] == Alarm_on[j], 'indicatorTakeover'] = 1 #when the usr take over alarm_on entry is 1
            AlarmIndex.loc[AlarmIndex['index'] == Alarm_off[j], 'indicatorTakeover'] = 1 # bc on and off get the same takeover value (indicator and manSwitch_index)
            AlarmIndex.loc[AlarmIndex['index'] == Alarm_on[j], 'manSwitch_index'] = Man_switch[i] # get manSwitch column of AlarmIndex's row when the index column of AlarmIndex equals to the alarm_on[j]
            AlarmIndex.loc[AlarmIndex['index'] == Alarm_off[j], 'manSwitch_index'] = Man_switch[i]  # bc on and off get the same takeover value (indicator and manSwitch_index)
            break
print("time in sec:", time.time() - start_time)
# Now we know when the takeover happens with their index


# After knowing the takeover index, we need to add Alarmtype column
## adding type of alarm Type to this dataset
mapping = {'FA': [137,178,191,259,284,332], 
           'NA': [84, 213, 339],
           'TA': [16,5,103,254,323,185]}

AlarmIndex['AlarmType'] = 'Z' ## default value is Z(ero)
AlarmIndex.loc[AlarmIndex['Alarm'].isin(mapping['FA']), 'AlarmType'] = 'FA'    # False Alarm
AlarmIndex.loc[AlarmIndex['Alarm'].isin(mapping['NA']), 'AlarmType'] = 'NoA'    # No Alarm
AlarmIndex.loc[AlarmIndex['Alarm'].isin(mapping['TA']), 'AlarmType'] = 'TA'    # True Alarm


##------Alarm distribution for takeover and not-takeover
# Number of "NOT-TAKEOVER" per alarm type
AlarmIndex[AlarmIndex.indicatorTakeover == -1]['AlarmType'].value_counts()
# Number of "TAKEOVER" per alarm type
AlarmIndex[AlarmIndex.indicatorTakeover == 1]['AlarmType'].value_counts()
# Number of alarm per type, Z means alarm == 0 when the alarm is off (we have all the indexes of alarm_on and off which off values don't get changed)
AlarmIndex.AlarmType.value_counts()
# How many people tookover where there was "FALSE ALARM"
AlarmIndex[(AlarmIndex.AlarmType == 'FA') & (AlarmIndex.indicatorTakeover == 1)]['Name'].value_counts()



##-------- Add takeover label and Reaction time to the main data
dataFrame['Takeover'] = 'XX'
dataFrame['AlarmDuration'] = '00000'
dataFrame['ReactionTime'] = '00000'
dataFrame['Coming_Alarm'] = '00000'
dataFrame['Coming_AlarmType'] = '00000'
# if the take over happens during alarm on to alarm off 
# calculate reaction time (takeover time - alarm on time)
time_clm = dataFrame['Timestamp']
#name_clm = [a.split('_')[0] for a in dataFrame_noBasic['Name']]
name_clm = dataFrame['Name']
name = []
reactionTime = []
alarmDuration = []

## assigning TK when the takeover has been happened
time_threshold = 10000 # 10 seconds before alarm goes off


# Calculate the alarm gap, between each alarm to check how far the alarms are from each other
# optimizae thereshold
userLst = AlarmIndex.Name.unique()
time_off = 0
alarmOff = []
for usr in userLst:
    alarm = 0 # when alarm is off
    userDataSize = np.max(dataFrame.loc[(dataFrame.Name == usr)].index)
    alarm_off_idx_List = list(AlarmIndex.loc[(AlarmIndex.Name == usr) & (AlarmIndex.Alarm == alarm)].index)
    cnt = 0
    for idx in alarm_off_idx_List:
        off_start = AlarmIndex.loc[idx]['index']
        off_end = 0
        if idx + 1 == AlarmIndex.shape[0]:
           off_end = userDataSize 
        else:
            nxt_idx = AlarmIndex.loc[idx+1]['index']
            if nxt_idx > userDataSize:
                off_end = userDataSize 
            else:
                off_end = nxt_idx - 1
        time_off = time_clm[off_end] - time_clm.loc[off_start]
        tmp = dict({'counter': cnt, 'alarm_off_index': idx, 'user': usr, 'off_on_duration': time_off})
        cnt = cnt + 1
        alarmOff.append(tmp)
    
alarmOffDuration = pd.DataFrame(alarmOff)

alarmOffDuration.groupby('user')['off_on_duration'].agg(['min', 'max'])
        





## Labeling Takeover as 'TK'
all_takeover = AlarmIndex[AlarmIndex.indicatorTakeover == 1]
usrLst_takeover = list(AlarmIndex[AlarmIndex.indicatorTakeover == 1]['Name'].unique())

start_time = time.time()
for usr in usrLst_takeover:
    alarm_takeover = list(all_takeover[all_takeover.Name == usr]['Alarm'].unique())
    all_timestamp = dataFrame[dataFrame.Name == usr]['Timestamp'].values
    alarm_takeover.remove(0) #when the alarm is off 
    for alarm in alarm_takeover:
        print("User is: ", usr)
        print("alarm is: ", alarm)
        
        alarm_on_idx = all_takeover.loc[(all_takeover.Name == usr) & (all_takeover.Alarm == alarm)]['index'].values[0]
        nxt_idx = all_takeover.loc[(all_takeover.Name == usr) & (all_takeover.Alarm == alarm)].index[0] + 1
        alarm_off_idx = all_takeover.loc[nxt_idx]['index']
        takeover_idx = all_takeover.loc[(all_takeover.Name == usr) & (all_takeover.Alarm == alarm)]['manSwitch_index'].values[0]
        reaction_time = time_clm[takeover_idx] - time_clm[alarm_on_idx]
        alarm_duration = time_clm[alarm_off_idx] - time_clm[alarm_on_idx]
        
        # get the alarm Type 
        coming_alarm_type = all_takeover.loc[(all_takeover.Name == usr) & (all_takeover.Alarm == alarm)]['AlarmType'].values[0]
        
        #10 seconds before alarm goes off
        approximate_time_before_alarm = time_clm[alarm_on_idx-1] - time_threshold
        actual_time_before_alarm = min(all_timestamp, key=lambda a:abs(a-approximate_time_before_alarm))
        idx_actual_time_before_alarm = dataFrame[(dataFrame.Timestamp == actual_time_before_alarm) & (dataFrame.Name == usr)].index.values[0]
        
        #apply labeling for each unique user when the alarm is off
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1, 'Takeover'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = 'TK'
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1, 'ReactionTime'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)]= str(reaction_time)
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'AlarmDuration'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = str(alarm_duration)
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'Coming_Alarm'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = str(alarm)
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'Coming_AlarmType'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = str(coming_alarm_type)
        
        print(dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1, 'Takeover'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)].shape)
        
print("computation time:", time.time() - start_time) 
print("after takeover labeling", dataFrame.Takeover.value_counts())


## Labeling Not Takeover as 'NTK'
start_time = time.time()
all_not_takeover = AlarmIndex[AlarmIndex.indicatorTakeover == -1]
usrLst_not_takeover = list(AlarmIndex[AlarmIndex.indicatorTakeover == -1]['Name'].unique())

for usr in usrLst_not_takeover:
    alarm_not_takeover = list(all_not_takeover[all_not_takeover.Name == usr]['Alarm'].unique())
    all_timestamp = dataFrame[dataFrame.Name == usr]['Timestamp'].values
    alarm_not_takeover.remove(0) #when the alarm is off 
    for alarm in alarm_not_takeover:
#        print("User is: ", usr)
#        print("alarm is: ", alarm)
        alarm_on_idx = all_not_takeover.loc[(all_not_takeover.Name == usr) & (all_not_takeover.Alarm == alarm)]['index'].values[0]
        nxt_idx = all_not_takeover.loc[(all_not_takeover.Name == usr) & (all_not_takeover.Alarm == alarm)].index[0] + 1
        alarm_off_idx = all_not_takeover.loc[nxt_idx]['index']
        alarm_duration = time_clm[alarm_off_idx] - time_clm[alarm_on_idx]
        
        # get the alarm Type 
        coming_alarm_type = all_not_takeover.loc[(all_not_takeover.Name == usr) & (all_not_takeover.Alarm == alarm)]['AlarmType'].values[0]
        
        #10 seconds before alarm goes off
        approximate_time_before_alarm = time_clm[alarm_on_idx-1] - time_threshold
        actual_time_before_alarm = min(all_timestamp, key=lambda a:abs(a-approximate_time_before_alarm))
        idx_actual_time_before_alarm = dataFrame[(dataFrame.Timestamp == actual_time_before_alarm) & (dataFrame.Name == usr)].index.values[0]
        
        #apply labeling for each unique user when the alarm is off
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1, 'Takeover'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = 'NTK'
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'AlarmDuration'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = alarm_duration
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'Coming_Alarm'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = str(alarm)
        dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1,  'Coming_AlarmType'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)] = str(coming_alarm_type)
       
        print(dataFrame.loc[idx_actual_time_before_alarm: alarm_on_idx-1, 'Takeover'][(dataFrame.Name == usr) & (dataFrame.Alarm == 0)].shape)
         
print("computation time:", time.time() - start_time) 
print("after not_takeover labeling", dataFrame.Takeover.value_counts())




# ========================= Creating a SubDataset with Takeover and Not Takeover ==================================
### get the chunks of data during Alarm (hypothesis: participant takes over during the alarms are happening on to off)  
            
#extract the data chunks that have a takeover label
labeled_takeover_dataframe = dataFrame.loc[dataFrame.Takeover != 'XX'].reset_index()


# data with Takeover indicator(Takeover) and Take over time(takeoverTime)
labeled_takeover_dataframe.to_csv("takeover_Alarm_" + 'Eye_Car_Data_10sec' + '.csv')





# ------------------# some statistical checking -------------------

# frequency of take-over per user    
import collections
counter=collections.Counter(name)
print(counter)

# create a dataframe
reactionObj = {'Name': name, 'reactionTime': reactionTime, 'alarmDuration': alarmDuration}
reactionTime_user = pd.DataFrame(reactionObj)

 



# STEP4------------------# VISUALIZATION  -------------------
#------------------------------------------------------------

# visualize the mean Takeover time per user
groupbyUser = reactionTime_user.groupby('Name', as_index=False)
groupbyUser.mean().plot(kind = 'bar')

reactionTime_user.groupby('Name').mean().plot(y='reactionTime', kind='bar') # ReactionTime Plot
reactionTime_user.groupby('Name').mean().plot(kind='bar')   #ReactionTime & Alarm for all the participants


# checking one user
par_12 = dataFrame_noBasic2[dataFrame_noBasic2['Name'].str.contains('012')]
par_12.groupby('Alarm').mean().plot(y = 'ReactionTime', kind = 'bar', title = 'Take over time (p12)')
par_12 = dataFrame_noBasic2[dataFrame_noBasic2['AlarmDuration'].str.contains('012')]



# visualize the take over r.s.t. Alarm (would like to see the false-alarm)
plt.plot(dataFrame_noBasic2.Alarm)
plt.plot(dataFrame_noBasic2.Takeover)
plt.show()



# Take over data
dataFrame_takeover = dataFrame_noBasic2
#dataFrame_takeover = pd.read_csv('takeover-Eye_Car_Data.csv')

#Average Take-over time of all participants for each incident 
dataFrame_takeover.groupby('Alarm').mean().plot(y='TakeoverTime', kind='bar') #Average Take-over time of all participants for each incident 

par_12.groupby('Alarm').mean().plot(y='TakeoverTime', kind='bar')
alarm16_tmp = par_12[par_12['Alarm'] == 16]









