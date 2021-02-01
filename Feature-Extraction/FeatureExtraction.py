# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:21:28 2020

@author: erfan pakdamanian
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import scipy.signal as ss
import wfdb
import csv
import sklearn
from quilt.data.ResidentMario import missingno_data
import missingno as msno 
import seaborn as sns
from sklearn.impute import SimpleImputer 

# Loading data from the iMotions the path to csv file directory
os.chdir("\\ML4TakeOver\\Data\\RawData")
directory = os.getcwd()
#---------------------------------------------------------
### Step 0 - Read the dataset, and do BASIC data exploration 
data = pd.read_csv('takeover4ML.csv', index_col=[0])

# The most basic thing is to get the data size: number of attributes (features) and instances (data points)
instance_count, attr_count = data.shape

# How values of each attributes are distributed
print(data.describe().T.loc[:, ('count', 'mean', 'std', 'min', 'max')])

# Is there any NaN value?
pd.isnull(data).any()
pd.isnull(data).sum()

#---------------------------------------------------------
# convert categorical value to the number
# convert datatype of object to int and strings 
data['Takeover'] = data.Takeover.astype('category')
data['AlarmType'] = data.AlarmType.astype('category')
data['TOT_Class'] = data.TOT_Class.astype('category')
data['Alarm'] = data.Alarm.astype('category')
data['Mode'] = data.Mode.astype('category')
data['EventW'] = data.EventW.astype('category')

cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

float_columns = data.select_dtypes(['float64']).columns
data[float_columns] = data[float_columns].apply(lambda x: x.astype('uint8'))

# some of the columns after the conversion have a zero
data = data.drop(['GazeDirectionLeftY', 'GazeDirectionLeftX', 'GazeDirectionRightX', 'GazeDirectionRightY','AlarmDuration'], axis = 1)




#---------------------------------------------------------
####----------------Step1. Correlations between attributes ------------

# Pandas offers us out-of-the-box three various correlation coefficients 1) Pearson's  2) Spearman rank 3) Kendall Tau
pearson = data.corr(method='pearson')
# assume target attr is the "Takeover or -3", then remove corr with itself
corr_with_target = pearson.iloc[-3][:]
# attributes sorted from the most predictive
predictivity = corr_with_target.sort_values(ascending=False)


# select some "STRONG" correlations between attribute "PAIRS"
attrs = pearson.iloc[:-3,:-3] # all except target
# only important correlations and not auto-correlations
threshold = 0.5

important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.iloc[abs(unique_important_corrs['correlation']).argsort()[::2]]

sns.pairplot(data)

## Step 1.1 Correlation Vizualition
data_tmp = data.drop(['ID', 'EventSource','GazeRightx', 'GazeRighty',
                                              'GazeLeftx', 'GazeLefty', 'AlarmType'], axis=1)
sns.heatmap(data_tmp) # compute and plot the pair-wise correlations
# save to file, remove the big white borders
# plt.savefig('attribute_correlations.png', tight_layout=True)


## Best Correlation Matrix Vis
plt.close("all") #Close all the opened pic tabs
plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
correlation = data.drop(['Name','ID','Timestamp'],axis=1).select_dtypes(include=['int64','uint8', 'int8']).iloc[:, :].corr()   #Temporarily removing unnecessary features
sns.heatmap(correlation, cmap=sns.diverging_palette(250, 10, n=3, as_cmap=True),ax=ax, vmax=1, square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.title('Correlation matrix')
plt.tight_layout()
plt.show()
k = data.shape[1]  # number of variables for heatmap
fig, ax = plt.subplots(figsize=(15, 15))
corrmat = data.drop(['Name','ID','Timestamp'],axis=1).corr() #Temporarily removing unnecessary features
# Generate a mask for the upper triangle
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cols = corrmat.nlargest(k, 'Takeover').index
cm = np.corrcoef(data.drop(['Name','ID','Timestamp'],axis=1)[cols].values.T)
sns.set(font_scale=1.0)
hm = sns.heatmap(cm, mask=mask, cbar=True, annot=True,
                  square=True, fmt='.2f', annot_kws={'size': 7},
                  yticklabels=cols.values,
                  xticklabels=cols.
                  values)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.title('TakeOver Annotated Correlation heatmap')
plt.tight_layout()
plt.show()


# Making histrogram of the label or target value 
take_tmp = np.array(data.TakeoverTime) # make an array to get zero excluded for the plot
sns.distplot(take_tmp[take_tmp>0], axlabel= "Time(ms)").set_title("TakeOver Time Distribution")



# relation of attribute pairs and joint distributions
sns.jointplot(data['Speed'], data['Takeover'], kind='scatter', joint_kws={'alpha':0.5})
sns.jointplot(data[data['TakeoverTime'] > 0]['Speed'], data[data['TakeoverTime'] > 0]['Takeover'], kind='hex')


data.to_csv("takeover_cleaned_feature" + '4ML' + '.csv')

