# -*- coding: utf-8 -*-
"""
@author: erfan pakdamanian
"""

#!pip install rgf_python

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # # Encoding categorical variables
from sklearn.compose import ColumnTransformer, make_column_transformer #labelencoder class takes cat. var. and assign value to them
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier
from sklearn.metrics import accuracy_score
from scipy import interp
from sklearn import svm, datasets
import time
from sklearn.metrics import plot_roc_curve

os.getcwd()

from google.colab import drive
drive.mount('/content/gdrive')

#!ls '/content/gdrive/My Drive/Colab Notebooks'

dataFrame_takeover_feature = pd.read_csv('QualityTakeover_4ML_NonZero.csv', index_col=[0])
dataset = dataFrame_takeover_feature

dataset.columns

"""## 2 - Overview of the dataset ##"""

#================= Visualizing "TakeOver/Not-takeover" for each Alarm type ========================
#==========================================================================================================

# Remove 000 from data for visualization
#dataFrame_Alarm = dataFrame_takeover[np.logical_not(dataFrame_takeover.AlarmType.isin(['Z']))]

# we don't have zero alarm type anymore
dataFrame_quality = dataset.copy()

# check the number of user's per alarm
tmp_result = pd.DataFrame(dataFrame_quality.groupby(['Takeover_Quality']).agg({'Name': 'unique'}).reset_index()) 
[len(a) for a in tmp_result['Name']]

# tmp2 = pd.DataFrame(dataFrame_quality.groupby(['Name']).agg({'Takeover_Quality': 'unique'}).reset_index())
# [len(a) for a in tmp2['Takeover_Quality']]

# How many takeover and not-takeover per alarm?
dataFrame_quality.groupby(['Takeover_Quality','Name']).size().plot(kind = 'barh', legend = False)    # Frequency Based
plt.show()
# dataFrame_quality.groupby(['Takeover_Quality','Name']).agg({"Name": lambda x: x.nunique()}).plot(kind = 'barh', legend = False)

# Takeover frequency per individuals
tmp_dataframe = pd.DataFrame(dataFrame_quality.groupby(['Name', 'Takeover_Quality','TOT_Three_Class']).agg({"Takeover_Quality": lambda x: x.nunique()}))

dataFrame_quality.groupby(['Name', 'TOT_Three_Class']).agg({"Takeover_Quality": lambda x: x.nunique()})
dataFrame_quality.groupby(['Name', 'TOT_Three_Class','Takeover_Quality']).size().unstack().plot(kind = 'bar', stacked = True)
#dataFrame_AlarmIndividual = pd.DataFrame(dataFrame_quality.groupby(['Name', 'Coming_AlarmType','Takeover']).size().reset_index(name = 'frequency'))

## Counting the number of value changes in AlarmType
np.count_nonzero(np.diff(dataFrame_quality['Coming_Alarm']))

dataFrame_quality.groupby(['TOT_Three_Class', 'Takeover_Quality']).agg({"Name": lambda x: x.nunique()}).unstack().plot(kind = 'barh', stacked = False) 

# Quality: 1: <3.5 , 2: >7, 3:  3.5<  <7
# RT: 0= high(fast), 1=mid, 2=low(slow)



## Number of takeover quality per person?!
## Counting the number of value changes in AlarmType
np.count_nonzero(np.diff(dataFrame_quality['Coming_Alarm']))     #655



"""# **Traditional ML on Quaity of Takoever Prediction**

Column Prepration. Since we used the same chunk of data (10 sec before alarm triggering) we have to remove the columns that are basically related to future events or after alarm! We should not add them because it'sgoing to be cheating ;)
"""

# STEP3------------------# Prepration for Machine Learning algorithms ------------
#------------------------------------------------------------
# Drop useless features for ML
dataset = dataset.drop(['Timestamp','ID', 'Name', 'EventSource', 'ManualGear','EventW','EventN','GazeDirectionLeftY','Alarm',
                        'GazeDirectionLeftX', 'GazeDirectionRightX', 'GazeDirectionRightY','CurrentBrake',
                        'PassBy','RangeN','trajectory','trajectory_Alarm'], axis=1)  #ManualGear has only "one" value
                                                    #EventW is pretty similar to EventN
dataset.shape

#---------------------------------------------------------
# convert categorical value to the number 
# convert datatype of object to int and strings 
dataset['LeftLaneType'] = dataset.LeftLaneType.astype(object)
dataset['RightLaneType'] = dataset.RightLaneType.astype(object)
dataset['TOT_Class'] = dataset.TOT_Class.astype(object)
dataset['Coming_Alarm'] = dataset.Coming_Alarm.astype(object)
dataset['Takeover'] = dataset.Takeover.astype(object)
dataset['Coming_AlarmType'] = dataset.Coming_AlarmType.astype(object)
dataset['NDTask'] = dataset.NDTask.astype(object)
dataset['TOT_Three_Class'] = dataset.TOT_Three_Class.astype(object)
dataset['TOT_Five_Class'] = dataset.TOT_Five_Class.astype(object)
dataset['Takeover_Quality'] = dataset.Takeover_Quality.astype(object)

#****** Drop features that happing after Alarm (anything after alarm interupt takeover prediction)****************
dataset = dataset.drop(['Mode','TOT_Class', 'AlarmDuration','Coming_Alarm','Coming_AlarmType','ReactionTime','TOT_Five_Class'], axis=1) # Coming Alarm maybe helpful for ReactionTime

dataset = dataset.drop(['Unnamed: 0.1','Takeover','TOT_Three_Class'], axis=1) 

# STEP3------------------Create pipeline for data transformatin (normalize numeric, and hot encoder categorical) ------------
#------------------------------------------------------------


dataset = dataset.dropna()

dataset.dtypes

print(dataset.shape)
dataset.columns

# List of all Categorical features
Cat_Features= ['LeftLaneType','RightLaneType','NDTask']

# Get the column index of the Contin. features
conti_features = []
Cont_Filter = dataset.dtypes!=object
Cont_Filter = dataset.columns.where(Cont_Filter).tolist()
Cont_Filter_Cleaned = [name for name in Cont_Filter if str(name) !='nan']

# How many columns will be needed for each categorical feature?
print(dataset[Cat_Features].nunique(),
      'There are',"--",sum(dataset[Cat_Features].nunique().loc[:]),"--",'groups in the whole dataset')

numeric_features = Cont_Filter_Cleaned
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['LeftLaneType','RightLaneType','NDTask']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# prepare target
def prepare_targets(y_train, y_test):
	ohe = OneHotEncoder(sparse = False)
	y_train = y_train.values.reshape(-1,1)
	y_test = y_test.values.reshape(-1,1)
	ohe.fit(y_train)
	y_train_enc = ohe.transform(y_train)
	y_test_enc = ohe.transform(y_test)
	return y_test_enc

"""## Defining the Classifiers for the classification of takeover"""

# Defining all classifiers

classifiers = []
classifiers.append(("logistic Regression", 
                    LogisticRegression()))
classifiers.append(("QuadraticDiscriminantAnalysis()", QuadraticDiscriminantAnalysis(priors = [0.4 , 0.4, 0.2])))
classifiers.append(("GradientBoostingClassifier",
                    GradientBoostingClassifier(n_estimators=20,learning_rate=0.01,subsample=0.6,random_state=127)))
classifiers.append(("RandomForestClassifier",
                    RandomForestClassifier(max_depth=2, n_estimators=10, max_features=1)))
classifiers.append(("Naive Bayes", GaussianNB()))
classifiers.append(("AdaBoostClassifier",
                    AdaBoostClassifier(n_estimators=100, random_state=0)))
classifiers.append(("RGFClassifier",
                    RGFClassifier(max_leaf=400,algorithm="RGF_Sib", test_interval=100,verbose=True)))

dataset.Takeover_Quality.value_counts()

y = dataset.Takeover_Quality.astype('int')
X = dataset.drop('Takeover_Quality', axis=1)

# Append classifier to preprocessing pipeline.
n_folds = 3
#result_table = pd.DataFrame(columns=['counter', 'classifiers', 'falsepr','truepr','roc_auc', 'f1_micro','model_acc'])
result_table_down = pd.DataFrame(columns=['counter', 'classifiers', 'falsepr','truepr','roc_auc', 'f1_micro','model_acc'])

for fold in range(n_folds):
    print("--------------------------------------------------------------------")
    print("The fold number is {}".format(fold+1))
    print("--------------------------------------------------------------------")
    # splitting up testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_test = y_test
    # concatenate our training data back together
    upsample_input = pd.concat([X_train, y_train], axis=1)
    
    # separate minority and majority classes
    Worst_qual = upsample_input[upsample_input.Takeover_Quality==1]
    Bad_qual  = upsample_input[upsample_input.Takeover_Quality==2]
    Good_qual = upsample_input[upsample_input.Takeover_Quality==3]
    
    #-----------------------------------------------------
    # Downsample the majorities
    Worst_qual_upsampled = resample(Worst_qual,
                              replace=True, # sample with replacement
                              n_samples=len(Good_qual), # match number in majority class
                              random_state=27) # reproducible results

    Bad_qual_upsampled = resample(Bad_qual,
                              replace=True, # sample with replacement
                              n_samples=len(Good_qual), # match number in majority class
                              random_state=27) # reproducible results

    # combine majority and upsampled minority
    downsampled = pd.concat([Bad_qual_upsampled, Good_qual, Worst_qual_upsampled])

    # check new class counts
    print(downsampled.Takeover_Quality.value_counts()) #63079

    # trying logistic regression again with the balanced dataset
    # trying logistic regression again with the balanced dataset
    y_train = downsampled.Takeover_Quality
    X_train = downsampled.drop('Takeover_Quality', axis=1)
    

    for name, clf in classifiers:    
        classifier_start = time.time()

        clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('reduce_dim', PCA()),
                                       ('classifier', clf)])
    
        y_score = clf_pipeline.fit(X_train, y_train)    
        y_true, y_pred = y_test, clf_pipeline.predict(X_test)
        mean_f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        
        print("---------------------------------------------")
        print("---------------------------------------------")
        print("Classifier name is {}". format(name))
        print("The {} classifier,  mean f1 score is {}: ".format(name, mean_f1))
        print("The confusion matrix is: ")
        cfm = confusion_matrix(y_true, y_pred)
        print(cfm)
        print("---------------------------------------------")
        print(classification_report(y_true, y_pred))
        
        # for a ROC curve we need a probability
        # https://stats.stackexchange.com/questions/263121/why-are-the-roc-curves-not-smooth
        y_true, y_pred_prob = y_test, clf_pipeline.predict_proba(X_test)
       
        y_true_enc = prepare_targets(y_train, y_test)

        # Compute ROC curve and ROC area for each class
       # fpr_cl, tpr_cl, _ = roc_curve(y_true_enc.ravel(), y_pred_prob.ravel())
       # roc_auc = auc(fpr_cl, tpr_cl)

        # Compute ROC curve and ROC area for each class
        number_of_class = 3
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_i in range(number_of_class):
            fpr[class_i], tpr[class_i], _ = roc_curve(y_true_enc[:,class_i], y_pred_prob[:, class_i])
            roc_auc[class_i] = auc(fpr[class_i], tpr[class_i])
            
          
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_enc.ravel(), y_pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_class)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for class_i in range(number_of_class):
            mean_tpr += interp(all_fpr, fpr[class_i], tpr[class_i])

        # Finally average it and compute AUC
        mean_tpr /= number_of_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        result_table_down= result_table_down.append({'counter':fold,
                                           'classifiers':name,
                                           'falsepr_macro':fpr["macro"], 
                                           'truepr_macro':tpr["macro"],
                                           'falsepr_micro': fpr['micro'],
                                           'truepr_micro': fpr['micro'], 
                                           'roc_auc':roc_auc,
                                           'f1_micro': mean_f1,
                                           'model_acc': accuracy}, ignore_index=True)
        classifier_finish = time.time()
        print("--------------------------------------------------------------------")
        print("the classifie computation duration is: {}".format(classifier_finish - classifier_start))
        print("--------------------------------------------------------------------")
        # print(result_table[['roc_auc', 'falsepr_macro', 'truepr_macro']])

result_table_down.to_csv("QualityTakeover_downsamp_classifier_results.csv")


import pickle
with open('quality_result_table_down.pkl', 'wb') as output:
    pickle.dump(result_table_down, output, pickle.HIGHEST_PROTOCOL)

with open('quality_result_table_down.pkl', 'rb') as input:
    result_table_down = pickle.load(input)



## Making a Algorithm comparison boxplot
results = []
for classifier_name in result_table_down.classifiers.unique():
    accuracy_list = np.array(result_table_down[result_table_down.classifiers == classifier_name]['model_acc'])  
    results.append(accuracy_list)
    
names = result_table_down.classifiers.unique()
# boxplot result_table_down comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

"""## Save the result table to the GDriver"""

with open("/content/gdrive/My Drive/result.pkl", 'wb') as output:
  pickle.dump(result_table_down, output, pickle.HIGHEST_PROTOCOL)

"""## Reload the Result table from GDrive"""

import pickle
with open('quality_result_table_down_colab_1.pkl', 'rb') as input:
    result_table_down = pickle.load(input)



## Making a Algorithm comparison boxplot
results = []
for classifier_name in result_table_down.classifiers.unique():
    accuracy_list = np.array(result_table_down[result_table_down.classifiers == classifier_name]['model_acc'])  
    results.append(accuracy_list)
    
names = result_table_down.classifiers.unique()
# boxplot result_table_down comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

"""We added "upsample" at the begining of datasets (train and test) because we are going to preprocess them, as three classes of "Takaover_Quality" is massivly imbalanced as it shows above"""

y = dataset.Takeover_Quality.astype('int')
X = dataset.drop('Takeover_Quality', axis=1)

# Append classifier to preprocessing pipeline.
n_folds = 3
#result_table = pd.DataFrame(columns=['counter', 'classifiers', 'falsepr','truepr','roc_auc', 'f1_micro','model_acc'])
result_table_down = pd.DataFrame(columns=['counter', 'classifiers', 'falsepr','truepr','roc_auc', 'f1_micro','model_acc'])

for fold in range(n_folds):
    print("--------------------------------------------------------------------")
    print("The fold number is {}".format(fold+1))
    print("--------------------------------------------------------------------")
    # splitting up testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    y_test = y_test
    # concatenate our training data back together
    upsample_input = pd.concat([X_train, y_train], axis=1)
    
    # separate minority and majority classes
    Worst_qual = upsample_input[upsample_input.Takeover_Quality==1]
    Bad_qual  = upsample_input[upsample_input.Takeover_Quality==2]
    Good_qual = upsample_input[upsample_input.Takeover_Quality==3]
    
    #-----------------------------------------------------
    # Downsample the majorities
    Worst_qual_upsampled = resample(Worst_qual,
                              replace=True, # sample with replacement
                              n_samples=len(Good_qual), # match number in majority class
                              random_state=27) # reproducible results

    Bad_qual_upsampled = resample(Bad_qual,
                              replace=True, # sample with replacement
                              n_samples=len(Good_qual), # match number in majority class
                              random_state=27) # reproducible results

    # combine majority and upsampled minority
    downsampled = pd.concat([Bad_qual_upsampled, Good_qual, Worst_qual_upsampled])

    # check new class counts
    print(downsampled.Takeover_Quality.value_counts()) #63079

    # trying logistic regression again with the balanced dataset
    # trying logistic regression again with the balanced dataset
    y_train = downsampled.Takeover_Quality
    X_train = downsampled.drop('Takeover_Quality', axis=1)
    

    for name, clf in classifiers:    
        classifier_start = time.time()

        clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('reduce_dim', PCA()),
                                       ('classifier', clf)])
    
        y_score = clf_pipeline.fit(X_train, y_train)    
        y_true, y_pred = y_test, clf_pipeline.predict(X_test)
        mean_f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        
        print("---------------------------------------------")
        print("---------------------------------------------")
        print("Classifier name is {}". format(name))
        print("The {} classifier,  mean f1 score is {}: ".format(name, mean_f1))
        print("The confusion matrix is: ")
        cfm = confusion_matrix(y_true, y_pred)
        print(cfm)
        print("---------------------------------------------")
        print(classification_report(y_true, y_pred))
        
        # for a ROC curve we need a probability
        # https://stats.stackexchange.com/questions/263121/why-are-the-roc-curves-not-smooth
        y_true, y_pred_prob = y_test, clf_pipeline.predict_proba(X_test)
       
        y_true_enc = prepare_targets(y_train, y_test)

        # Compute ROC curve and ROC area for each class
       # fpr_cl, tpr_cl, _ = roc_curve(y_true_enc.ravel(), y_pred_prob.ravel())
       # roc_auc = auc(fpr_cl, tpr_cl)

        # Compute ROC curve and ROC area for each class
        number_of_class = 3
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for class_i in range(number_of_class):
            fpr[class_i], tpr[class_i], _ = roc_curve(y_true_enc[:,class_i], y_pred_prob[:, class_i])
            roc_auc[class_i] = auc(fpr[class_i], tpr[class_i])
            
          
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_enc.ravel(), y_pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_class)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for class_i in range(number_of_class):
            mean_tpr += interp(all_fpr, fpr[class_i], tpr[class_i])

        # Finally average it and compute AUC
        mean_tpr /= number_of_class

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        result_table_down= result_table_down.append({'counter':fold,
                                           'classifiers':name,
                                           'falsepr_macro':fpr["macro"], 
                                           'truepr_macro':tpr["macro"],
                                           'falsepr_micro': fpr['micro'],
                                           'truepr_micro': fpr['micro'], 
                                           'roc_auc':roc_auc,
                                           'f1_micro': mean_f1,
                                           'model_acc': accuracy}, ignore_index=True)
        classifier_finish = time.time()
        print("--------------------------------------------------------------------")
        print("the classifie computation duration is: {}".format(classifier_finish - classifier_start))
        print("--------------------------------------------------------------------")
        # print(result_table[['roc_auc', 'falsepr_macro', 'truepr_macro']])

result_table_down.to_csv("QualityTakeover_downsamp_classifier_results.csv")


import pickle
with open('quality_result_table_down.pkl', 'wb') as output:
    pickle.dump(result_table_down, output, pickle.HIGHEST_PROTOCOL)

with open('quality_result_table_down.pkl', 'rb') as input:
    result_table_down = pickle.load(input)



## Making a Algorithm comparison boxplot
results = []
for classifier_name in result_table_down.classifiers.unique():
    accuracy_list = np.array(result_table_down[result_table_down.classifiers == classifier_name]['model_acc'])  
    results.append(accuracy_list)
    
names = result_table_down.classifiers.unique()
# boxplot result_table_down comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

import pickle
# with open('quality_result_table_down.pkl', 'wb') as output:
#     pickle.dump(result_table_down, output, pickle.HIGHEST_PROTOCOL)

with open('quality_result_table_down_colab_1.pkl', 'rb') as input:
    result_table_down = pickle.load(input)



## Making a Algorithm comparison boxplot
results = []
for classifier_name in result_table_down.classifiers.unique():
    accuracy_list = np.array(result_table_down[result_table_down.classifiers == classifier_name]['model_acc'])  
    results.append(accuracy_list)
    
names = result_table_down.classifiers.unique()
# boxplot result_table_down comparison
fig = plt.figure(figsize=(22, 8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



"""#**Deep Learning on Quality of Takeover**"""

from pandas import read_csv
import tensorflow as tf
#import GPUtil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import h5py
import pytest

# Assigning values to X, Y
y = dataset.Takeover_Quality
X = dataset.drop('Takeover_Quality', axis=1)


# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
Worst_qual = X[X.Takeover_Quality==1]
Bad_qual  = X[X.Takeover_Quality==2]
Good_qual = X[X.Takeover_Quality==3]

# upsample minorityF
Worst_qual_upsampled = resample(Worst_qual,
                          replace=True, # sample with replacement
                          n_samples=len(Good_qual), # match number in majority class
                          random_state=27) # reproducible results

Bad_qual_upsampled = resample(Bad_qual,
                          replace=True, # sample with replacement
                          n_samples=len(Good_qual), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([Bad_qual_upsampled, Good_qual, Worst_qual_upsampled])

# check new class counts
upsampled.Takeover_Quality.value_counts() #478219

# Trying logistic regression again with the balanced dataset
y_train = upsampled.Takeover_Quality
X_train = upsampled.drop('Takeover_Quality', axis=1)

## Preprocessing
# Get the column index of the Contin. features
conti_features = []
Cont_Filter = dataset.dtypes!=object
Cont_Filter = dataset.columns.where(Cont_Filter).tolist()
Cont_Filter_Cleaned = [name for name in Cont_Filter if str(name) !='nan']
for i in Cont_Filter_Cleaned:
    position = dataset.columns.get_loc(i)
    conti_features.append(position)
print(conti_features) 

numeric_features = Cont_Filter_Cleaned
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['LeftLaneType','RightLaneType','NDTask']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])



# prepare input data
def prepare_inputs(X_train, X_test):
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.fit_transform(X_test)
    return X_train_enc, X_test_enc

# prepare target
def prepare_targets(y_train, y_test):
    ohe = OneHotEncoder(sparse = False)
    y_train = y_train.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    ohe.fit(y_train)
    y_train_enc = ohe.transform(y_train)
    y_test_enc = ohe.transform(y_test)
    return y_train_enc, y_test_enc

# Some global variables and general settings
saved_model_dir = './saved_model'
tensorboard_logs = './logs'
pd.options.display.float_format = '{:.2f}'.format
sns.set_context('notebook')
nnet_tools_path = os.path.abspath('NNet')

# prepare input of NN 
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output NN 
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# cleanup the old training logs and models
#!rm -rf $tensorboard_logs model-*.h5 $saved_model_dir

# training callbacks
# Create a simple early stopping
# set early stopping monitor so the model stops training when it won't improve anymore
mc_file = 'Qualitymodel-best3class-{epoch:02d}-{val_loss:.2f}.h5'
es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
mc_cb = ModelCheckpoint(mc_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


# define the  model
model = Sequential()
model.add(Dense(23, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(14, activation='relu'))
model.add(Dense(8, activation='relu'))
# logits layer
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Create a simple early stopping
# set early stopping monitor so the model stops training when it won't improve anymore

keras_callbacks = [
      EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8),
      ModelCheckpoint(mc_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

# fit the keras model on the dataset
history_3 = model.fit(X_train_enc, y_train_enc, validation_split=0.10, epochs=10,
                    batch_size=16, verbose=2, callbacks=keras_callbacks) #val_split: Fraction of the training data to be used as validation data

model.summary()

# Save the entire model as a SavedModel.
#!mkdir -p saved_model
model.save('"/content/gdrive/My Drive/Colab Notebooks')

#!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
def print_heading(string, color=None):
    print_html(string, tag='h3', color=color)

# 2. Save Keras Model or weights on google drive

# create on Colab directory
model.save('TakeoverQuality_model_NeuralNetwork.h5')    
model_file = drive.CreateFile({'title' : 'TakeoverQuality_model_NeuralNetwork.h5'})
model_file.SetContentFile('TakeoverQuality_model_NeuralNetwork.h5')
model_file.Upload()

# download to google drive
drive.CreateFile({'id': model_file.get('id')})

from google.colab import files
files.download("best-ThreeClasses-quality-06-0.00.hdf5")

import glob, math, os, sys, zipfile
from IPython.display import display, HTML
# Some global variables and general settings
saved_model_dir = './saved_model'
tensorboard_logs = './logs'
pd.options.display.float_format = '{:.2f}'.format
sns.set_context('notebook')
nnet_tools_path = os.path.abspath('NNet')

def print_html(string, tag='span', color=None, size=None):
    size = f'font-size:{size};' if size else ''
    color = f'color:{color};' if color else ''
    display(HTML(f'<{tag} style="{color}{size}">{string}</{tag}>'))

def print_heading(string, color=None):
    print_html(string, tag='h3', color=color)
    
def print_message(string, color=None):
    print_html(string, color=color)

# pick best model file from filesystem
best_model_path = sorted(glob.glob('Qualitymodel-best-*.h5'))[-1]
print_heading('Best Model:')
print_message(best_model_path)

# cleanup old model
#!rm -rf $saved_model_dir

# save model in tf and h5 formats
tf_model_path = f'{saved_model_dir}/model'
h5_model_path = f'{saved_model_dir}/model.h5'
model.save(tf_model_path, save_format='tf')
model.save(h5_model_path, save_format='h5')

from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

#import json
#with open('"/content/gdrive/My Drive/QualityModel.json', 'w', encoding='utf-8') as json_file:
#    json.dump(data, json_file, ensure_ascii=False, indent=4)

# load the saved best model
saved_model = load_model('Qualitymodel-best-05-0.00.h5')

# list all data in history# evaluate the model
_, train_acc = saved_model.evaluate(X_train_enc, y_train_enc, verbose=2)
_, test_acc = saved_model.evaluate(X_test_enc, y_test_enc, verbose=1)
print('Accuracy of test: %.2f' % (test_acc*100))
print('Accuracy of the: '+'1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc)) # test: 85.15

print(history_3.history.keys())

"""# New Section
TEST--------
"""

import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model
X,y = x_enc, y_enc
model =  load_model('Qualitymodel-best-05-0.00.h5')

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

metrics = pd.DataFrame(history_3.history)
metrics.to_csv("metrics.csv")

#!cp metrics.csv "saved_model/"

dl_metrics = pd.read_csv("saved_model/metrics.csv")
dl_metrics.head(10)

vis_metrics = pd.DataFrame(columns=["epoch", "accuracy_value", "accuracy_label"])

tmp = dl_metrics['accuracy']
tmp = tmp.append(dl_metrics['val_accuracy'])
vis_metrics['accuracy_value'] = tmp

#vis_metrics['index'] = np.arange(0,399)

vis_metrics.iloc[0:398]["accuracy_label"] = "accuracy"
vis_metrics.iloc[399:798]["accuracy_label"] = "val_accuracy"
vis_metrics['epoch'] = vis_metrics.index

sns.lineplot(x="epoch", y="accuracy_value", hue="accuracy_label", style="accuracy_label", markers=False, dashes=True, data = vis_metrics)

vis_metrics

#print_heading(f'Evaluating {best_model_path}')

# load the saved best model
#saved_model = load_model(tf_model_path)

# evaluate the model
_, train_acc = saved_model.evaluate(X_train_enc, y_train_enc, verbose=2)
_, test_acc = saved_model.evaluate(X_test_enc, y_test_enc, verbose=1)
print('Accuracy of test: %.2f' % (test_acc*100))
print('Accuracy of the: '+'1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

# plot training history
plt.plot(history_3.history['loss'], label='train')
plt.plot(history_3.history['val_loss'], label='test')
plt.legend(['train', 'test'], loc='upper left')
plt.ylabel('Loss')
plt.show()

# # summarize history for accuracy
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#note in kera model.predict() will return predict probabilities
pred_prob =  saved_model.predict(X_test_enc, verbose=0)
fpr, tpr, threshold = metrics.roc_curve(y_test_enc.ravel(), pred_prob.ravel())
roc_auc = metrics.auc(fpr, tpr)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_enc[:,i], pred_prob[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])    
   
# Compute micro-average ROC curve and ROC area
fpr['micro'], tpr['micro'], _ = metrics.roc_curve(y_test_enc.ravel(), pred_prob.ravel())
roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3

fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = metrics.auc(fpr['macro'], tpr['macro'])

plt.figure(1)
plt.plot(fpr['micro'], tpr['micro'],
         label='micro-average ROC curve (area = {0:0.2f})' \
               ''.format(roc_auc['micro']),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr['macro'], tpr['macro'],
         label='macro-average ROC curve (area = {0:0.2f})' \
               ''.format(roc_auc['macro']),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})' \
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Result for Receiver operating characteristic to multi-class of Reaction Time')
plt.legend(loc='lower right')
plt.show()

# Creating Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
cnf_matrix = metrics.confusion_matrix(y_test_enc.argmax(axis=1), pred_prob.argmax(axis=1))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['low', 'mid', 'high'])



#!pip install shap
#
#prediction = estimator.predict(X)
#accuracy_score(y, prediction)
#
#perm = PermutationImportance(my_model, random_state=1).fit(X,y)
#eli5.show_weights(perm, feature_names = X.columns.tolist())



# How many takeover and not-takeover per alarm?
dataFrame_quality.groupby(['Takeover_Quality','Name']).size().plot(kind = 'barh', legend = False)    # Frequency Based
plt.show()
# dataFrame_quality.groupby(['Takeover_Quality','Name']).agg({"Name": lambda x: x.nunique()}).plot(kind = 'barh', legend = False)

# Takeover frequency per individuals
tmp_dataframe = pd.DataFrame(dataFrame_quality.groupby(['Name', 'Takeover_Quality','TOT_Three_Class']).agg({"Takeover_Quality": lambda x: x.nunique()}))

dataFrame_quality.groupby(['Name', 'TOT_Three_Class']).agg({"Takeover_Quality": lambda x: x.nunique()})
dataFrame_quality.groupby(['Name', 'TOT_Three_Class','Takeover_Quality']).size().unstack().plot(kind = 'bar', stacked = True)
#dataFrame_AlarmIndividual = pd.DataFrame(dataFrame_quality.groupby(['Name', 'Coming_AlarmType','Takeover']).size().reset_index(name = 'frequency'))

dataFrame_quality.groupby(['TOT_Three_Class', 'Takeover_Quality']).agg({"Name": lambda x: x.nunique()}).unstack().plot(kind = 'barh', stacked = False) 

# Quality: 1: <3.5 , 2: >7, 3:  3.5<  <7
# RT: 0= high(fast), 1=mid, 2=low(slow)

dataFrame_takeover_feature.ReactionTime.mean()

1779.815501654964 # Sd Reaction time
5591.25068223461 #mean Reaction Time


"""No Alarm vs True Alarm: Comparing their quality"""

### testing the feature importance
y = dataset.Takeover_Quality.astype('int')
X = dataset.drop('Takeover_Quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
y_test = y_test
# prepare input of NN 
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output NN 
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
def baseline_model():
    model_fe = Sequential()
    model_fe.add(Dense(23, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
    model_fe.add(Dense(14, activation='relu'))
    model_fe.add(Dense(8, activation='relu'))
    # logits layer
    model_fe.add(Dense(3, activation='softmax'))
    model_fe.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_fe

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1, batch_size=1)

#estimator.fit(X, y)
#prediction1 = estimator.predict(X_test_enc)
#accuracy_score(Y_test_enc, prediction1)
keras_callbacks = [
      EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=8)
]
estimator.fit(X_train_enc, y_train_enc,  verbose=2, validation_split=0.10,callbacks=keras_callbacks)

#perm = PermutationImportance(estimator, random_state=1).fit(X_train_enc, y_train_enc)
#eli5.show_weights(perm, feature_names = X_train_enc.columns.tolist())

perm = PermutationImportance(estimator, random_state=1).fit(X_train_enc, y_train_enc)

#kheili tool mikeshe engar kolan eli5 baraye tedad balaye features ha konde!
#
from google.colab import drive
drive.mount('/content/gdrive')

import pickle
with open("/content/gdrive/My Drive/perm.pkl", 'wb') as output:
  pickle.dump(perm, output, pickle.HIGHEST_PROTOCOL)

with open("/content/gdrive/My Drive/estimator.pkl", 'wb') as output:
  pickle.dump(estimator, output, pickle.HIGHEST_PROTOCOL)

eli5.show_weights(perm, feature_names = X_train_enc.columns.tolist())