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
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

# STEP2------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
os.chdir("\\ML4TakeOver\\Data\\RawData")
directory = os.getcwd()
#dataFrame_takeover_feature = pd.read_csv('takeover_cleaned_feature4ML.csv', index_col=[0])
dataFrame_takeover_feature = pd.read_csv('takeover4ML.csv', index_col=[0])

dataset = dataFrame_takeover_feature

chunk_users = ['015_M3', '015_m2', '015_M1', '014_M3', #Select a handful of ppl for saving resource
               '014_M2', '014_m1']

chunk_dataset = dataset[dataset['Name'].isin(chunk_users)]


dataset = chunk_dataset
dataset.shape


###### ======================================Encoding notes=======================================
# Alarm Type:  TA =2, NoA =1, FA = 0 , Z = 3
# TakeOver :   TK =1 , NTK= 0
# Alarm   :    339.0 =339.0, 103.0= 4, 332.0=14, 259.0=11, 16.0=2, 178.0=6, 284.0=12, 
#               213.0=9, 323.0=13, 185.0=7, 84.0=3, 137.0=5,  5.0=1, 191.0=8, 254.0=10
# Mode   :  +1 (Auto)= +1,  -1(Manual)= 0

##### ===========================================================================================
dt_tmp = dataset
dt_tmp['Takeover'] = dt_tmp.Takeover.astype('category')
# Number of "NOT-TAKEOVER" per alarm type
dataset[dataset.Takeover == 'NTK']['Coming_AlarmType'].value_counts()
# Number of "TAKEOVER" per alarm type
dataset[dataset.Takeover == 'TK']['Coming_AlarmType'].value_counts()


   
## STEP3========================= Eploring the data, mainly the Label (Takeover) ====================
## ===================================================================================================
#  let's check the "Takeover" distributions
sns.countplot("Takeover",data=dataset)

# Let's check the Percentage for "TakeOver"
Count_NoTakeOver = len(dataset[dataset["Takeover"]== 0 ]) # Non-TakeOver are repersented by 0
Count_TakeOver = len(dataset[dataset["Takeover"]== 1 ]) # TakeOver by 1
Percentage_of_NoTakeOver = Count_NoTakeOver/(Count_NoTakeOver+Count_TakeOver)
print("percentage of None-TakeOver, 0 = ",Percentage_of_NoTakeOver*100)
Percentage_of_TakeOver= Count_TakeOver/(Count_NoTakeOver+Count_TakeOver)
print("percentage of TakeOver, 1 = ",Percentage_of_TakeOver*100)

# the amount related to valid "TakeOver" and "None-Takeover"
Amount_TakeOver  = dataset[dataset["Takeover"]== 1]
Amount_NoTakeOver = dataset[dataset["Takeover"]== 0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Amount_TakeOver.plot.hist(title="TakeOver", legend =None)
plt.subplot(122)
Amount_NoTakeOver.plot.hist(title="No-Takeover",legend =None)


# Pandas offers us out-of-the-box three various correlation coefficients 1) Pearson's  2) Spearman rank 3) Kendall Tau
pearson = dataset.corr(method='pearson')
# assume target attr is the "Takeover or -3", then remove corr with itself
corr_with_target = pearson.iloc[-3][:]
# attributes sorted from the most predictive
predictivity = corr_with_target.sort_values(ascending=False)



## STEP4=========================-# Prepration for Machine Learning algorithms=========================
## ====================================================================================================

# Drop useless features for ML
dataset = dataset.drop(['Timestamp','index','ID', 'Name', 'EventSource', 'ManualGear','EventW','EventN','GazeDirectionLeftY','Alarm',
                        'GazeDirectionLeftX', 'GazeDirectionRightX', 'GazeDirectionRightY','CurrentBrake',
                        'PassBy','RangeN'], axis=1)  #ManualGear has only "one" value
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


#****** Drop features that happing after Alarm (anything after alarm interupt takeover prediction)****************
dataset = dataset.drop(['Mode','TOT_Class', 'AlarmDuration','Coming_Alarm','ReactionTime','Coming_AlarmType'], axis=1) # Coming Alarm maybe helpful for ReactionTime
# ------------------------------------------------------.

# takeover (NT, TK) is our target 
input_data = dataset.iloc[:, dataset.columns != 'Takeover']
X = input_data
y = dataset[['Takeover']].values.ravel()


# ======================================= Encoding Categorical variables =========================

# # Encoding categorical variables
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer #labelencoder class takes cat. var. and assign value to them

# List of all Categorical features
Cat_Features= ['LeftLaneType','RightLaneType','NDTask']

# Get the column index of the categorical features
categorical_features = []
for i in Cat_Features:
    position = dataset.columns.get_loc(i)
    categorical_features.append(position)
print(categorical_features) 


# Get the column index of the Contin. features
conti_features = []
Cont_Filter = dataset.dtypes!=object
Cont_Filter = dataset.columns.where(Cont_Filter).tolist()
Cont_Filter_Cleaned = [name for name in Cont_Filter if str(name) !='nan']
for i in Cont_Filter_Cleaned:
    position = dataset.columns.get_loc(i)
    conti_features.append(position)
print(conti_features) 


# How many columns will be needed for each categorical feature?
print(dataset[Cat_Features].nunique(),
      'There are',"--",sum(dataset[Cat_Features].nunique().loc[:]),"--",'groups in the whole dataset')



# ===============================Create pipeline for data transformatin (normalize numeric, and hot encoder categorical)
# =============================================================================
from sklearn.pipeline import make_pipeline

numeric = make_pipeline(
 StandardScaler())

categorical = make_pipeline(
 # handles categorical features
 # sparse = False output an array not sparse matrix
 OneHotEncoder(sparse=False)) # Automatically take care of Dummy Trap

# creates a simple preprocessing pipeline (that will be combined in a full prediction pipeline below) 
# to scale the numerical features and one-hot encode the categorical features.

preprocess = make_column_transformer((numeric, Cont_Filter_Cleaned),
                                      (categorical, ['LeftLaneType','RightLaneType','Coming_AlarmType','NDTask']), 
                                       remainder='passthrough')

# =============================================================================
# Taking care of splitting
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42) 

# apply preprocess step (normalize the numeric value and one hot encoding for the categorical)
preprocess.fit_transform(X_train)
	

# =============================================================================
#SVM is usually optimized using two parameters gamma,C .
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] # C: the Cost parameter, Gamma: Control Bias and variance
# A High value of Gamma leads to more accuracy but biased results and vice-versa. 
# Similarly, a large value of Cost parameter (C) indicates poor accuracy but low bias and vice-versa.

tuned_parameters2 = [{'kernel': ['linear'], 'C': [1, 100]}]


model = make_pipeline(
    preprocess,
    SVC())


##### Try Simple Version ##############
from sklearn import svm
clf = svm.SVC()
X_train = preprocess.fit_transform(X_train)

grid_result = clf.fit(X_train, y_train)

X_test = preprocess.fit_transform(X_test)
clf.predict(X_test)

## we should try this in near future: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


##############
############################
##########################################
########################################################
######################################################################

# the GridSearchCV object with pipeline and the parameter space with 5 folds cross validation.
scores = ['precision', 'recall']
best_params = []
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters2, scoring='%s_macro' % score
    )
    X_train = preprocess.fit_transform(X_train)
    grid_result = clf.fit(X_train, y_train)
    best_params.append(grid_result.best_params_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    X_test = preprocess.fit_transform(X_test)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
# =============================================================================


#  ================= Resampling the imbalanced Label of "TakeOver" ========================================
#==========================================================================================================

# We create the preprocessing pipelines for both numeric and categorical data.
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


numeric_features = Cont_Filter_Cleaned
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['LeftLaneType','RightLaneType','Coming_AlarmType','NDTask']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    
# Append classifier to preprocessing pipeline.
# Separate input features and target
y = dataset.Takeover
X = dataset.drop('Takeover', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
take_over = X[X.Takeover=='TK']
not_takeover = X[X.Takeover=='NTK']

# upsample minority
not_takeover_upsampled = resample(not_takeover,
                          replace=True, # sample with replacement
                          n_samples=len(take_over), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([take_over, not_takeover_upsampled])

# check new class counts
upsampled.Takeover.value_counts() #713585


# trying logistic regression again with the balanced dataset
y_train = upsampled.Takeover
X_train = upsampled.drop('Takeover', axis=1)



##### LOGISTIC REGRESSION ###############################   
#########################################################   
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

y_score = clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test)) # model score: 0.846

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))


##### DECISION TREE    ##################################   
######################################################### 
from sklearn.tree import DecisionTreeClassifier
clf_3 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('reduce_dim', PCA()),
                        ('clf', DecisionTreeClassifier(random_state=0))])
    
y_score = clf_3.fit(X_train, y_train)
print("model score: %.3f" % clf_3.score(X_test, y_test)) # model score: 0.99

y_true_3, y_pred_3 = y_test, clf_3.predict(X_test)
print(classification_report(y_true_3, y_pred_3))


##### RANDOM FOREST    ##################################   
#########################################################   
clf_2 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('reduce_dim', PCA()),
                        ('clf',RandomForestClassifier(max_depth=2, random_state=0))]) 
    
y_score = clf_2.fit(X_train, y_train)
print("model score: %.3f" % clf_2.score(X_test, y_test)) # model score: 0.830

y_true_2, y_pred_2 = y_test, clf_2.predict(X_test)
print(classification_report(y_true_2, y_pred_2))



#####  Regularized Greedy Forest (RGF)     ##################################   
############################################################################  
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from rgf.sklearn import RGFClassifier

y_upsampled = upsampled.Takeover
X_upsampled = upsampled.drop('Takeover', axis=1)

clf_5 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('reduce_dim', PCA()),
                        ('classifier', RGFClassifier(max_leaf=400,
                                                   algorithm="RGF_Sib",
                                                   test_interval=100,
                                                   verbose=True))])

n_folds = 5

rgf_scores = cross_val_score(clf_5,
                             X_upsampled,
                             y_upsampled,
                             cv=StratifiedKFold(n_folds))

rgf_score = sum(rgf_scores)/n_folds
print('RGF Classifier score: {0:.5f}'.format(rgf_score)) #RGF Classifier score: 0.92304


XGBClassifier(class_weight='balanced')

#####  Gradiaent Boosting      #############################################  
############################################################################  
from sklearn.ensemble import GradientBoostingClassifier

clf_gb = Pipeline(steps=[('preprocessor', preprocessor),
                         ('reduce_dim', PCA()),
                         ('classifier', GradientBoostingClassifier(n_estimators=20,
                                                                   learning_rate=0.01,
                                                                   subsample=0.6,
                                                                   random_state=127))])
gb_scores = cross_val_score(clf_gb,
                            X_upsampled,
                            y_upsampled,
                            scoring="f1_weighted",
                            cv=StratifiedKFold(n_folds))

gb_score = sum(gb_scores)/n_folds
print('Gradient Boosting Classifier score: {0:.5f}'.format(gb_score)) #score: 0.79832
print('>> Mean CV score is: ', round(np.mean(gb_scores),3))
pltt = sns.distplot(pd.Series(gb_scores,name='CV scores distribution(Gradiaent Boosting)'), color='r')



  
##### ADA Boost   #########################################################
########################################################################### 
from sklearn.ensemble import AdaBoostClassifier
clf_4 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('reduce_dim', PCA()),
                      ('classifier', AdaBoostClassifier(n_estimators=100, random_state=0))])

y_score = clf_4.fit(X_train, y_train)
print("model score: %.3f" % clf_4.score(X_test, y_test)) # model score: 0.887

y_true_4, y_pred_4 = y_test, clf_4.predict(X_test)
print(classification_report(y_true_4, y_pred_4))


##### GAUSSIAN PROCESS  #################################   
######################################################### 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(1.0)
clf_3 = Pipeline(steps=[('preprocessor', preprocessor),
                        ('reduce_dim', PCA()),
                        ('clf',GaussianProcessClassifier(kernel=kernel, random_state=0))]) # model score: 0.830
    
y_score = clf_3.fit(X_train, y_train)
print("model score: %.3f" % clf_3.score(X_test, y_test)) # model score: 0.830

y_true_3, y_pred_3 = y_test, clf_3.predict(X_test)
print(classification_report(y_true_3, y_pred_3))




# # =============================================================================



#  ================= DownSampling the majority  imbalanced Label of "TakeOver" ======================================
#==================================================================================================================

# separate minority and majority classes
take_over = X[X.Takeover=='TK']
not_takeover = X[X.Takeover=='NTK']

# downsample majority
takeover_downsampled = resample(take_over,
                                replace = False, # sample without replacement
                                n_samples = len(not_takeover), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([takeover_downsampled, not_takeover])
# checking counts
downsampled.Takeover.value_counts()

# trying logistic regression again with the balanced dataset
y_train_down = downsampled.Takeover
X_train_down = downsampled.drop('Takeover', axis=1)


##### LOGISTIC REGRESSION ###############################   
#########################################################   
# Now we have a full prediction pipeline.
clf_down = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

y_score_down = clf_down.fit(X_train_down, y_train_down)
print("model score: %.3f" % clf_down.score(X_test, y_test)) # model score: 0.846

y_true, y_pred = y_test, clf_down.predict(X_test)
print(classification_report(y_true, y_pred))




##### ADA Boost   ##################################   
######################################################### 
from sklearn.ensemble import AdaBoostClassifier
clf_4_down = Pipeline(steps=[('preprocessor', preprocessor),
                      ('reduce_dim', PCA()),
                      ('classifier', AdaBoostClassifier(n_estimators=100, random_state=0))])

y_score = clf_4_down.fit(X_train_down, y_train_down)
print("model score: %.3f" % clf_4_down.score(X_test, y_test)) # model score: 0.887

y_true_down_4, y_pred_down_4 = y_test, clf_4_down.predict(X_test)
print(classification_report(y_true_down_4, y_pred_down_4))















# # =============================================================================
# example of one hot encoding for a neural network
from pandas import read_csv
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

# Check the GPU availability
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# Assigning values to X, Y
y = dataset.Takeover
X = dataset.drop('Takeover', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
take_over = X[X.Takeover=='TK']
not_takeover = X[X.Takeover=='NTK']

# upsample minority
not_takeover_upsampled = resample(not_takeover,
                          replace=True, # sample with replacement
                          n_samples=len(take_over), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([take_over, not_takeover_upsampled])

# check new class counts
upsampled.Takeover.value_counts() #713585

# trying logistic regression again with the balanced dataset
y_train = upsampled.Takeover
X_train = upsampled.drop('Takeover', axis=1)


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
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc

# load the dataset
X = dataset.drop('Takeover', axis=1)
y = dataset.Takeover
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
take_over = X[X.Takeover=='TK']
not_takeover = X[X.Takeover=='NTK']

# upsample minority
not_takeover_upsampled = resample(not_takeover,
                          replace=True, # sample with replacement
                          n_samples=len(take_over), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([take_over, not_takeover_upsampled])

# check new class counts
upsampled.Takeover.value_counts() #713585

# trying logistic regression again with the balanced dataset
y_train = upsampled.Takeover
X_train = upsampled.drop('Takeover', axis=1)

# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# define the  model
model = Sequential()
model.add(Dense(23, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(14, activation='relu'))
model.add(Dense(8, activation='relu'))
#logits layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# simple early stopping
#set early stopping monitor so the model stops training when it won't improve anymore
# checkpoint
filepath="best-model-{epoch:02d}-{val_loss:.2f}.hdf5"
keras_callbacks = [
      EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8),
      ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

# fit the keras model on the dataset
history = model.fit(X_train_enc, y_train_enc, validation_split=0.10, epochs=30,
                    batch_size=16, verbose=2, callbacks=keras_callbacks) #val_split: Fraction of the training data to be used as validation data

# load the saved best model
saved_model = load_model('best-model-02-0.04.hdf5')

# list all data in history
print(history.history.keys())

# evaluate the model
_, train_acc = saved_model.evaluate(X_train_enc, y_train_enc, verbose=2)
_, test_acc = saved_model.evaluate(X_test_enc, y_test_enc, verbose=1)
print('Accuracy of test: %.2f' % (test_acc*100))
print('Accuracy of the: '+'1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc)) # test: 91.04

# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(['train', 'test'], loc='upper left')
plt.ylabel('Loss')
plt.show()


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





