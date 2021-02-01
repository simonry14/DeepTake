# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:56:08 2020

@author: erfan pakdamanian
"""
# STEP1----------------- # Importing the libraries------------
#-------------------------------------------------------------
import os
import numpy as np
import matplotlib
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
from sklearn.metrics import accuracy_score# same for tunning hyper parameter but will use random combinations of parameters


# STEP2------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
os.chdir("C:\ML4TakeOver\\Data\\RawData")
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


   
## STEP3========================= Eploring the data, mainly the Label (Takeover) ====================
## ===================================================================================================
#  let's check the "Takeover" distributions
sns.countplot("TOT_Class",data=dataset)

# Let's check the Percentage for "ReactionTime"
Count_FastRT = len(dataset[dataset["TOT_Class"]== 0 ]) # Faster: <4000
Count_LowRT = len(dataset[dataset["TOT_Class"]== 1 ]) # Slower: >4000
Percentage_of_FastRT = Count_FastRT/(Count_FastRT+Count_LowRT)
print("Percentage_of_FastRT, 0 = ",Percentage_of_FastRT*100)
Percentage_of_SlowRT= Count_LowRT/(Count_FastRT+Count_LowRT)
print("Percentage_of_SlowRT, 1 = ",Percentage_of_SlowRT*100)

# the amount related to valid "TakeOver" and "None-Takeover"
Amount_SlowRT  = dataset[dataset["TOT_Class"]== 1] #Slower
Amount_FastRT = dataset[dataset["TOT_Class"]== 0] #Faster
plt.figure(figsize=(10,6))
plt.subplot(121)
Amount_SlowRT.plot.hist(title="SlowReaction", legend =None)
plt.subplot(122)
Amount_FastRT.plot.hist(title="FastReaction",legend =None)


# Pandas offers us out-of-the-box three various correlation coefficients 1) Pearson's  2) Spearman rank 3) Kendall Tau
pearson = dataset.corr(method='pearson')
# assume target attr is the "Takeover or -3", then remove corr with itself
corr_with_target = pearson.iloc[-1][:]
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
dataset = dataset.drop(['Mode','AlarmDuration','Coming_Alarm'], axis=1) # Coming Alarm maybe helpful for ReactionTime

# Check the reaction time values in each category of Alarm
print('FalseAlarm ReactionTime:', dataset[dataset['Coming_AlarmType']== 'FA'].ReactionTime.mean())  # 2007.2
print('TrueAlarm ReactionTime:', dataset[dataset['Coming_AlarmType']== 'TA'].ReactionTime.mean())   # 4712.5
print('NoAlarm ReactionTime:', dataset[dataset['Coming_AlarmType']== 'NoA'].ReactionTime.mean())    # 5003.5

# How many times they takeover in each alarm
len(dataset[dataset['Coming_AlarmType']== 'FA'][dataset['Takeover']=='TK'].ReactionTime.unique()) #92
len(dataset[dataset['Coming_AlarmType']== 'FA'][dataset['Takeover']=='NTK'].ReactionTime.unique())

len(dataset[dataset['Coming_AlarmType']== 'TA'][dataset['Takeover']=='TK'].ReactionTime.unique()) #355
len(dataset[dataset['Coming_AlarmType']== 'TA'][dataset['Takeover']=='NTK'].ReactionTime.unique())

len(dataset[dataset['Coming_AlarmType']== 'NoA'][dataset['Takeover']=='TK'].ReactionTime.unique()) #81
len(dataset[dataset['Coming_AlarmType']== 'NoA'][dataset['Takeover']=='NTK'].ReactionTime.unique())

dataFrame_takeover_feature[dataFrame_takeover_feature['Coming_AlarmType']== 'NoA'][
        dataFrame_takeover_feature['Takeover']=='NTK'].Name.value_counts()



# Drop Reaction Time feature
dataset = dataset.drop(['ReactionTime','Coming_AlarmType'], axis=1)

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

## we should try this: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/


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










# # =============================================================================
# example of one hot encoding for a neural network
from pandas import read_csv
import keras
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
y = dataset.TOT_Class
X = dataset.drop('TOT_Class', axis=1)


# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
FastRT = X[X.TOT_Class==0]
SlowRT = X[X.TOT_Class==1]

# upsample minority
not_takeover_upsampled = resample(FastRT,
                          replace=True, # sample with replacement
                          n_samples=len(SlowRT), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([SlowRT, not_takeover_upsampled])

# check new class counts
upsampled.TOT_Class.value_counts() #433756

# trying logistic regression again with the balanced dataset
y_train = upsampled.TOT_Class
X_train = upsampled.drop('TOT_Class', axis=1)


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
#------------------------------------


import tensorflow as tf
import random
tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)


# prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# define the  model
model = Sequential()
model.add(Dense(23, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model.add(Dense(14, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
model.add(Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
#logits layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# simple early stopping
#set early stopping monitor so the model stops training when it won't improve anymore
# checkpoint
filepath="best-model-TK-{epoch:02d}-{val_loss:.2f}.hdf5"
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

saved_model = model

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


#---------------------ROC curve ------------------------
base_fpr = np.linspace(0, 1, 101)
y_pred_keras = model.predict(X_test_enc).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_enc, y_pred_keras)
roc_auc = roc_auc_score(y_test_enc, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
tpr = np.interp(base_fpr, fpr_keras, tpr_keras)
tpr[0] = 0.0


# predict probabilities for test set
yhat_probs = model.predict(X_test_enc, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test_enc, verbose=0)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test_enc, yhat_classes)
print('Accuracy: %f' % accuracy)
# recall: tp / (tp + fn)
recall = recall_score(y_test_enc, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test_enc, yhat_classes)
print('F1 score: %f' % f1)

# ROC AUC
auc = roc_auc_score(y_test_enc, yhat_probs)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test_enc, yhat_classes)
print(matrix)






import pickle
with open('result_data.pkl', 'rb') as input:
    result_table = pickle.load(input)
    
print(result_table.shape)

tmp_result = result_table.append({'counter':0,'classifiers':'DNN',
                                           'falsepr':fpr_keras, 
                                           'truepr':tpr_keras, 
                                           'roc_auc':roc_auc,
                                           'auc': auc_keras,
                                           'tprs': tpr,
                                           'f1_micro': 0,
                                           'model_acc': test_acc}, ignore_index=True)

print(tmp_result.shape)

with open('result_data_NN.pkl', 'wb') as output:
    pickle.dump(tmp_result, output, pickle.HIGHEST_PROTOCOL)


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

classifiersName = ['logistic Regression', 'Gradient Boosting',
                   'Random Forest', 'Naive Bayes', 'AdaBoost',
                   'RGF', 'DNN']
n_folds = 5
i = 0
clrs = sns.color_palette("deep", len(tmp_result.classifiers.unique()))
plt.figure(figsize=(12, 8))
for cl in tmp_result.classifiers.unique():
    if cl == 'DNN':
        tprs = tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == 0), 'tprs'].values[0]
        aucs = np.array(tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == 0), 'auc'].values[0])
        mean_tprs = tprs.mean(axis=0)
#        std = 0.09975142667115305
#        tprs_upper = np.minimum(mean_tprs + std, 1)
#        tprs_lower = mean_tprs - std
        tprs = tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == 0), 'tprs'].values[0]
        aucs = np.array(tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == 0), 'auc'].values[0])
        plt.plot(base_fpr, tprs, color = clrs[i], alpha = 0.7, label=r' %s, AUC = %0.2f ' % ("DeepTake",aucs),)
#        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = clrs[i], alpha = 0.2)
        i = i + 1
    else:
        print(cl)
        tprs = tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == (n_folds - 1)), 'tprs'].values[0]
        aucs = np.array(tmp_result.loc[(tmp_result.classifiers == cl) & (tmp_result.counter == (n_folds - 1)), 'auc'].values[0])
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        print(std)
        mean_auc = auc(base_fpr, mean_tprs)
        std_auc = np.std(aucs)
        
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        plt.plot(base_fpr, mean_tprs, color = clrs[i], alpha = 0.7, label=r' %s, AUC = %0.2f ' % ( classifiersName[i],mean_auc),)
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = clrs[i], alpha = 0.2)
        i = i + 1
        
        
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
  

#---------------------------------------------

# define the  model
model_2 = Sequential()
model_2.add(Dense(12, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal'))
model_2.add(Dense(8, activation='relu'))
#logits layer
model_2.add(Dense(1, activation='sigmoid'))
# compile the keras model
model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_2.fit(X_train_enc, y_train_enc, validation_split=0.20, epochs=30,
                    batch_size=16, verbose=2, callbacks=[early_stopping_monitor])

# evaluate the model
_, train_acc_2 = model_2.evaluate(X_train_enc, y_train_enc, verbose=2)
_, test_acc_2 = model_2.evaluate(X_test_enc, y_test_enc, verbose=1)
print('Accuracy of test: %.2f' % (test_acc_2*100))
print('Accuracy of the: '+'1) Train: %.3f, 2) Test: %.3f' % (train_acc_2, test_acc_2)) #Test: 0.885
