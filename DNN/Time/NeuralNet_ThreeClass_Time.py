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
import seaborn as sns
from sklearn.impute import SimpleImputer 
from sklearn import metrics
from scipy import interp
from itertools import cycle
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
#from rgf.sklearn import RGFClassifier
from sklearn.metrics import accuracy_score# same for tunning hyper parameter but will use random combinations of parameters


###=========================== Importing the DATASET ======================================
###=========================================================================================
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


   
## ========================= Eploring the data, mainly the Label (ReactionTime) ====================
## ===================================================================================================
#  let's check the "Takeover" distributions
#sns.countplot("TOT_Class",data=dataset)

# Let's check the Percentage for "ReactionTime"
Count_FastRT = len(dataset[dataset["TOT_Class"]== 0 ]) # Faster: <4000
Count_LowRT = len(dataset[dataset["TOT_Class"]== 1 ]) # Slower: >4000
Percentage_of_FastRT = Count_FastRT/(Count_FastRT+Count_LowRT)
print("Percentage_of_FastRT, 0 = ",Percentage_of_FastRT*100)
Percentage_of_SlowRT= Count_LowRT/(Count_FastRT+Count_LowRT)
print("Percentage_of_SlowRT, 1 = ",Percentage_of_SlowRT*100)


# Pandas offers us out-of-the-box three various correlation coefficients 1) Pearson's  2) Spearman rank 3) Kendall Tau
pearson = dataset.corr(method='pearson')
# assume target attr is the "Takeover or -3", then remove corr with itself
corr_with_target = pearson.iloc[-1][:]
# attributes sorted from the most predictive
predictivity = corr_with_target.sort_values(ascending=False)



## =========================-# Prepration for Machine Learning algorithms=========================
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
dataset['TOT_Three_Class'] = dataset.TOT_Three_Class.astype(object)

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



# Drop Reaction Time features which happen after Alarm
dataset = dataset.drop(['ReactionTime','Takeover','Coming_AlarmType','TOT_Class'], axis=1)




# ======================================= Count the number of Needed columns =========================

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


# How many columns will be needed for each categorical feature
print(dataset[Cat_Features].nunique(),
      'There are',"--",sum(dataset[Cat_Features].nunique().loc[:]),"--",'groups in the whole dataset')



# # ============================== Creating a Neural Network ===============================================

from pandas import read_csv
import tensorflow as tf
import GPUtil
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

# Check the GPU availability with various approaches
from tensorflow.python.client import device_lib
tf.test.gpu_device_name()  # if GPU is working
print(device_lib.list_local_devices()) # Which GPUs are connected
print("Number GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda()) #to make sure that your version of tensorflow includes GPU support.
GPUtil.showUtilization()
with tf.compat.v1.Session() as sess:
    devices = sess.list_devices()
    print(tf.config.list_physical_devices('GPU'))



# Assigning values to X, Y
y = dataset.TOT_Three_Class
X = dataset.drop('TOT_Three_Class', axis=1)


# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
FastRT = X[X.TOT_Three_Class==0]
MidRT  = X[X.TOT_Three_Class==1]
SlowRT = X[X.TOT_Three_Class==2]

# upsample minorityF
FastRT_upsampled = resample(FastRT,
                          replace=True, # sample with replacement
                          n_samples=len(MidRT), # match number in majority class
                          random_state=27) # reproducible results

SlowRT_upsampled = resample(SlowRT,
                          replace=True, # sample with replacement
                          n_samples=len(MidRT), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([MidRT, SlowRT_upsampled, FastRT_upsampled])

# check new class counts
upsampled.TOT_Three_Class.value_counts() #478219

# Trying logistic regression again with the balanced dataset
y_train = upsampled.TOT_Three_Class
X_train = upsampled.drop('TOT_Three_Class', axis=1)


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
#------------------------------------  





# prepare input of NN 
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# prepare output NN 
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)



import keras
# define the  model
model = Sequential()
model.add(Dense(23, input_dim=X_train_enc.shape[1], activation='relu', kernel_initializer='he_normal',  kernel_regularizer=keras.regularizers.l2(l=0.1)))
model.add(Dense(14, activation='relu',  kernel_regularizer=keras.regularizers.l2(l=0.1)))
model.add(Dense(8, activation='relu',  kernel_regularizer=keras.regularizers.l2(l=0.1)))
# logits layer
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Create a simple early stopping
# set early stopping monitor so the model stops training when it won't improve anymore
filepath="best-ThreeClasses-ReactionTime-{epoch:02d}-{val_loss:.2f}.hdf5"
keras_callbacks = [
      EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8),
      ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
]

# fit the keras model on the dataset
history_3 = model.fit(X_train_enc, y_train_enc, validation_split=0.10, epochs=10,
                    batch_size=16, verbose=2, callbacks=keras_callbacks) #val_split: Fraction of the training data to be used as validation data


# load the saved best model
saved_model = load_model('best-ThreeClasses-ReactionTime-Colab-18-0.15.hdf5')



#saved_model = model

# list all data in history# evaluate the model
_, train_acc = saved_model.evaluate(X_train_enc, y_train_enc, verbose=2)
_, test_acc = saved_model.evaluate(X_test_enc, y_test_enc, verbose=1)
print('Accuracy of test: %.2f' % (test_acc*100))
print('Accuracy of the: '+'1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc)) # test: 91.04

print(history_3.history.keys())


# # ============================== Plots ===============================================


#1)
# plot training history
plt.plot(history_3.history['loss'], label='train')
plt.plot(history_3.history['val_loss'], label='test')
plt.legend(['train', 'test'], loc='upper left')
plt.ylabel('Loss')
plt.show()


# summarize history for accuracy
plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#) ROC
#note in kera model.predict() will return predict probabilities
saved_model = model
pred_prob =  saved_model.predict(X_test_enc, verbose=0)
fpr, tpr, threshold = metrics.roc_curve(y_test_enc.ravel(), pred_prob.ravel())
roc_auc = metrics.auc(fpr, tpr)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_enc[:,i], pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
   
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_enc.ravel(), pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Result for Receiver operating characteristic to multi-class of Reaction Time')
plt.legend(loc="best")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


#3)

# Confusion Matrix

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
plot_confusion_matrix(cnf_matrix, classes=['low', 'mid', 'high'],
                      title='Non-normalized confusion matrix for the prediction of three classes')