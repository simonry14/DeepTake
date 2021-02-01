# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:20:14 2020

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

# STEP2------------------# Importing the DATASET ------------
#------------------------------------------------------------
# Loading data from the iMotions the path to csv file directory
os.chdir("\\ML4TakeOver\\Data\\RawData")
directory = os.getcwd()


dataFrame_takeover_feature = pd.read_csv('QualityTakeover_4ML_NonZero.csv', index_col=[0])
dataset = dataFrame_takeover_feature



#================= Visualizing "TakeOver/Not-takeover" for each Alarm type ========================
#==========================================================================================================

# Remove 000 from data for visualization
#dataFrame_Alarm = dataFrame_takeover[np.logical_not(dataFrame_takeover.AlarmType.isin(['Z']))]

# we don't have zero alarm type anymore
dataFrame_Alarm = dataset.copy()

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
#tmp_dataframe.to_csv("UserComingAlarmType"+'.csv')

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

# Defining all classifiers

classifiers = []
classifiers.append(("logistic Regression", 
                    LogisticRegression()))
classifiers.append(("GradientBoostingClassifier",
                    GradientBoostingClassifier(n_estimators=20,learning_rate=0.01,subsample=0.6,random_state=127)))
classifiers.append(("RandomForestClassifier",
                    RandomForestClassifier(max_depth=2, n_estimators=10, max_features=1)))
#classifiers.append(("QuadraticDiscriminantAnalysis()", QuadraticDiscriminantAnalysis()))
classifiers.append(("Naive Bayes", GaussianNB()))
classifiers.append(("AdaBoostClassifier",
                    AdaBoostClassifier(n_estimators=100, random_state=0)))
#classifiers.append(("RGFClassifier",
#                    RGFClassifier(max_leaf=400,algorithm="RGF_Sib", test_interval=100,verbose=True)))





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
        mean_f1 = f1_score(y_true, y_pred, average='weighted')
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

result_table_down.to_csv("ML_Quality_3cl_upsamp_results.csv")


import pickle
#write 
with open('quality_3cl_ML_result_weighted.pkl', 'wb') as output:
    pickle.dump(result_table_down, output, pickle.HIGHEST_PROTOCOL)

#read
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




from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

# Making ROC curve broad view
plt.figure(figsize=(15,10))
plt.plot([0, 1], [0, 1], 'k--')
for i in range(result_table_down.shape[0]):
    plt.plot(result_table_down.loc[i,'falsepr_macro'], result_table_down.loc[i,'truepr_micro'], label= result_table_down.loc[i,'classifiers'])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Zoomed in ROC curve
plt.figure(figsize=(15,10))
itr = 0
tmp = result_table_down[result_table_down.counter == itr] # You can choose what iteration you want to show
plt.xlim(0, 0.6)
plt.ylim(0.3, 1)
plt.plot([0, 1], [0, 1], 'k--')
for i in range(tmp.shape[0]):
    plt.plot(tmp.loc[i,'falsepr'], tmp.loc[i,'truepr'], 
             label=tmp.loc[i,'classifiers']+"--"+'(area = %0.3f)' % tmp.loc[i,'roc_auc'] +'--'+'iteration:'+str(itr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()


#----------------------------------------------
# Investigate High correlation accuracy
import seaborn as sns    
mat.figure(figsize= (20, 10))
sns.heatmap({name_of_dataFrame}.corr(),annot= True)

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


