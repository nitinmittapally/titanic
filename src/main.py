# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:49:29 2021

@author: nitin
"""

import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score
# %inline matlplotlib
print("importing modules complete")

print("current working dir:" + os.getcwd())
train_data_path = "./data/train.csv"
train_data = pd.read_csv(train_data_path, index_col="PassengerId")

train_data.info();
train_data.describe()
train_data.isnull().sum(axis=0) # summing all downward 


# values to carry forward
valid_titles = None
modeEmbarked = None
medianAge = None 
scaler = None


def barGraph(feature, data):
    gData = data.groupby([feature, "Survived"]).size().unstack('Survived')
    gData.plot(kind='bar', stacked = True)


def processName(data, isTrain = True):
    print("Processing Name and generating titles")
    data["title"] = train_data['Name'].str.extract(r", (\w+)\.")
    global valid_titles
    if(isTrain):
        title_pct = data["title"].value_counts(normalize = True)
        valid_titles = title_pct[title_pct > 0.05]
    data["title_adjusted"] = data["title"].apply(lambda x: 
                                                 "MISC" if x not in valid_titles 
                                                 else x)
    
def processAge(data, isTrain = True):
    global medianAge
    if isTrain:
        medianAge = data.groupby(["title_adjusted"]).median()["Age"]
    data["Age"] = data.apply(lambda rec: medianAge[rec["title_adjusted"]] if np.isnan(rec["Age"]) else rec["Age"], axis = 1)

def processEmbarked(data, isTrain = True):
    global modeEmbarked
    if isTrain:
        modeEmbarked = data["Embarked"].mode()
    data["Embarked"].fillna(modeEmbarked[0], inplace = True)

def encodeCategories(data):
    print("Encoding categorical data")
    ohe_cols = ["Pclass", "Embarked", "title_adjusted"]
    dummies = pd.get_dummies(data[ohe_cols], columns=ohe_cols, prefix=ohe_cols)
    data[dummies.columns] = dummies
    label_encoder = LabelEncoder()
    data["Sex"] = label_encoder.fit_transform(data["Sex"])


def dropColumns(data):
    print("Dropping additional columns")
    columns_to_drop = ['Name', 'Ticket', 'Cabin', "Embarked", "title", "Pclass", "Sex", "title_adjusted"]
    data.drop(columns_to_drop, axis = 1, inplace = True)

def scaleData(data, isTrain = True):
    global scaler
    if(scaler is None or isTrain):
        scaler = StandardScaler()
        scaler.fit(data)
    scaler.transform(data, inplace = True)
    

def processData(data):
    print("Started processing data")
    processName(data)
    processAge(data)
    processEmbarked(data)
    encodeCategories(data)
    dropColumns(data)
    
def modelMetric(y_pred, y):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    print("AUC" + str(auc(fpr, tpr)))
    print("Accuracy" + str((y_pred == y_test).sum()/y_pred.shape[0]))


train = processData(train_data)

# understanding the data
cols = train_data.columns[1:]
for feature in cols:
    if train_data[feature].dtype != 'float64':
        print("-"*30)
        print(train_data[[feature, "Survived"]].groupby(feature).mean())

plt.figure(figsize= [20,16])

plt.boxplot(train_data["Fare"], showmeans=True, meanline=True)
x = [train_data[train_data['Survived']==1]['Fare'], train_data[train_data['Survived']==0]['Fare']]
plt.hist(x = x, stacked=True, color = ['g','r'],label = ['Survived','Dead'])

sns.barplot(x="Embarked_C", y = "Survived", data=train_data)



# trying all different algorithms 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import NuSVC

x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:, 1:], 
                                                    train_data.iloc[:,0],
                                                    test_size = 0.2, 
                                                    random_state = 4)


algs = [
     LogisticRegression(max_iter=500), 
     KNeighborsClassifier(n_neighbors=10), 
     DecisionTreeClassifier(),
     AdaBoostClassifier(),
     BaggingClassifier(),
     RandomForestClassifier(),
     GradientBoostingClassifier(),  
     NuSVC(),
        ]
score = dict()

for alg in algs: 
    score[alg.__class__.__name__] = cross_val_score(alg, x_train, y_train, cv=10, scoring=).mean()
 











