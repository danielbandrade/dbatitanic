#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 18:41:07 2018

@author: dbaprof
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import re
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

##Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
plt.close("all")

# Carregando os dados
train_data = pd.read_csv('train.csv', sep = ',')
train_data["Sex"]=train_data["Sex"].map({"male":1,"female":0,1:1,0:0}).values
test_data = pd.read_csv('test.csv', sep = ',')
test_data["Sex"]=test_data["Sex"].map({"male":1,"female":0,1:1,0:0}).values

sns.heatmap(train_data.corr(),annot=True);
plt.title('Correlation')
sns.factorplot(x="Pclass",y="Survived",hue="Sex",data=train_data,kind="bar")
plt.title('Class Sex Survived Dist')
plt.show()

# Training Data
train_data.reset_index()
my_train = train_data[["Sex","Survived","Pclass","Fare"]]
x_train = my_train[["Sex","Pclass","Fare"]].values
y_train = train_data[["Survived"]].values
y_train = y_train[:,0]

# Gaussian NB
nb = GaussianNB()
nb.fit(x_train,y_train)


# Test Data
test_data.reset_index()
my_test = test_data[["Sex","Pclass","Fare"]]
my_test.loc[pd.isnull(my_test['Fare'])==True,'Fare']=my_test.Fare.median()
x_test = my_test[["Sex","Pclass","Fare"]].values

# Prediction
y_test = nb.predict(x_test)

submit = pd.DataFrame(y_test)
psgID = test_data[["PassengerId"]]
submit = submit.merge(psgID,left_index=True, right_index=True)
submit = submit.set_index("PassengerId")
submit.to_csv('danielbandrade.csv', sep=',')


