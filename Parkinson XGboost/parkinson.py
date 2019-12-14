#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: ayaz
"""
import pickle
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('parkinsons2.csv')


#X and Y
X = df.loc[:,df.columns!='status'].values[:,1:]
y = labels=df.loc[:,'status'].values

#Feature Scaling
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(X)

#Splitting
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)

#Model training
model=XGBClassifier()
model.fit(x_train,y_train)

#Prediction

y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)

pickle.dump(model, open('parkinson.pkl','wb'))

#loading model
mod = pickle.load(open('parkinson.pkl', 'rb'))

y_new = mod.predict(var)

print(y_new)