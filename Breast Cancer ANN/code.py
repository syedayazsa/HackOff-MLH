#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: ayaz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = pd.read_csv('data1.csv')
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
X = dataset.iloc[:,2:19].values
Y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0) 

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialize
classifier = Sequential()

#input/first layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 17))

#2nd layer
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))

#output
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting
classifier.fit(x_train, y_train, batch_size = 10, epochs = 25)



#Prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)

pickle.dump(classifier, open('code.pkl','wb'))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Loading model
mod = pickle.load(open('code.pkl', 'rb'))

y_new = mod.predict(var)

print(y_new)
