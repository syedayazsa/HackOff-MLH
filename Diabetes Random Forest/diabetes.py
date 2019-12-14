# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import pickle
# Importing the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Pickling
pickle.dump(classifier, open('diabetes.pkl','wb'))

mod = pickle.load(open('diabetes.pkl', 'rb'))
y_new = mod.predict(var)

print(y_new)