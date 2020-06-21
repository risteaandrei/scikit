import ssl

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

ssl._create_default_https_context = ssl._create_unverified_context

iris = sns.load_dataset('iris')
#print(iris.info())

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

svc_model = SVC()
svc_model.fit(X_train, y_train)

#predictions = svc_model.predict(X_test)
#
#print(classification_report(y_test, predictions))
#print("##")
#print(confusion_matrix(y_test, predictions))

param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)

grid_pred = grid.predict(X_test)

print(classification_report(y_test, grid_pred))
print("##")
print(confusion_matrix(y_test, grid_pred))