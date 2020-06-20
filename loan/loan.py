import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

loans = pd.read_csv('loan/loan_data.csv')
#print(loans.info())

# sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')
# plt.show()

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
#print(final_data.info())

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)

# dtree = DecisionTreeClassifier()
# dtree.fit(X_train, y_train)
# 
# predictions = dtree.predict(X_test)
# 
# print(classification_report(y_test, predictions))
# print('##')
# print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

print(classification_report(y_test, predictions))
print('##')
print(confusion_matrix(y_test, predictions))
