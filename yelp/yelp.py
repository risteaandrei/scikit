import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

sns.set_style('white')

yelp = pd.read_csv('yelp/yelp.csv')
#print(yelp.info())

yelp['text length'] = yelp['text'].apply(len)

#g = sns.FacetGrid(yelp, col='funny')
#g.map(plt.hist, 'text length', bins=50)
#sns.boxplot(x='stars', y='text length', data=yelp)
#sns.countplot(x='stars', data=yelp)
#stars = yelp.groupby('stars').mean()
#sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)
#plt.show()

yelp_class = yelp[(yelp['stars']==1)|(yelp['stars']==5)]

X = yelp_class['text']
y = yelp_class['stars']

#cv = CountVectorizer()
#X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 

#nb = MultinomialNB()
#nb.fit(X_train, y_train)

#predictions = nb.predict(X_test)

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))

pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', MultinomialNB())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
