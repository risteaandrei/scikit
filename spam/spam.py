import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ssl
import string

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import pickle

def text_process(mess):
    """
    1. remove punctuation
    2. remove stop words
    3. return list of clean text words
    """
    nopunc = [c for c in mess if c not in string.punctuation]
    nopunc = ''.join(nopunc).split()
    return [word for word in nopunc if word.lower() not in stopwords.words('english')]

#ssl._create_default_https_context = ssl._create_unverified_context
#nltk.download_shell()

messages = pd.read_csv('spam/SMSSpamCollection', sep='\t', names=['label', 'message'])
messages['length'] = messages['message'].apply(len)

#bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
#pickle.dump(bow_transformer.vocabulary_, open('spam/vocabulary.pkl', 'wb'))

#bow_transformer = CountVectorizer()
#bow_transformer.vocabulary_ = pickle.load(open('spam/vocabulary.pkl', 'rb'))

#messages_bow = bow_transformer.transform(messages['message'])

#tfidf_transformer = TfidfTransformer()
#tfidf_transformer.fit(messages_bow)

#messages_tfidf = tfidf_transformer.transform(messages_bow)

#spam_detect_model = MultinomialNB.fit(messages_tfidf, messages['label'])
#all_pred = spam_detect_model.predict(messages_tfidf)

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(label_test, predictions))