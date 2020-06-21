import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0

df = pd.read_csv('universities/College_Data', index_col=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))

df['Cluster'] = df['Private'].apply(converter)

print(confusion_matrix(df['Cluster'], kmeans.labels_))
print("##")
print(classification_report(df['Cluster'], kmeans.labels_))


