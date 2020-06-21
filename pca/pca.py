import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(pca.components_)

#plt.figure(figsize=(8,6))
#plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'])
#plt.xlabel('First Principal Component')
#plt.ylabel('Second Principal Component')

df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(12, 6))
sns.heatmap(df_comp, cmap='plasma')

plt.show()

