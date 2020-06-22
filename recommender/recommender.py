import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('recommender/u.data', sep='\t', names=column_names)

movie_titles = pd.read_csv('recommender/Movie_Id_Titles')

df = pd.merge(df, movie_titles, on='item_id')

sns.set_style('white')

#print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
#print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = df.groupby('title')['rating'].count()
#print(ratings.head())

#ratings['num of ratings'].hist(bins = 70)
#ratings['rating'].hist(bins = 70)
#sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
#print(moviemat.head())

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_starwars = corr_starwars.join(ratings['num of ratings'])
#print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False).head())


corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False).head())

#plt.show()




