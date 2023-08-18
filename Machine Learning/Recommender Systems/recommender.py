# Recommender Systems

# Library Imports
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# First DF is user data (every rating given by every user)
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
df.head()

# Reading in movie titles
movie_titles = pd.read_csv('Movie_Id_Titles')
movie_titles.head()

# Merging both DFs on the item id
df = pd.merge(df, movie_titles, on='item_id')
df.head()

# Top 5 rated movies
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

# Top 5 number reviewed movies
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

# Making a ratings DF with average rating per movie and number of ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['# ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

# Rating count drops heavily after ~100
plt.figure(figsize=(10, 4))
ratings['# ratings'].hist(bins=70)

# Spikes in the data at full values (1 star, 2 stars, etc)
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)

# Jointplot showing the perspective of the two hists shown above
sns.jointplot(data=ratings, x='rating', y='# ratings', alpha=0.5)

# Making a povit table of users and the movies they've reviewed
movies_matrix = df.pivot_table(
    index='user_id', columns='title', values='rating')

ratings.sort_values('# ratings', ascending=False).head(10)

# We'll focus on two movies for this example: Star Wars and Liar Liar
starwars_ratings = movies_matrix['Star Wars (1977)']
liarliar_ratings = movies_matrix['Liar Liar (1997)']

# calculating the correlation of rating for our selected movies with all others
starwars_similar_ratings = movies_matrix.corrwith(starwars_ratings)
liarliar_similar_ratings = movies_matrix.corrwith(liarliar_ratings)

# Making a DF of correlations with Star Wars
corr_starwars = pd.DataFrame(starwars_similar_ratings, columns=[
                             'Correlation with Star Wars'])
corr_starwars.dropna(inplace=True)
corr_starwars.sort_values('Correlation with Star Wars',
                          ascending=False).head(10)
# Problem here is that some movies have perfect correlation with Star Wars,
# even if very few people reviewed the movie
# Join with our ratings DF to filter on a certain cutoff for # of ratings
corr_starwars = corr_starwars.join(ratings['# ratings'])
corr_starwars.head(10)
# Now we have a top 10 listing of movies we can recommend for users who enjoyed Star Wars
corr_starwars[corr_starwars['# ratings'] > 100].sort_values(
    'Correlation with Star Wars', ascending=False).head(10)

# We can follow the same process for Liar Liar:
corr_liarliar = pd.DataFrame(liarliar_similar_ratings, columns=[
                             'Correlation with Liar Liar'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['# ratings'])
corr_liarliar.head(10)
# Top 10 movies to recommend for users who liked Liar Liar
corr_liarliar[corr_liarliar['# ratings'] > 100].sort_values(
    'Correlation with Liar Liar', ascending=False).head(10)
