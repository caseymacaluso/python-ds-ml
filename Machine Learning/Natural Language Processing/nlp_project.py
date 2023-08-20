# Natural Language Processing (NLP) Project

# Library Imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import and inspect data
yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

# Adding 'length' column which shows the character length of each review
yelp['length'] = yelp['text'].apply(len)

# Grid of histograms that shows the distribution of review lengths for each star rating
g = sns.FacetGrid(data=yelp, col='stars')
g.map(sns.histplot, 'length', bins=10)

# Boxplots for each star rating vs review length
sns.boxplot(data=yelp, x='stars', y='length')

# Countplot showing the number of reviews in each star rating
sns.countplot(data=yelp, x='stars')

# DF that shows the average for the numeric columns for each star rating
avgs = yelp.groupby('stars').mean()

# Making a correlation matrix of the above DF
avg_corr = avgs.corr()

# Heatmap showing the correlations these values
sns.heatmap(data=avg_corr, cmap='plasma', annot=True)

# For this project, only using reviews with 1 or 5 stars
yelp_class = yelp[yelp['stars'].isin([1, 5])]

# Pulling 'text' and 'stars' columns for model training/testing
text = yelp_class['text']
stars = yelp_class['stars']

# Initializing our CountVectorizer
transformer = CountVectorizer()

# Transform the review text
text = transformer.fit_transform(text)

# Splitting data into training and testing data
text_train, text_test, stars_train, stars_test = train_test_split(
    text, stars, test_size=0.3, random_state=101)

# Initialize our Naive Bayes model
nb = MultinomialNB()

# Fit to training data
nb.fit(text_train, stars_train)

# Make predictions on testing data
predictions = nb.predict(text_test)

# Metrics
print(classification_report(stars_test, predictions))
print(confusion_matrix(stars_test, predictions))

#               precision    recall  f1-score   support

#            1       0.88      0.70      0.78       228
#            5       0.93      0.98      0.96       998

#     accuracy                           0.93      1226
#    macro avg       0.91      0.84      0.87      1226
# weighted avg       0.92      0.93      0.92      1226

# [[159  69]
#  [ 22 976]]

# Model did fairly well here, but let's see if incorporating TF-IDF changes anything

# Initialize a pipeline to clean up our process
pipeline = Pipeline([('bow', CountVectorizer()), ('tfidf',
                    TfidfTransformer()), ('classifier', MultinomialNB())])

# Redefining our variables (we overwrote the text variable with vectorized information)
text = yelp_class['text']
stars = yelp_class['stars']

# Redefine our training and testing data
text_train, text_test, stars_train, stars_test = train_test_split(
    text, stars, test_size=0.3, random_state=101)

# Fit pipeline to training data
pipeline.fit(text_train, stars_train)

# Make predictions on testing data
pipe_predictions = pipeline.predict(text_test)

# Metrics
print(classification_report(stars_test, pipe_predictions))
print(confusion_matrix(stars_test, pipe_predictions))

#               precision    recall  f1-score   support

#            1       0.00      0.00      0.00       228
#            5       0.81      1.00      0.90       998

#     accuracy                           0.81      1226
#    macro avg       0.41      0.50      0.45      1226
# weighted avg       0.66      0.81      0.73      1226

# [[  0 228]
#  [  0 998]]

# Interestingly, TF-IDF performed worse in this scenario. Could possibly incorporate different models in the future to see if results change
