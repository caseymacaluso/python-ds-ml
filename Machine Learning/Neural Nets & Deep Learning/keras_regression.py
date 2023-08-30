# Keras Regression

# Library Imports
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data
df = pd.read_csv('DATA/kc_house_data.csv')

# Checking nulls
df.isnull().sum()
df.describe().transpose()

# Distribution of the price
plt.figure(figsize=(12, 8))
sns.distplot(df['price'])

# Countplot for houses with different # of bedrooms
sns.countplot(df['bedrooms'])

# Checking correlation with other columns vs. price
df.corr()['price'].sort_values()

# Scatterplot of 'sqft_living' vs. 'price'
plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['sqft_living'])

# Plotting price vs the geographical coordinates
plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['long'])
plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['lat'])

# Making a pseudo-geographical plot
plt.figure(figsize=(12, 8))
sns.scatterplot(df['long'], df['lat'], hue=df['price'])

# Checking some of the highest prices
df.sort_values('price', ascending=False).head(20)

# How much is 1% of the data?
len(df)*0.01  # ~216

# Dataframe of the bottom 99% of the data
bottom_99 = df.sort_values('price', ascending=False).iloc[216:]

# Bit of a better look at prices based on location
plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=bottom_99, hue='price',
                palette='RdYlGn', edgecolor=None, alpha=0.2)

# Price based on whether a waterfront is present or not
sns.boxplot(x='waterfront', y='price', data=df)

df.head()
# id isn't useful here
df = df.drop('id', axis=1)

# Convert date, pull month and year information
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

sns.boxplot(x='year', y='price', data=df)
sns.boxplot(x='month', y='price', data=df)

# Graphing average price based on the listing month
df.groupby('month').mean()['price'].plot()

# Don't need the date column anymore
df = df.drop('date', axis=1)

# Don't need the zipcode here
df = df.drop('zipcode', axis=1)

# Checking value counts for remaining data
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()

# Define X and y for model
X = df.drop('price', axis=1).values
y = df['price'].values

# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

# Scaler for training and testing features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model
model = Sequential()

# Add layers
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit data to training data, add in testing data for validation, 400 cycles through data
model.fit(x=X_train, y=y_train, validation_data=(
    X_test, y_test), batch_size=128, epochs=400)

# Plot losses vs validation loss
losses = pd.DataFrame(model.history.history)
losses.plot()

# Make predictions with testing data
predictions = model.predict(X_test)

# MAE: 101577.874
mean_absolute_error(y_test, predictions)

# RMSE: 164173.259
np.sqrt(mean_squared_error(y_test, predictions))

# R^2: 0.797
explained_variance_score(y_test, predictions)

# Plot showing general model performance
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

# Testing model on a single house
single_house = df.drop('price', axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19))
model.predict(single_house)  # [[289484.06]]
df.iloc[0]['price']  # 221900.0

# Conclusion: though we're able to explain roughly 80% of the variance with the model, we do appear to overshoot in some cases. Some model
# adjustments may help to clean this up.
