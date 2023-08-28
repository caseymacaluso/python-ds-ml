# Keras Regression

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('DATA/kc_house_data.csv')

df.isnull().sum()

df.describe().transpose()

plt.figure(figsize=(12, 8))
sns.distplot(df['price'])

sns.countplot(df['bedrooms'])

df.corr()['price'].sort_values()

plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['sqft_living'])

plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['long'])

plt.figure(figsize=(12, 8))
sns.scatterplot(df['price'], df['lat'])

plt.figure(figsize=(12, 8))
sns.scatterplot(df['long'], df['lat'], hue=df['price'])

df.sort_values('price', ascending=False).head(20)

len(df)*0.01  # ~216

bottom_99 = df.sort_values('price', ascending=False).iloc[216:]

plt.figure(figsize=(12, 8))
sns.scatterplot(x='long', y='lat', data=bottom_99, hue='price',
                palette='RdYlGn', edgecolor=None, alpha=0.2)

sns.boxplot(x='waterfront', y='price', data=df)

df.head()

df = df.drop('id', axis=1)

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].apply(lambda date: date.month)
df['year'] = df['date'].apply(lambda date: date.year)

sns.boxplot(x='year', y='price', data=df)
sns.boxplot(x='month', y='price', data=df)

df.groupby('month').mean()['price'].plot()

df = df.drop('date', axis=1)

df = df.drop('zipcode', axis=1)

df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()


X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x=X_train, y=y_train, validation_data=(
    X_test, y_test), batch_size=128, epochs=400)

losses = pd.DataFrame(model.history.history)

losses.plot()


predictions = model.predict(X_test)

mean_absolute_error(y_test, predictions)

np.sqrt(mean_squared_error(y_test, predictions))

explained_variance_score(y_test, predictions)

plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')

single_house = df.drop('price', axis=1).iloc[0]

single_house = scaler.transform(single_house.values.reshape(-1, 19))

model.predict(single_house)
df.iloc[0]
