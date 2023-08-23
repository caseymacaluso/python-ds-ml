# Tensorflow Basics

# Library Imports
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading in data
df = pd.read_csv('DATA/fake_reg.csv')

# Exploring the data
df.head()
sns.pairplot(df)

# Setting our variables for training and testing data
X = df[['feature1', 'feature2']].values
y = df['price'].values

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)


# help(MinMaxScaler)

# Need to scale our feature set
scaler = MinMaxScaler()

scaler.fit(X_train)

# Scaling both training and testing features
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define our model
model = Sequential()
# Adding three layers with four nodes in each
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
# One end node to cap off the neural net
model.add(Dense(1))

# Compile the model (since this is more of a regression problem, we use 'mse' to calculate loss)
model.compile(optimizer='rmsprop', loss='mse')

# Fit model (epochs = number of times the model goes through the dataset)
model.fit(x=X_train, y=y_train, epochs=250)

# List of loss values recorded per epoch
model.history.history

loss = model.history.history['loss']

# Plotting loss per epoch. Loss amount drops heavily at a certain point, then levels off
sns.lineplot(x=range(len(loss)), y=loss)
plt.title('Training Loss per Epoch')

model.metrics_names

# Evaluating training and testing scores
training_score = model.evaluate(
    X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)

# Make model predictions based on scaled testing data
test_predictions = model.predict(X_test)

# Formatting a DF that has the true Y (y_test) and predicted values (test_predictions)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=['True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['True Y', 'Model Predictions']
pred_df.head()

# Pretty strong model performance here
sns.scatterplot(data=pred_df, x='True Y', y='Model Predictions')

# Low MAE
mean_absolute_error(pred_df['True Y'], pred_df['Model Predictions'])

# Saving model for future use
model.save('gem_model.h5')

# Example of loading saved model into new variable
new_model = load_model('gem_model.h5')
