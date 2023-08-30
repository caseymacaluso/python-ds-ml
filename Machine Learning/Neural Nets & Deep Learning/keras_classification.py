# Keras Classification

# Library imports
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading in data
df = pd.read_csv('DATA/cancer_classification.csv')

# Inspecting data
df.info()
df.describe().transpose()

# Countplot showing # of benign vs # malignant tumors
sns.countplot(x='benign_0__mal_1', data=df)

# Plotting correlation of other columns vs. 'benign_0__mal_1'
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# X and y for model
X = df.drop('benign_0__mal_1', axis=1)
y = df['benign_0__mal_1']

# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)

# Scaling training features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model definition
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Example of overfitting
model.fit(x=X_train, y=y_train, epochs=600,
          validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
# validation loss starts to get out of control, we need to try and find a good cutoff point
model_loss.plot()

# Redefine model, this time with an EarlyStopping parameter
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Early stop param, will find an optimal place to stop if loss improvement doesn't happen after a certain point
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=25)

# Notice this time, it stops much sooner than the first model
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])

# Validation losses are much more controlled here
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# Defining model one more time, this time with Dropout layers
# Dropout layers randomly set input units to 0 with a specific frequency,
# which helps to prevent overfitting
model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
# Notice this model went a little longer than the second, likely as some inputs were randomly turned off during fitting
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])

# Plotting final losses
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# Predicting with testing data
predictions = (model.predict(X_test) > 0.5)*1

# Metrics
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.95      0.98      0.96        55
#            1       0.99      0.97      0.98        88

#     accuracy                           0.97       143
#    macro avg       0.97      0.97      0.97       143
# weighted avg       0.97      0.97      0.97       143

# [[54  1]
#  [ 3 85]]

# Conclusion: Model performed pretty well, only misclassified a very small number of entries
