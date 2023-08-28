# Keras Classification

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

df = pd.read_csv('DATA/cancer_classification.csv')

df.info()
df.describe().transpose()

sns.countplot(x='benign_0__mal_1', data=df)

df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

X = df.drop('benign_0__mal_1', axis=1)
y = df['benign_0__mal_1']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# Example of overfitting
model.fit(x=X_train, y=y_train, epochs=600,
          validation_data=(X_test, y_test), verbose=1)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')


early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

predictions = (model.predict(X_test) > 0.5)*1


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
