# Tensorboard Example

# Code here is mostly the same from the 'keras_classification' file

from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Reading in data
df = pd.read_csv('DATA/cancer_classification.csv')

X = df.drop('benign_0__mal_1', axis=1)
y = df['benign_0__mal_1']

# Training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101)

# Scaling training features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Early stop param, will find an optimal place to stop if loss improvement doesn't happen after a certain point
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=25)

log_directory = 'logs\\fit'
board = TensorBoard(log_dir=log_directory, histogram_freq=1, write_graph=True,
                    write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)

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
    X_test, y_test), verbose=1, callbacks=[early_stop, board])

print(log_directory)

# Navigate to project directory and run the following line:
# tensorboard --logdir logs\fit
# Tensorboard should be running on localhost:6006
