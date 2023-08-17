# K Nearest Neighbors Notes

# Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Imports
df = pd.read_csv("Classified Data.csv", index_col=0)
df.head()

# Need to ensure that all our data is on the same scale for KNN
# That way, we can more accurately determine the target class.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fitting the scalar to all but the target variable
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
# Making a dataframe of our scaled features
df_features = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_features.head()

# Creating our training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_features, df['TARGET CLASS'], test_size=0.30, random_state=101)

# Creating our KNN model, starting with neighbors at 1 (we'll alter this later)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
#               precision    recall  f1-score   support

#            0       0.91      0.95      0.93       159
#            1       0.94      0.89      0.92       141

#     accuracy                           0.92       300
#    macro avg       0.92      0.92      0.92       300
# weighted avg       0.92      0.92      0.92       300

# [[151   8]
#  [ 15 126]]

# Stats look pretty good here, but let's see if we can make them even better


# Elbow Method to find best k-value for the model
# Goal: plot the error rate (when prediction and actual don't match) and see at what k-value we get the lowest error rate
error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_rate, color='green', ls='dashed', marker='o', markersize=10)
plt.title('Error Rate vs K Value')

# k = 18 seems like a good spot to be at error wise. Error becomes a little volatile after that. Plus, error is already pretty low

knn = KNeighborsClassifier(n_neighbors = 18)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

#               precision    recall  f1-score   support

#            0       0.94      0.97      0.96       159
#            1       0.97      0.93      0.95       141

#     accuracy                           0.95       300
#    macro avg       0.95      0.95      0.95       300
# weighted avg       0.95      0.95      0.95       300

# [[155   4]
#  [ 10 131]]

# Managed to make some minor improvements to our model by choosing that higher k-value!