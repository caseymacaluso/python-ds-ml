# K Nearest Neighbors Project

# Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Imports
df = pd.read_csv('KNN_Project_data.csv')
df.head()

# Some exploratory DA to get a feel for the data
sns.pairplot(df, hue='TARGET CLASS')

# Need to scale our data for this model to work
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fitting data on all but the target class
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
# Dataframe of our scaled features
df_features = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_features.head()

# Making our training and testing data (30% size)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_features, df['TARGET CLASS'], test_size=0.30, random_state=101)

# KNN model, starting with 1 neighbor and will adjust later
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

# Metrics to assess model performance
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.73      0.72      0.72       152
#            1       0.71      0.72      0.72       148

#     accuracy                           0.72       300
#    macro avg       0.72      0.72      0.72       300
# weighted avg       0.72      0.72      0.72       300

# [[109  43]
#  [ 41 107]]

# Definitely some chances for improvement here, so lets experiment to see what k-value will work best for us

# Goal: plot the error rate (when prediction != actual) and see which k-value gives the lowest error rate
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    predictions_i = knn.predict(x_test)
    error_rate.append(np.mean(predictions_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='green', ls='dashed', marker='o', markersize=8)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
# Let's do k = 31 (anything past doesn't really give much of a difference)

knn = KNeighborsClassifier(n_neighbors = 31)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.87      0.81      0.84       152
#            1       0.82      0.87      0.84       148

#     accuracy                           0.84       300
#    macro avg       0.84      0.84      0.84       300
# weighted avg       0.84      0.84      0.84       300

# [[123  29]
#  [ 19 129]]

# Nice improvements from our original model here. We were able to improve the accuracy by about 12%,
# And precision, recall and f1-scores all saw solid improvements as well.