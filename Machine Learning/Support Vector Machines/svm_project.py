# Support Vector Machine (SVM) Project

 # Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Iris Data Import
iris = sns.load_dataset('iris')

# Appears to be a separation between setosa and the pairing of versicolor and virginica
sns.pairplot(iris, hue='species')

# KDE plot of sepal width and sepal length
plt.figure(figsize=(12,8))
sns.kdeplot(data=iris, x='sepal_width', y='sepal_length', fill=True, cmap='flare')

# Setting up our training and testing data
from sklearn.model_selection import train_test_split
x = iris.drop('species', axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=101)

# Model instantiation
from sklearn.svm import SVC
svm = SVC()

# Fitting to training data
svm.fit(x_train, y_train)

# Predicting on testing data
predictions = svm.predict(x_test)

# Show model performance metrics
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        13
#   versicolor       1.00      0.95      0.97        20
#    virginica       0.92      1.00      0.96        12

#     accuracy                           0.98        45
#    macro avg       0.97      0.98      0.98        45
# weighted avg       0.98      0.98      0.98        45

# [[13  0  0]
#  [ 0 19  1]
#  [ 0  0 12]]

# Very strong model performance here, but let's practice with GridSearch to see if we can get even better
# Note that this is with the default 'gamma' value on the SVC object set to 'scale' which is why the model appears to perform better here

# Setting up parameter grid to test the model under a number of combinations
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

# GridSearch instantiation
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# Fit to training data
grid.fit(x_train, y_train)

grid.best_params_

# Predict on testing data
grid_predictions = grid.predict(x_test)

# Model performance metrics
print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))

#               precision    recall  f1-score   support

#       setosa       1.00      1.00      1.00        13
#   versicolor       1.00      0.95      0.97        20
#    virginica       0.92      1.00      0.96        12

#     accuracy                           0.98        45
#    macro avg       0.97      0.98      0.98        45
# weighted avg       0.98      0.98      0.98        45

# [[13  0  0]
#  [ 0 19  1]
#  [ 0  0 12]]

# Performance here was exactly the same as the previous model, so we have a solid SVM model in use.