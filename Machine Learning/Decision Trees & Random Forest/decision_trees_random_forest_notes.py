# Decision Trees & Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data covers individuals involved in spinal surgery to correct kyphosis
df = pd.read_csv('kyphosis.csv')
df.head()
# Kyphosis = whether or not the condition was present after the surgery
# Age = Age measured in months
# Number = number of vertebrae involved in operation
# Start = topmost vertebrae that the surgery started on

# Some exploratory DA to get a feel for the data
sns.pairplot(df, hue='Kyphosis')

# Making our training and testing data
from sklearn.model_selection import train_test_split
x = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=101)

# Importing and defining our decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# Fit to our training data
dtree.fit(x_train, y_train)
# Make predictions with our testing data
predictions = dtree.predict(x_test)

# Import metrics to assess model performance
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#       absent       0.68      0.76      0.72        17
#      present       0.33      0.25      0.29         8

#     accuracy                           0.60        25
#    macro avg       0.51      0.51      0.50        25
# weighted avg       0.57      0.60      0.58        25

# [[13  4]
#  [ 6  2]]

# Model did okay, did end up misclassifying 10 observations. Let's try a random forest to see how this performs.

# Import and define our random forest algorithm
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

# Fit RF model to training data
rfc.fit(x_train, y_train)
# Make predictions with testing data
rfc_predictions = rfc.predict(x_test)
print(classification_report(y_test, rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))

#               precision    recall  f1-score   support

#       absent       0.74      1.00      0.85        17
#      present       1.00      0.25      0.40         8

#     accuracy                           0.76        25
#    macro avg       0.87      0.62      0.62        25
# weighted avg       0.82      0.76      0.71        25

# [[17  0]
#  [ 6  2]]

# Random forest model performed better than the decision tree. Only 6 Type 1 (false positive) errors,
# And better model statistics are observed here (precision, accuracy, recall, etc.)