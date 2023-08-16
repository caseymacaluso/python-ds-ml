# Logistic Regression Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')

ad_data.head()
ad_data.info()
ad_data.describe()

# Showing the age distribution of our data
ad_data['Age'].hist(bins=40)

# Tend to earn more as people get into their 40's, income falls of as people get older
sns.jointplot(x='Age', y='Area Income', data=ad_data)

# kde plot for Age vs. Time spent on site
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind="kde")

# Generally speaking, less time spent on internet = less time spent on site
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)

# Pairplot for the data, setting hue on the variable we want to predict
sns.pairplot(data=ad_data, hue='Clicked on Ad')

# Setting our x and y for our logistic regression model
x = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# Importing model tools from sklearn
from sklearn.model_selection import train_test_split

# Splitting data for training and testing sets, doing a 30% training set size
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

# Fitting model to training data
logmodel.fit(x_train, y_train)

# Making predictions based on our testing data
predictions = logmodel.predict(x_test)

# Metrics to evaluate model performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Confusion matrix to evaluate statistics
confusion_matrix(y_test, predictions)
  # [[149,   8]
  #  [ 14, 129]]
  
# Classification report to illustrate model performance
print(classification_report(y_test, predictions))
#               precision    recall  f1-score   support

#            0       0.91      0.95      0.93       157
#            1       0.94      0.90      0.92       143

#     accuracy                           0.93       300
#    macro avg       0.93      0.93      0.93       300
# weighted avg       0.93      0.93      0.93       300

# Summary: model performed very well when evaulating on testing data. We had precision, recall, f1-score and accuracy metrics all above 90%