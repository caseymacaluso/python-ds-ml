# Decision Trees & Random Forest - Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')
loans.info()
loans.head()
loans.describe()
# Data Description:
# credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# installment: The monthly installments owed by the borrower if the loan is funded.
# log.annual.inc: The natural log of the self-reported annual income of the borrower.
# dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# fico: The FICO credit score of the borrower.
# days.with.cr.line: The number of days the borrower has had a credit line.
# revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# Histogram showing fico score, with credit.policy as different colors
plt.figure(figsize=(12,6))
sns.histplot(data=loans, x='fico', hue='credit.policy', bins=30)

# Histogram showing fico score, with not.fully.paid as different colors
plt.figure(figsize=(12,6))
sns.histplot(data=loans, x='fico', hue='not.fully.paid', bins=30)

# Countplot showing different counts segmented by the 'purpose' column
plt.figure(figsize=(12,6))
sns.countplot(x='purpose', data=loans, hue='not.fully.paid')

# Jointplot showing spread of data points when looking at fico score vs interest rate
sns.jointplot(x='fico', y='int.rate', data=loans)

# Linear plots showing trend of fico score vs interest rate depending on whether the loan has been fully paid off
sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid')

loans.info()

# Need to convert purpose column to dummy categories
cat_feats = ['purpose']
# Does the dummyfying and dataframe creation all in one step. Could also do it in separate steps to be clear, but this works very cleanly.
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)

# Set up our training and testing data
from sklearn.model_selection import train_test_split
x = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=101)

# Import and define our decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

# Fit model to training data and make predictions on testing data
dtree.fit(x_train, y_train)
predictions = dtree.predict(x_test)

# Metrics to check performance
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.86      0.82      0.84      2431
#            1       0.20      0.25      0.22       443

#     accuracy                           0.73      2874
#    macro avg       0.53      0.53      0.53      2874
# weighted avg       0.75      0.73      0.74      2874

# [[1985  446]
#  [ 334  109]]

# Seems to be misclassifying a good number of observations, so let's see if we can trim that down.

# Import and define our random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

# Fit to training data and make predictions on testing data
rfc.fit(x_train, y_train)
rfc_predictions = rfc.predict(x_test)

# Metrics to assess performance
print(classification_report(y_test, rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))

#               precision    recall  f1-score   support

#            0       0.85      1.00      0.92      2431
#            1       0.57      0.03      0.05       443

#     accuracy                           0.85      2874
#    macro avg       0.71      0.51      0.48      2874
# weighted avg       0.81      0.85      0.78      2874

# [[2422    9]
#  [ 431   12]]

# Neither model did a superb job at capturing the data everywhere.
# The decision tree model had better results for the '1' class in recall and f-score,
# While the random forest model had better accuracy and misclassified observations less.

# Overall, the random forest model worked better in this scenario, but depending on the situation, the decision tree model
# may be a better choice for the task at hand.