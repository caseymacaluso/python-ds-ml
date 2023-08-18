# Support Vector Machine (SVM) Notes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

# Gives an overview of the data and measures contained within
print(cancer['DESCR'])

print(cancer['feature_names'])

# Making our dataframe
df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df_feat.head()
df_feat.info()

print(cancer['target'])
print(cancer['target_names'])
# 0 = malignant, 1 = benign

df_target = pd.DataFrame(cancer['target'], columns = ['Benign'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)

from sklearn.svm import SVC
svm = SVC(gamma='auto')

svm.fit(x_train, y_train)
predictions = svm.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00        66
#            1       0.61      1.00      0.76       105

#     accuracy                           0.61       171
#    macro avg       0.31      0.50      0.38       171
# weighted avg       0.38      0.61      0.47       171

# [[  0  66]
#  [  0 105]]

# Notice how everything gets classified into one bin
# NOTE: this is with the 'gamma' value on our SVC set to 'auto'. by default, it is set to 'scale' and results for that
# Appear to shape a more accurate model in this case. However, this gives us an opportunity to use
# GridSearch to help define parameters

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(x_train, y_train)

grid.best_params_

grid_predictions = grid.predict(x_test)

print(classification_report(y_test, grid_predictions))
print(confusion_matrix(y_test, grid_predictions))

#               precision    recall  f1-score   support

#            0       0.94      0.89      0.91        66
#            1       0.94      0.96      0.95       105

#     accuracy                           0.94       171
#    macro avg       0.94      0.93      0.93       171
# weighted avg       0.94      0.94      0.94       171

# [[ 59   7]
#  [  4 101]]

# Our GridSearch model performs much better than our original model