# K-Means Clustering Project

# Library Imports
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data Import
data = pd.read_csv('College_Data.csv', index_col=0)

# Data Description
# Private: A factor with levels No and Yes indicating private or public university
# Apps: Number of applications received
# Accept: Number of applications accepted
# Enroll: Number of new students enrolled
# Top10perc: Pct. new students from top 10% of H.S. class
# Top25perc: Pct. new students from top 25% of H.S. class
# F.Undergrad: Number of fulltime undergraduates
# P.Undergrad: Number of parttime undergraduates
# Outstate: Out-of-state tuition
# Room.Board: Room and board costs
# Books: Estimated book costs
# Personal: Estimated personal spending
# PhD: Pct. of faculty with Ph.D.’s
# Terminal: Pct. of faculty with terminal degree
# S.F.Ratio: Student/faculty ratio
# perc.alumni: Pct. alumni who donate
# Expend: Instructional expenditure per student
# Grad.Rate: Graduation rate

data.head()
data.info()
data.describe()

# Scatterplot showing the room & board cost vs graduation rate, with private school set at a distinguisher
plt.figure(figsize=(12, 12))
sns.scatterplot(data=data, x='Room.Board', y='Grad.Rate', hue='Private')

# Scatterplot showing out of state tuition vs # fulltime undergrads, with private school set at a distinguisher
plt.figure(figsize=(12, 12))
sns.scatterplot(data=data, x='Outstate', y='F.Undergrad', hue='Private')

# Histogram showing out of state tuition, with private school set at a distinguisher
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='Outstate', hue='Private', bins=20)

# Histogram showing graduation rate, with private school set at a distinguisher
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='Grad.Rate', hue='Private', bins=20)
# One observation has a graduation rate above 100%
print(data[data['Grad.Rate'] > 100])

# Find that observation and set the grad rate to 100
data.loc[data['Grad.Rate'] > 100, 'Grad.Rate'] = 100

# Checking histogram again to ensure things are working as intented
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='Grad.Rate', hue='Private', bins=20)

# Defining our K Means model
km = KMeans(n_clusters=2)
km.fit(data.drop('Private', axis=1))

# Checking cluster centers
km.cluster_centers_
# ([[1.81323468e+03, 1.28716592e+03, 4.91044843e+02, 2.53094170e+01,
#         5.34708520e+01, 2.18854858e+03, 5.95458894e+02, 1.03957085e+04,
#         4.31136472e+03, 5.41982063e+02, 1.28033632e+03, 7.04424514e+01,
#         7.78251121e+01, 1.40997010e+01, 2.31748879e+01, 8.93204634e+03,
#         6.50926756e+01],
#        [1.03631389e+04, 6.55089815e+03, 2.56972222e+03, 4.14907407e+01,
#         7.02037037e+01, 1.30619352e+04, 2.46486111e+03, 1.07191759e+04,
#         4.64347222e+03, 5.95212963e+02, 1.71420370e+03, 8.63981481e+01,
#         9.13333333e+01, 1.40277778e+01, 2.00740741e+01, 1.41705000e+04,
#         6.75925926e+01]])

# Adding a dummy variable to our data so we can compare data groups with our model
data['Cluster'] = np.where(data['Private'] == 'Yes', 1, 0)
data.head()

# Metrics to compare our 'Cluster' dummy column with the labels from our model
print(classification_report(data['Cluster'], km.labels_))
print(confusion_matrix(data['Cluster'], km.labels_))

#               precision    recall  f1-score   support

#            0       0.21      0.65      0.31       212
#            1       0.31      0.06      0.10       565

#     accuracy                           0.22       777
#    macro avg       0.26      0.36      0.21       777
# weighted avg       0.29      0.22      0.16       777

# [[138  74]
#  [531  34]]

# Poor model performance, but bear in mind the model doesn't have access to the 'Private' label for clustering,
# and is purely using the other features in the data to attempt clustering of the data.

# Let's investigate further by adding an inverted cluster column to our data and compare those
data['Cluster_inv'] = np.where(data['Private'] == 'Yes', 0, 1)
print(classification_report(data['Cluster_inv'], km.labels_))
print(confusion_matrix(data['Cluster_inv'], km.labels_))

#               precision    recall  f1-score   support

#            0       0.79      0.94      0.86       565
#            1       0.69      0.35      0.46       212

#     accuracy                           0.78       777
#    macro avg       0.74      0.64      0.66       777
# weighted avg       0.76      0.78      0.75       777

# [[531  34]
#  [138  74]]

# Much better model performance here. It's possible that in the underlying logic for the model,
# the cluster labels got flipped around
