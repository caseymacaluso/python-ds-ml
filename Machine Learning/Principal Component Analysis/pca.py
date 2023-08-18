# Principal Component Analysis

# Library Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Data Imports
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

# Make a dataframe of the data
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()

# Need to scale our data to ensure we make appropriate estimations
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
# Instantiate our PCA model. Doing 2 principal components, but can go higher if you wish
# Fitting to scaled data
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

# Original data had 30 columns
scaled_data.shape

# PCA data has 2
x_pca.shape

# Since our PCA data has 2 columns, we can easily represent this with a scatterplot
plt.figure(figsize=(12, 9))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
# With this plot, we can see a clear separation between observations that have malignant or benign tumors

print(pca.components_)

# Composing these PCA components into a dataframe to investigate them further
df_components = pd.DataFrame(pca.components_, columns=cancer['feature_names'])

# Heatmap that shows the correlation between the features and the component itself.
# Using this and some domain knowledge, an inference can be made as to what each component vaguely represents
plt.figure(figsize=(12, 8))
sns.heatmap(df_components, cmap='plasma')

# Example:
# Comp 1 = Mass Characteristics (smoothness, compactness, etc.)
# Comp 2 = Mass Size
