# K Means Cluster Notes

from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Making a mock dataset to simulate our clusters
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.8, random_state=101)

# Scatterplot of our mock data
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

# Import and instantiate K-Means
# Choosing clusters of 4 to match our mock data
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
kmeans.cluster_centers_

# Plotting the difference between the K-Means algorithm and the actual clusters
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

# Both clusters are pretty similar, there's some noise in the middle with the k-means algorithm
