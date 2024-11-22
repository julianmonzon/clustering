"""
Clustering Analysis with K-Means and K-Medoids (Including an Outlier)

This script demonstrates clustering analysis on a synthetic dataset using K-Means and K-Medoids. 
It generates sample data, visualizes it, and applies both algorithms to compare their results.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create example data
cluster1 = np.random.normal(5, 3, (100, 2))
cluster2 = np.random.normal(15, 5, (100, 2))
cluster3 = np.random.normal(-5, 2, (100, 2))

# Add an outlier far from the clusters
outlier = np.array([[100, 100]])

# Combine clusters and the outlier into a single dataset
data = np.concatenate((cluster1, cluster2, cluster3, outlier), axis=0)

# Step 2: Check the shape of the dataset
print(f"Dataset shape: {data.shape}")

# Step 3: Visualize the original dataset
sns.scatterplot(x=data[:, 0], y=data[:, 1])
plt.title('Original Data')
plt.show()

# Step 4: Train the K-Means model
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

# Visualize K-Means clustering results
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=kmeans.predict(data), palette='viridis')
plt.title('K-Means Clustering Results')
plt.legend(title='Cluster')
plt.show()

# Step 5: Train the K-Medoids model
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(data)

# Visualize K-Medoids clustering results
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=kmedoids.predict(data), palette='viridis')
plt.title('K-Medoids Clustering Results')
plt.legend(title='Cluster')
plt.show()