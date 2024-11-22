"""
Clustering with Multiple Methods on the Iris Dataset

This script applies various clustering methods to the Iris dataset, including K-Means, K-Medoids,
Hierarchical Clustering, DBSCAN, and Gaussian Mixture Model (GMM). The script compares their 
clustering performance and visualizes the results.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame for visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df['true_labels'] = y

# Visualize the original dataset with true labels
sns.pairplot(df, hue='true_labels', palette='viridis')
plt.title('Iris Dataset with True Labels')
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
df['kmeans'] = kmeans.predict(X_scaled)

sns.pairplot(df, hue='kmeans', palette='viridis')
plt.title('K-Means Clustering Results')
plt.show()

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(X_scaled)
df['kmedoids'] = kmedoids.predict(X_scaled)

sns.pairplot(df, hue='kmedoids', palette='viridis')
plt.title('K-Medoids Clustering Results')
plt.show()

# Apply Hierarchical Clustering (Agglomerative Clustering)
hierarchical = AgglomerativeClustering(n_clusters=3)
df['hierarchical'] = hierarchical.fit_predict(X_scaled)

sns.pairplot(df, hue='hierarchical', palette='viridis')
plt.title('Hierarchical Clustering Results')
plt.show()

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.9, min_samples=5).fit(X_scaled)
df['dbscan'] = dbscan.labels_

sns.pairplot(df, hue='dbscan', palette='viridis')
plt.title('DBSCAN Clustering Results')
plt.show()

# Apply Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3, random_state=0).fit(X_scaled)
df['gmm'] = gmm.predict(X_scaled)

sns.pairplot(df, hue='gmm', palette='viridis')
plt.title('Gaussian Mixture Model Clustering Results')
plt.show()
