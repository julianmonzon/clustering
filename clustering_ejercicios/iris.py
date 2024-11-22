"""
Clustering with K-Means and K-Medoids on the Iris Dataset

This script demonstrates clustering analysis on the Iris dataset, a popular dataset in machine learning. 
The features are scaled, and K-Means and K-Medoids clustering algorithms are applied to analyze 
and compare their clustering performance.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

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

# Visualize K-Means clustering results
sns.pairplot(df, hue='kmeans', palette='viridis')
plt.title('K-Means Clustering Results')
plt.show()

# Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(X_scaled)
df['kmedoids'] = kmedoids.predict(X_scaled)

# Visualize K-Medoids clustering results
sns.pairplot(df, hue='kmedoids', palette='viridis')
plt.title('K-Medoids Clustering Results')
plt.show()
