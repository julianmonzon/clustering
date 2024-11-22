"""
Clustering with K-Means and K-Medoids (Including Multiple Outliers)

This script demonstrates clustering analysis on a synthetic dataset with three clusters and multiple outliers. 
It uses K-Means and K-Medoids algorithms to showcase their clustering capabilities and their behavior with outliers.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create the synthetic dataset
df1 = pd.DataFrame({
    'x': np.random.normal(5, 3, 100),
    'y': np.random.normal(-2, 2, 100)
})

df2 = pd.DataFrame({
    'x': np.random.normal(15, 2, 100),
    'y': np.random.normal(22, 2, 100)
})

df3 = pd.DataFrame({
    'x': np.random.normal(-5, 3, 100),
    'y': np.random.normal(8, 2, 100)
})

# Step 2: Combine the clusters and add outliers
df = pd.concat([df1, df2, df3], ignore_index=True)
outliers = pd.DataFrame({'x': [200, -200], 'y': [200, -200]})
df = pd.concat([df, outliers], ignore_index=True)

# Step 3: Visualize the original dataset
sns.relplot(data=df, x='x', y='y')
plt.title('Original Data with Outliers')
plt.show()

# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

# Visualize K-Means clustering results
sns.scatterplot(data=df, x='x', y='y', hue=kmeans.predict(df), palette='viridis')
plt.title('K-Means Clustering Results')
plt.legend(title='Cluster')
plt.show()

# Step 5: Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(df)

# Visualize K-Medoids clustering results
sns.scatterplot(data=df, x='x', y='y', hue=kmedoids.predict(df), palette='viridis')
plt.title('K-Medoids Clustering Results')
plt.legend(title='Cluster')
plt.show()