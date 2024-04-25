import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

st.title("Interactive K-Means Clustering Exploration")

# Introduction
st.markdown("""
## Introduction to K-Means Clustering
K-Means clustering is a popular unsupervised learning algorithm used to partition a set of points into K distinct non-overlapping subgroups (clusters) where each point belongs to only one group. It is widely used in data analysis for pattern recognition, image compression, and finding similar items in data.
""")

# Concept and Algorithm
st.subheader("Concept and Algorithm")
st.markdown("""
K-Means clustering works by assigning each data point to the nearest cluster, while keeping the centroids as small as possible. The algorithm alternates between two steps:
1. **Assignment step**: Assign each point to the nearest cluster centroid.
2. **Update step**: Compute the centroids of the clusters formed by the re-assigned points.
""")
st.latex(r'''
\text{Repeat until the assignments do not change.}
''')
st.markdown("""
This process minimizes the within-cluster sum of squares (inertia) and is typically used to make the data points within a cluster more similar (or closer) to each other than to those in other clusters.
""")

# Sidebar Configuration
st.sidebar.header('Configuration')
num_clusters = st.sidebar.slider('Number of clusters', 1, 10, 3)
num_samples = st.sidebar.slider('Number of samples', 50, 1000, 300)
random_state = st.sidebar.slider('Random seed', 0, 999, 42)
cluster_std = st.sidebar.slider('Cluster standard deviation', 0.1, 2.0, 0.5)

# Data Generation
st.subheader("Data Preparation")
st.markdown("""
Here we generate synthetic data suitable for clustering to demonstrate how K-Means works. This synthetic dataset allows users to observe how changing parameters like number of clusters and sample size impacts the clustering result.
""")
X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=cluster_std, random_state=random_state)

# K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
clusters = kmeans.fit_predict(X)

# Visualization
st.subheader("Visualization of Clustered Data")
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, alpha=0.6)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=50, c='red', marker='X')  # mark the centroids
plt.title("Visualization of clustered data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(fig)

# Metrics and Interpretation
st.subheader("Metrics and Interpretation")
inertia = kmeans.inertia_
st.write(f"Sum of squared distances to centroids (Inertia): {inertia:.2f}")
st.markdown("""
The inertia is a measure of how internally coherent clusters are. It is the sum of the squared distances between each point and its closest centroid. Lower values are better and zero is optimal.
""")

st.markdown("""
### Conclusion
K-Means is highly effective for datasets where the structure is clear and distinct. However, it can struggle with complex geometries or varying sizes of clusters. Experimenting with the parameters can provide deeper insights into the clustering dynamics.
""")
