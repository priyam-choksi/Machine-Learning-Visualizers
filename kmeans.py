import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

st.subheader("K-Means Clustering")

# Sidebar options
num_clusters = st.sidebar.slider('Number of clusters', 1, 10, 3)
num_samples = st.sidebar.slider('Number of samples', 50, 1000, 300)
random_state = st.sidebar.slider('Random seed', 0, 999, 42)
cluster_std = st.sidebar.slider('Cluster standard deviation', 0.1, 2.0, 0.5)

# Generate random data
X, _ = make_blobs(n_samples=num_samples, centers=num_clusters, cluster_std=cluster_std, random_state=random_state)

# K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
clusters = kmeans.fit_predict(X)

# Visualization
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, alpha=0.6)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], s=50, c='red', marker='X')  # mark the centroids
plt.title("Visualization of clustered data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(fig)

# Metrics
inertia = kmeans.inertia_
st.write(f"Sum of squared distances to centroids (Inertia): {inertia:.2f}")
