import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

st.subheader("Principal Component Analysis (PCA)")

# Sidebar options for configuration
num_features = st.sidebar.slider('Number of features', 2, 10, 5)
num_informative = st.sidebar.slider('Number of informative features', 1, num_features-1, 2)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
random_state = st.sidebar.slider('Random seed', 0, 9999, 42)

# Generate synthetic data
X, _ = make_classification(n_samples=num_samples, n_features=num_features, n_informative=num_informative, 
                           n_redundant=0, n_clusters_per_class=1, random_state=random_state)

# Standardize the data
X_scaled = StandardScaler().fit_transform(X)

# PCA transformation
pca = PCA(n_components=num_features)
X_pca = pca.fit_transform(X_scaled)

# Explained variance plot
fig, ax = plt.subplots()
ax.bar(range(1, num_features + 1), pca.explained_variance_ratio_, alpha=0.6, color='b')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Explained Variance by Each Principal Component')
ax.set_xticks(range(1, num_features + 1))
st.pyplot(fig)

# Scree plot
fig2, ax2 = plt.subplots()
ax2.plot(range(1, num_features + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Scree Plot')
ax2.grid(True)
st.pyplot(fig2)

# Visualization of the first two principal components if possible
if num_features >= 2:
    fig3, ax3 = plt.subplots()
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='blue')
    ax3.set_xlabel('First Principal Component')
    ax3.set_ylabel('Second Principal Component')
    ax3.set_title('Projection on the First Two Principal Components')
    st.pyplot(fig3)

# Display information about the components
st.write("Principal Components:")
st.write(pca.components_)
