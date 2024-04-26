import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

st.title("Principal Component Analysis (PCA) Visualizer")
st.write("""
This app demonstrates the PCA technique using synthetic data. PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
""")

# Sidebar options for configuration
st.sidebar.header('Data and PCA Configuration')
num_features = st.sidebar.slider('Number of features', 2, 10, 5)
num_informative = st.sidebar.slider('Number of informative features', 1, num_features - 1, 2)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
random_state = st.sidebar.slider('Random seed', 0, 9999, 42)

# Generate synthetic data
st.subheader("Data Generation")
st.write("""
The data generated here consists of several features, only some of which are informative. The rest are noise, mimicking real-world data where many features may not contribute to the signal.
""")
X, _ = make_classification(n_samples=num_samples, n_features=num_features, n_informative=num_informative, 
                           n_redundant=0, n_clusters_per_class=1, random_state=random_state)

# Standardize the data
st.write("Data is standardized to have zero mean and unit variance before applying PCA, which is critical as PCA is sensitive to the variances of the initial variables.")
X_scaled = StandardScaler().fit_transform(X)

# PCA transformation
pca = PCA(n_components=num_features)
X_pca = pca.fit_transform(X_scaled)

# Explained variance plot
st.subheader("Explained Variance Ratio")
st.write("""
The plot below shows the percentage of variance explained by each of the principal components. This helps to understand how much information each component captures from the data.
""")
fig, ax = plt.subplots()
ax.bar(range(1, num_features + 1), pca.explained_variance_ratio_, alpha=0.6, color='b')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Explained Variance by Each Principal Component')
ax.set_xticks(range(1, num_features + 1))
st.pyplot(fig)

# Scree plot
st.subheader("Scree Plot")
st.write("""
A scree plot displays the variance explained by each component and helps in deciding how many components are needed. It shows the cumulative variance against the number of components.
""")
fig2, ax2 = plt.subplots()
ax2.plot(range(1, num_features + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Scree Plot')
ax2.grid(True)
st.pyplot(fig2)

# Visualization of the first two principal components if possible
if num_features >= 2:
    st.subheader("First Two Principal Components")
    st.write("This scatter plot projects the data onto the first two principal components, showing the new feature space that captures most of the variance in the data.")
    fig3, ax3 = plt.subplots()
    ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c='blue')
    ax3.set_xlabel('First Principal Component')
    ax3.set_ylabel('Second Principal Component')
    ax3.set_title('Projection on the First Two Principal Components')
    st.pyplot(fig3)

# Display information about the components
st.write("Principal Components (each row corresponds to a component):")
st.write(pca.components_)
