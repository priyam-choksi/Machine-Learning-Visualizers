import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

st.subheader("Support Vector Machine (SVM) Classifier")

# Sidebar options
kernel_type = st.sidebar.selectbox('Kernel type', ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
C_value = st.sidebar.slider('Regularization parameter (C)', 0.01, 10.0, 1.0, step=0.01)
degree_poly = st.sidebar.slider('Degree of the polynomial kernel', 1, 5, 3) if kernel_type == 'poly' else None
gamma_value = st.sidebar.slider('Kernel coefficient (gamma)', 0.01, 1.0, 0.1, step=0.01)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
num_features = st.sidebar.slider('Number of features (2 for visualization)', 2, 5, 2)
random_state = st.sidebar.slider('Random seed', 0, 99, 42)
test_size = st.sidebar.slider('Test size (percentage)', 10, 50, 30)

# Generate synthetic data
X, y = make_blobs(n_samples=num_samples, centers=2, n_features=num_features, random_state=random_state)
X = StandardScaler().fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=random_state)

# SVM model
model = SVC(kernel=kernel_type, C=C_value, degree=degree_poly if kernel_type == 'poly' else 3, gamma=gamma_value)
model.fit(X_train, y_train)

# Predictions and performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Visualization (only if 2 features for simplicity)
if num_features == 2:
    fig, ax = plt.subplots()
    # Plot decision boundary
    plot_decision_regions(X, y, clf=model, legend=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Regions")
    st.pyplot(fig)
