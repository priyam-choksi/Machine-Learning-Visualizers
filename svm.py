import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

st.title("Support Vector Machine (SVM) Classifier Explorer")
st.markdown(r"""
This interactive application demonstrates how a Support Vector Machine (SVM) classifier works using synthetic data. SVM is a powerful and versatile machine learning model, capable of performing linear and nonlinear classification, regression, and even outlier detection.
""")  # Used a raw string here for safety

# Sidebar options for configuration
st.sidebar.header('Model Configuration')
kernel_type = st.sidebar.selectbox('Kernel type', ['linear', 'poly', 'rbf', 'sigmoid'], index=2)
C_value = st.sidebar.slider('Regularization parameter (C)', 0.01, 10.0, 1.0, step=0.01)
degree_poly = st.sidebar.slider('Degree of the polynomial kernel', 1, 5, 3) if kernel_type == 'poly' else None
gamma_value = st.sidebar.slider('Kernel coefficient (gamma)', 0.01, 1.0, 0.1, step=0.01)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
num_features = st.sidebar.slider('Number of features (2 for visualization)', 2, 5, 2)
random_state = st.sidebar.slider('Random seed', 0, 99, 42)
test_size = st.sidebar.slider('Test size (percentage)', 10, 50, 30)

# Explanation of SVM
st.header("Understanding SVM")
st.markdown(r"""
### What is a Support Vector Machine?
SVM is a type of supervised machine learning algorithm that can be used for classification or regression problems. It works by finding the hyperplane that best divides a dataset into classes.
""")

st.markdown(r"""
### How Does SVM Work?
- **Linear SVM**: Linear SVM tries to find the best margin (distance between the line and the support vectors) that separates the classes. This margin is maximized to help the model generalize well to new data.
- **Nonlinear SVM**: For nonlinear data, SVM uses what is called the kernel trick to transform data into a higher dimension where a linear separator could be used. The kernel defines how to calculate the distance between data points in this new feature space.
""")

# Mathematical basis
st.subheader("Mathematical Formulation")
st.markdown(r"""
The SVM classifier works by solving the following optimization problem:
""")
st.latex(r'''
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i
''')

st.markdown(r"""
Where:
- $\mathbf{w}$ is the weight vector of the hyperplane.
- $b$ is the bias.
- $\xi_i$ are the slack variables, $C$ is the regularization parameter.
""")

st.markdown(r"""
### Kernel Trick
For non-linear boundaries, SVM uses kernels to map input features into high-dimensional feature spaces where the data points might be more easily separable by a hyperplane.
""")
st.latex(r'''
K(x, x') = \exp(-\gamma \|x - x'\|^2)
''')
st.markdown(r"""
For instance, the RBF kernel (Radial Basis Function) is used to transform data points and find a non-linear decision boundary.
""")

# Generate and visualize synthetic data
st.subheader("Data Generation and Model Training")
X, y = make_blobs(n_samples=num_samples, centers=2, n_features=num_features, random_state=random_state)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=random_state)

# Train SVM model
model = SVC(kernel=kernel_type, C=C_value, degree=degree_poly if kernel_type == 'poly' else 3, gamma=gamma_value)
model.fit(X_train, y_train)

# Performance and visualization
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

if num_features == 2:
    st.subheader("Decision Boundary Visualization")
    fig, ax = plt.subplots()
    plot_decision_regions(X, y, clf=model, legend=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Regions")
    st.pyplot(fig)

st.write("The plot above shows the decision boundaries created by the SVM, demonstrating how different kernel settings and regularization parameters impact the model's performance.")
