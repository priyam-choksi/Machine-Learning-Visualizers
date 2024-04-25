import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

st.subheader("Naive Bayes Classifier")

# Sidebar options
num_features = st.sidebar.slider('Number of features (2 for visualization)', 2, 5, 2)
num_classes = st.sidebar.slider('Number of classes', 2, 5, 2)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
random_state = st.sidebar.slider('Random seed', 0, 99, 42)
test_size = st.sidebar.slider('Test size (percentage)', 10, 50, 25)

# Generate synthetic data
X, y = make_classification(n_samples=num_samples, 
                           n_features=num_features, 
                           n_informative=num_features-1,  # Leave room for at least one redundant or repeated feature
                           n_redundant=1,  # Adjust this based on your requirements
                           n_repeated=0,  # You can set this as needed
                           n_classes=num_classes, 
                           n_clusters_per_class=1, 
                           random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

# Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions and accuracy
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
    plt.title("Naive Bayes Decision Regions")
    st.pyplot(fig)

# Display probabilities
if st.checkbox("Show class probabilities for test set"):
    proba = model.predict_proba(X_test)
    st.write("Class probabilities for the test set:")
    st.write(proba)
