import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

st.title("Interactive Naive Bayes Classifier Exploration")

# Introduction
st.markdown("""
## Introduction to Naive Bayes Classifier
Naive Bayes classifiers are a family of straightforward, yet powerful, machine learning algorithms based on Bayes' Theorem with an assumption of independence among predictors. They are particularly effective for large datasets and applications like spam filtering, document classification, and sentiment analysis.
""")

# Bayes' Theorem
st.subheader("Bayes' Theorem")
st.markdown("""
The foundation of Naive Bayes is Bayes' Theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event.
""")
st.latex(r'''
P(H | E) = \frac{P(E | H) \times P(H)}{P(E)}
''')
st.markdown("""
Where:
- \( P(H | E) \) is the probability of hypothesis \( H \) given the evidence \( E \).
- \( P(E | H) \) is the probability of the evidence \( E \) given that hypothesis \( H \) is true.
- \( P(H) \) is the probability of the hypothesis \( H \) being true (before seeing the evidence).
- \( P(E) \) is the probability of the evidence (regardless of the hypothesis).
""")

# Configuration Sidebar
st.sidebar.header('Configuration')
num_features = st.sidebar.slider('Number of features (2 for visualization)', 2, 5, 2)
num_classes = st.sidebar.slider('Number of classes', 2, 5, 2)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)
random_state = st.sidebar.slider('Random seed', 0, 99, 42)
test_size = st.sidebar.slider('Test size (percentage)', 10, 50, 25)

# Data Preparation
st.subheader("Data Preparation")
st.markdown("""
The data for training the Naive Bayes model is synthetically generated to allow for clear visualization and understanding of how changes in parameters affect the model's predictions.
""")
X, y = make_classification(n_samples=num_samples, 
                           n_features=num_features, 
                           n_informative=num_features-1, 
                           n_redundant=1, 
                           n_repeated=0, 
                           n_classes=num_classes, 
                           n_clusters_per_class=1, 
                           random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

# Model Training and Visualization
st.subheader("Model Training and Testing")
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

if num_features == 2:
    fig, ax = plt.subplots()
    plot_decision_regions(X, y, clf=model, legend=2)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Naive Bayes Decision Regions")
    st.pyplot(fig)

# Probability Display
st.subheader("Model Interpretation")
if st.checkbox("Show class probabilities for test set"):
    proba = model.predict_proba(X_test)
    st.write("Class probabilities for the test set:")
    st.write(proba)

st.markdown("""
### Conclusion
Despite the simplicity of the Naive Bayes algorithm and the strong independence assumptions it makes, this classifier often performs well on large datasets and is surprisingly robust in many real-world scenarios.
""")
