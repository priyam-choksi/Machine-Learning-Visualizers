import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.subheader("Random Forest Classifier")

# Sidebar options for configuration
num_trees = st.sidebar.slider('Number of trees in the forest', 1, 100, 10)
max_depth = st.sidebar.slider('Maximum depth of trees', 1, 20, 5)
num_features = st.sidebar.slider('Number of features', 2, 10, 5)
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200)

# Generate synthetic data for classification
X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest model
forest = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, random_state=42)
forest.fit(X_train, y_train)

# Predictions and performance
y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization of confusion matrix
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Feature importance visualization
fig2, ax2 = plt.subplots()
importances = forest.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
ax2.bar(range(X.shape[1]), importances[sorted_indices])
ax2.set_xticks(range(X.shape[1]))
ax2.set_xticklabels(sorted_indices)
ax2.set_title('Feature Importance')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Importance')
st.pyplot(fig2)

# Display performance metrics
st.write(f"Accuracy of the model: {accuracy:.2f}")
