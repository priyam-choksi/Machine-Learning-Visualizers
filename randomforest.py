import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.title("Random Forest Classifier Visualizer")
st.write("""
This application demonstrates the Random Forest classification process using a synthetic dataset. Random Forests build multiple decision trees and merge them together to get a more accurate and stable prediction.
""")

# Sidebar options for configuration
st.sidebar.header('Model Configuration')
num_trees = st.sidebar.slider('Number of trees in the forest', 1, 100, 10, help="The number of trees in the forest.")
max_depth = st.sidebar.slider('Maximum depth of trees', 1, 20, 5, help="The maximum depth of each tree.")
num_features = st.sidebar.slider('Number of features', 2, 10, 5, help="The total number of features in the dataset.")
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 200, help="The number of samples in the dataset.")

# Explanation of Random Forest
st.header("Understanding Random Forests")
st.write("""
### What is a Random Forest?
A Random Forest is an ensemble learning method for classification (and regression) that operates by constructing multiple decision trees during training. The decision of the majority of trees is chosen by the Random Forest as the final decision.
""")

# Mathematical basis
st.write("""
### How Random Forest Works:
- **Bootstrap aggregating (Bagging)**: Each decision tree in the forest is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.
- **Feature Randomness**: When splitting a node during the construction of a tree, the split that is chosen is no longer the best split among all features. Instead, the split that is best among a random subset of features is chosen. This results in a forest that is more diverse, leading to lower correlation among individual predictions and higher robustness overall.
""")

st.write("""
### Mathematical Formulation:
At each node in the decision trees, a split is selected that maximizes the gain in purity, often measured by Gini impurity or entropy in classification problems. The formula for Gini impurity for a set of items with \( J \) classes is:
""")

# Detailed explanation of Gini Impurity
st.header("Gini Impurity Explained")
st.markdown("""
The Gini impurity is a measure commonly used in decision trees, including those that comprise a Random Forest, to quantify how often a randomly chosen element would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. It plays a crucial role in the decision-making process of building decision trees, which are the foundational components of a Random Forest.
""")

st.subheader("Concept of Gini Impurity")
st.markdown("""
Gini impurity measures the disorder or purity of a set of elements. In the context of decision trees, which are used in Random Forests, each node of the tree splits the data into subsets. The Gini impurity helps to evaluate how mixed these subsets are, which in turn helps to decide where to make the splits. The goal is to create subsets that are as pure as possible, meaning they ideally consist of data points that belong to a single class.
""")

st.subheader("How Gini Impurity is Used in Decision Trees")
st.markdown("""
When building a decision tree, Gini impurity is used to quantify how 'impure' a node is—a node being a point where a decision split occurs in the tree. Here's the formula for Gini impurity:
""")
st.latex(r"\text{Gini} = 1 - \sum_{i=1}^{n} p_i^2")
st.markdown("""
Where \( p_i \) is the probability of an object being classified to a particular class.
""")

st.subheader("Calculation Example")
st.markdown("""
Imagine a binary classification at a node containing 10 items: 6 of one class and 4 of another. The Gini impurity would be calculated as follows:
""")
st.latex(r"\text{Gini} = 1 - \left( \left(\frac{6}{10}\right)^2 + \left(\frac{4}{10}\right)^2 \right) = 1 - (0.36 + 0.16) = 0.48")
st.markdown("""
The Gini impurity ranges from 0 (perfect purity, all elements in a node belong to a single class) to 0.5 (maximum impurity in a binary classification, where a node contains an equal number of elements from both classes).
""")

st.subheader("Application in Random Forest")
st.markdown("""
During the training of a Random Forest, each decision tree is built by selecting the best splits at each node based on the decrease in Gini impurity resulting from the split. Though alternative measures like entropy can also be used, Gini impurity is favored for its lower computational overhead, as it doesn’t require computing logarithmic functions.
""")

st.subheader("Overall Role in Random Forest")
st.markdown("""
In a Random Forest, the reduction of Gini impurity across all trees helps ensure that the ensemble model (the forest) is built from trees that are both individually accurate and diverse. This diversity and accuracy help make Random Forest a robust model against overfitting and effective for various classification tasks.
""")

st.latex(r'''
Gini = 1 - \sum_{j=1}^{J} p_j^2
''')
st.write("""
where \( p_j \) is the proportion of the class \( j \) among the samples in the set.
""")

# Generate synthetic data for classification
st.subheader("Data Generation and Training")
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
st.subheader("Confusion Matrix")
st.write("The confusion matrix visualizes the accuracy of a classifier by comparing the actual labels with the predicted labels.")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Feature importance visualization
st.subheader("Feature Importance")
st.write("This graph shows the importance of each feature in making predictions, as determined by the Random Forest.")
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
