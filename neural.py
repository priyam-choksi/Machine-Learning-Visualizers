import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

st.title('Neural Network Visualizer')
st.markdown('''
This application demonstrates how a simple neural network can classify data that is not linearly separable. Using synthetic data, the app allows you to manipulate the neural network's architecture and training parameters to see how they affect the network's ability to create complex decision boundaries.
''')

# Sidebar for user input on model configuration
st.sidebar.header('Model Configuration')
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 500)
noise = st.sidebar.slider('Noise level', 0.01, 0.1, 0.05)
n_layers = st.sidebar.slider('Number of layers', 1, 5, 2)
n_neurons = st.sidebar.text_input('Number of neurons per layer (comma-separated)', '10, 10')
activation = st.sidebar.selectbox('Activation function', ['relu', 'tanh', 'sigmoid'], index=0)
learning_rate = st.sidebar.slider('Learning rate', 0.001, 0.1, 0.01, step=0.001)

# Explanation of neural network concepts
st.header("Neural Network Concepts")
st.markdown('''
### What is a Neural Network?
A neural network is a series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature.
''')

st.markdown('''
### How Does it Work?
Neural networks use an input layer, where each input neuron sends data to a predefined number of hidden layers of nodes. Each node represents a neuron, and the network's depth and the number of neurons in each layer can be adjusted to increase the model's complexity and ability to learn more complex patterns.
''')

# Mathematical explanation
st.subheader("Mathematical Basis of Neural Networks")
st.markdown('''
Neural networks operate using a system of weighted connections and biases adjusted through learning algorithms. The basic computational unit of a brain is a neuron, approximately implemented as a node in artificial neural networks. Each node takes input \(x\) and produces output \(y\), computed as:
''')
st.latex(r'''
y = f\left(\sum_{i=1}^{n} w_i \cdot x_i + b\right)
''')
st.markdown('''
where \(w_i\) are weights, \(x_i\) are inputs, \(b\) is a bias, and \(f\) is an activation function like sigmoid, tanh, or ReLU, which adds non-linearity to the model allowing it to learn more complex functions.
''')

# Generate and scale synthetic data
st.subheader("Data Preparation and Model Training")
X, y = make_circles(n_samples=num_samples, noise=noise, factor=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
model = Sequential()
neurons_per_layer = [int(x) for x in n_neurons.split(',')]
for neurons in neurons_per_layer:
    model.add(Dense(neurons, activation=activation))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model and plot training history
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, verbose=0)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('Loss Over Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('Accuracy Over Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend()
st.pyplot(fig)

# Display predictions and accuracy
predictions = (model.predict(X_test_scaled) > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.reshape(-1, 1))
st.subheader('Model Performance')
st.write(f'Test Accuracy: {accuracy:.2f}')
