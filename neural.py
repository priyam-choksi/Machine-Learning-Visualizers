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
st.write('This app demonstrates a simple neural network using synthetic data.')

# Sidebar options
num_samples = st.sidebar.slider('Number of samples', 100, 1000, 500)
noise = st.sidebar.slider('Noise level', 0.01, 0.1, 0.05)
n_layers = st.sidebar.slider('Number of layers', 1, 5, 2)
n_neurons = st.sidebar.text_input('Number of neurons per layer (comma-separated)', '10, 10')
activation = st.sidebar.selectbox('Activation function', ['relu', 'tanh', 'sigmoid'], index=0)
learning_rate = st.sidebar.slider('Learning rate', 0.001, 0.1, 0.01, step=0.001)

# Generate synthetic data
X, y = make_circles(n_samples=num_samples, noise=noise, factor=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = Sequential()
neurons_per_layer = [int(x) for x in n_neurons.split(',')]
for neurons in neurons_per_layer:
    model.add(Dense(neurons, activation=activation))
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, verbose=0)

# Plotting the training history
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

# Prediction results
predictions = (model.predict(X_test_scaled) > 0.5).astype(int)
accuracy = np.mean(predictions == y_test.reshape(-1, 1))
st.write(f'Test Accuracy: {accuracy:.2f}')
