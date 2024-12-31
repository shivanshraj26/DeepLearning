# Import necessary libraries
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

# Load the Reuters dataset, limiting to the 10,000 most frequently occurring words
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Retrieve the mapping of words to their index and reverse it for decoding purposes
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode the first newswire for inspection
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# Note: Indices 0, 1, and 2 are reserved for padding, start of sequence, and unknown words.

# Function to vectorize sequences into a binary matrix representation
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Initialize a zero matrix
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Set the corresponding indices for each word to 1
    return results

# Vectorize the train and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Function to one-hot encode labels into binary class matrices
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))  # Initialize a zero matrix
    for i, label in enumerate(labels):
        results[i, label] = 1  # Set the index corresponding to the label to 1
    return results

# One-hot encode train and test labels (manual implementation)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# Alternatively, use the built-in to_categorical method for one-hot encoding
from tensorflow.keras.utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu'),  # First hidden layer with 64 units and ReLU activation
    layers.Dense(64, activation='relu'),  # Second hidden layer with 64 units and ReLU activation
    layers.Dense(46, activation='softmax')  # Output layer with 46 units (one per class) and softmax activation
])

# Compile the model with an optimizer, loss function, and metric
model.compile(optimizer='rmsprop',               # RMSprop optimizer for better handling of sparse gradients
              loss='categorical_crossentropy',  # Cross-entropy loss for multi-class classification
              metrics=['accuracy'])            # Accuracy as the evaluation metric

# Prepare a validation set (first 1000 samples) and training set (remaining samples)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Train the model
history = model.fit(partial_x_train,        # Training features
                    partial_y_train,        # Training labels
                    epochs=25,              # Number of epochs
                    batch_size=512,         # Batch size for gradient updates
                    validation_data=(x_val, y_val))  # Validation set to monitor performance

# Evaluate the model on the test set
results = model.evaluate(x_test, one_hot_test_labels)

# Print the results
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Test Loss: 1.111000657081604, Test Accuracy: 0.7804986834526062
