# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

# Load the Reuters dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Function to vectorize sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Create a zero matrix of shape (num_samples, dimension)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Set index positions corresponding to words in the sequence to 1
    return results

# Vectorize the training and testing data
x_train = vectorize_sequences(train_data)  # Vectorized training data
x_test = vectorize_sequences(test_data)    # Vectorized testing data

# One-hot encode labels using Keras' built-in function
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Build a simple feedforward neural network
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10000,)),  # Input layer: Dense layer with 64 units, ReLU activation, input shape matching the vectorized data
    layers.Dense(64, activation='relu'),  # Hidden layer: Another Dense layer with 64 units and ReLU activation
    layers.Dense(46, activation='softmax')  # Output layer: Dense layer with 46 units (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='rmsprop',  # Optimizer: RMSprop, a gradient descent optimization algorithm
              loss='categorical_crossentropy',  # Cross-entropy loss for multi-class classification
              metrics=['accuracy'])  # Use accuracy as the evaluation metric

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print the results
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Test Loss: 0.9769346714019775, Test Accuracy: 0.7809438705444336
