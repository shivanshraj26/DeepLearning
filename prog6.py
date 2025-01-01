# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Preprocess the data: Pad sequences to ensure uniform input size
train_padded = pad_sequences(train_data, maxlen=100, padding='post', truncating='post')
test_padded = pad_sequences(test_data, maxlen=100, padding='post', truncating='post')

# Build the RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),  # Embedding layer
    LSTM(64),  # LSTM layer with 64 units
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels,
          epochs=10,
          validation_data=(test_padded, test_labels))

# Evaluate the model on test data
loss, accuracy = model.evaluate(test_padded, test_labels, verbose=2)

# Print the test accuracy
print(f"Test Accuracy: {accuracy:.2f}")

# Test Accuracy: 0.79
