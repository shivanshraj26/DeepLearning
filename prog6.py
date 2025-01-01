# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb

# Load the IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Preprocess the data: Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=100, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=100, padding='post', truncating='post')

# Build the RNN model
model = Sequential([
    Embedding(input_dim=10000, output_dim=32, input_length=100),  # Embedding layer
    LSTM(64),  # LSTM layer with 64 units
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

# Print the test accuracy
print(f"Test Accuracy: {accuracy:.2f}")

# Test Accuracy: 0.79
