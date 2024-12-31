import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load the IMDb dataset
# Only consider the 10,000 most frequent words
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure uniform input length
max_length = 200  # Maximum review length (truncate or pad shorter reviews)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_length),  # Embedding layer
    LSTM(64, return_sequences=False),  # LSTM layer with 64 units
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),  # Fully connected layer with ReLU activation
    Dropout(0.5),  # Another dropout layer
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam',                # Adam optimizer
              loss='binary_crossentropy',      # Binary cross-entropy loss
              metrics=['accuracy'])            # Track accuracy during training and testing

# Train the model
history = model.fit(x_train, y_train,
                    epochs=5,                 # Number of epochs
                    batch_size=64,             # Batch size for gradient updates
                    validation_split=0.2)      # Use 20% of training data for validation

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Test accuracy: 0.84
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
