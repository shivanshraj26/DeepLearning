# Import necessary libraries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load the Cats vs Dogs dataset
dataset_name = 'cats_vs_dogs'
(data_train, data_test), dataset_info = tfds.load(
    dataset_name,
    split=['train[:80%]', 'train[80%:]'],  # Split data into 80% train, 20% test
    as_supervised=True,                   # Include labels with images
    with_info=True                        # Include dataset metadata
)

# Image size for resizing
IMG_SIZE = 150

# Data preprocessing: resize and normalize
def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize to 150x150
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image, label

# Prepare train and test datasets
train_dataset = data_train.map(preprocess_image).shuffle(1000).batch(32).prefetch(1)
test_dataset = data_test.map(preprocess_image).batch(32).prefetch(1)

# Visualize a few samples from the training data
def plot_samples(dataset, n_samples=5):
    plt.figure(figsize=(12, 8))
    for i, (image, label) in enumerate(dataset.take(n_samples)):
        ax = plt.subplot(1, n_samples, i + 1)
        plt.imshow(image.numpy())
        plt.title('Cat' if label.numpy() == 0 else 'Dog')  # Label: 0 = Cat, 1 = Dog
        plt.axis('off')
    plt.show()

plot_samples(data_train.map(preprocess_image))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # First convolutional layer: 32 filters, 3x3 kernel, ReLU activation
    MaxPooling2D(2, 2),  # Max pooling with a 2x2 window
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer: 64 filters, 3x3 kernel, ReLU activation
    MaxPooling2D(2, 2),  # Max pooling with a 2x2 window
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer: 128 filters, 3x3 kernel, ReLU activation
    MaxPooling2D(2, 2),  # Max pooling with a 2x2 window
    Flatten(),  # Flatten feature maps into a 1D vector
    Dense(512, activation='relu'),  # Fully connected layer with 512 units and ReLU activation
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Output layer with 1 unit (binary classification) and sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer for efficient training
              loss='binary_crossentropy',  # Binary Cross-Entropy loss for binary class classification
              metrics=['accuracy'])  # Use accuracy as the evaluation metric

# Train the model
history = model.fit(
    train_dataset,  # Use training data for fitting
    validation_data=test_dataset,  # Validate performance using test data
    epochs=10  # Number of epochs
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2f}")

# Test Accuracy: 0.84
