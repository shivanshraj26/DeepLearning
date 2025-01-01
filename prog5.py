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
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (sigmoid activation)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot training and validation accuracy/loss over epochs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Plot accuracy
plt.figure(figsize=(12, 8))
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot loss
plt.figure(figsize=(12, 8))
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
