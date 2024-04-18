### Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

### Get current directory
current_dir = os.getcwdir()

### Define data path (assuming data is stored in a .npz file in the current directory)
data_path = os.path.join(current_dir, ".npz")

### Load data using TensorFlow's built-in function (modify if data format is different)
(training_images, training_labels), _ = tf.keras.datasets.load_data(path=data_path)

### Function to reshape and normalize images
def reshape_and_normalize(images):
    """
    Reshapes images to a format suitable for the convolutional neural network (CNN).
    Also normalizes pixel values between 0 and 1.

    Args:
        images: A NumPy array containing the images.

    Returns:
        A NumPy array containing the reshaped and normalized images.
    """

    ### Reshape to (number of images, 28, 28, 1) for grayscale images
    images = images.reshape(-1, 28, 28, 1) 

    ### Normalize pixel values by dividing by 255.0
    images = images / 255.0

    return images

### Reshape and normalize training images
(training_images, _), _ = tf.keras.datasets.load_data(path=data_path)
training_images = reshape_and_normalize(training_images)

### Print some information about the processed data
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")


### Custom callback to stop training early if accuracy reaches 80%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=[]):
        if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.8:
            print("\nReached 80% accuracy so cancelling training!")
            self.model.stop_training = True


### Define and train a convolutional neural network model
def convolutional_model():
    """
    Creates a simple convolutional neural network model for image classification.

    Returns:
        A compiled TensorFlow Keras model.
    """

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # First convolutional layer
        tf.keras.layers.MaxPooling2D(2, 2),  # Max pooling layer
        tf.keras.layers.Flatten(),  # Flatten layer to prepare for dense layers
        tf.keras.layers.Dense(128, activation='relu'),  # First dense layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units for 10 class probabilities
    ])

    model.summary()  # Print model summary

    model.compile(optimizer='adam',  # Optimizer
                  loss='sparse_categorical_crossentropy',  # Loss function
                  metrics=['accuracy'])  # Metrics to track during training

    return model


model = convolutional_model()

### Ensure model parameter count is under 1 million for grading purposes
model_params = model.count_params()
assert model_params < 1000000, (
    f'Your model has {model_params:,} params. For successful grading, please keep it '
    f'under 1,000,000 by reducing the number of units in your Conv2D and/or Dense layers.'
)

callbacks = myCallback()  # Create callback instance

### Train the model with early stopping
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

### Print the number of epochs the model trained for
print(f"Your model was trained for {len(history.epoch)} epochs")
