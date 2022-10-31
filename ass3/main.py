# Imports & Constants
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Question 1

# Question 2
# Load data (and specify dimensional arguments for reference later)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
in_shape = (32, 32, 3) # 32x32 image, with depth of 3 (RGB color channels)

# Build model
lenet_model = keras.Sequential([keras.layers.Input(shape=in_shape),                             # ""Input"" layer (assuming this doesn't count towards 7-layer count for LeNet?)
                                keras.layers.Conv2D(6, (5, 5), strides=1, activation="relu"),   # 6 kernels of 5x5, stride=1
                                keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),            # 2x2 kernels, stride=2
                                keras.layers.Conv2D(16, (5, 5), strides=1, activation="relu"),  # 16 kernels of 5x5, stride=1
                                keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),            # 2x2 kernels, stride=2
                                keras.layers.Conv2D(120, (5, 5), strides=1, activation="relu"), # 120 kernels of 5x5
                                keras.layers.Dense(84, activation="relu"),                      # 84 fully connected fprop
                                keras.layers.Dense(10, activation="softmax")])                  # 10 neurons for 10 classes, softmax for multiclass classification

# Train model
batch_size = 50
epochs = 25
lenet_model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate model (for funsies, since we already have test splits)
eval_score = model.evaluate(x_test, y_test)
print(f"  Loss  : {eval_score[0]}\nAccuracy: {eval_score[1]}")

# Question 3
