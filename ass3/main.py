# Imports & Constants
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Question 1
# Load data (and specify dimensional arguments for reference later)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
in_shape = (28, 28, 1) # 28x28x1 image

x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)
print(f"Expected: {in_shape}\n Actual : {x_train.shape}")


# Build models
conv_model_increasing = keras.Sequential([keras.layers.Input(shape=in_shape),
                                          keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=96, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=384, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=1024, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=1536, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Flatten(),
                                          keras.layers.Dense(10, activation="softmax")])
conv_model_increasing.summary()

conv_model_decreasing = keras.Sequential([keras.layers.Input(shape=in_shape),
                                          keras.layers.Conv2D(filters=1536, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=1024, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=512, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=384, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=96, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Flatten(),
                                          keras.layers.Dense(10, activation="softmax")])
conv_model_decreasing.summary()

vae_model = keras.Sequential([keras.layers.Input(shape=in_shape),
                                          keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=96, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Conv2D(filters=96, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
                                          keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
                                          keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
                                          keras.layers.Flatten(),
                                          keras.layers.Dense(10, activation="softmax")])
vae_model.summary()

# Train models
vae_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
vae_model.fit(x_train, y_train, epochs=1, validation_split=0.1)

# Evaluate models
eval_score = vae_model.evaluate(x_test, y_test, verbose=0)
print(f"    Loss: {eval_score[0]}\nAccuracy: {eval_score[1]}")



# Question 2
# Load data (and specify dimensional arguments for reference later)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
in_shape = (32, 32, 3) # 32x32 image, with depth of 3 (RGB color channels)
print(f"Expected: {in_shape}\n Actual : {x_train.shape}")
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)

# Build model
lenet_model = keras.Sequential([keras.layers.Input(shape=in_shape),                                                 # Keras input object
                                keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation="relu"),   # 6 kernels of 5x5, stride=1
                                keras.layers.MaxPooling2D(pool_size=2, strides=2),                                # 2x2 kernels, stride=2
                                keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation="relu"),  # 16 kernels of 5x5, stride=1
                                keras.layers.MaxPooling2D(pool_size=2, strides=2),                                # 2x2 kernels, stride=2
                                keras.layers.Conv2D(filters=120, kernel_size=5, strides=1, activation="relu"), # 120 kernels of 5x5
                                keras.layers.Flatten(),                                                             # Also assuming the flattening layer doesn't count towards 7-layer count for LeNet, since it's not trainable anyways
                                keras.layers.Dense(84, activation="relu"),                                          # 84 fully connected fprop
                                keras.layers.Dense(10, activation="softmax")])                                      # 10 neurons for 10 classes, softmax for multiclass classification

lenet_model.summary()
# Train model
lenet_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lenet_model.fit(x_train, y_train, epochs=25, validation_split=0.1)

# Evaluate model (for funsies, since we already have test splits)
eval_score = lenet_model.evaluate(x_test, y_test, verbose=0)
print(f"    Loss: {eval_score[0]}\nAccuracy: {eval_score[1]}")

# Question 3
