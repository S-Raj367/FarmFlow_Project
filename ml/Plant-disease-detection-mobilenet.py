import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Configure variables for Transfer learning
image_size = 224
target_size = (image_size, image_size)
input_shape = (image_size, image_size, 3)

batch_size = 32
dataset_root = "/kaggle/input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

train_dir = os.path.join(dataset_root, "train")
test_dir = os.path.join(dataset_root, "valid")

def load_data(train_dir, test_dir, target_size, batch_size):
    """Load and preprocess the training and testing data."""
    train_aug = ImageDataGenerator(
        rescale=1/255.0,
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
    )

    test_aug = ImageDataGenerator(rescale=1/255.0)

    train_data = train_aug.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_data = test_aug.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_data, test_data

def build_model(input_shape, num_classes):
    """Build the MobileNetV2 model for transfer learning."""
    mbnet_v2 = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    mbnet_v2.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = mbnet_v2(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, train_data, test_data, epochs=30):
    """Train the model with early stopping and learning rate reduction."""
    early_stopping_cb = callbacks.EarlyStopping(monitor="val_loss", patience=5)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.3)

    history = model.fit(
        train_data,
        epochs=epochs,
        steps_per_epoch=1000,  # Adjust based on dataset size
        validation_data=test_data,
        callbacks=[early_stopping_cb, reduce_lr]
    )
    return history

def plot_training_results(history):
    """Plot the training and validation accuracy/loss."""
    hist = history.history

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs #")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs #")
    plt.legend()

    plt.show()

def save_model_and_categories(model, train_data, model_name="data/plant_disease_detection.h5", categories_name="categories.json"):
    """Save the trained model and categories."""
    model.save(model_name)
    print(f"Model saved as {model_name}")

    with open(categories_name, "w") as file:
        json.dump(train_data.class_indices, file)
    print(f"Categories saved as {categories_name}")

def main():
    # Load the data
    train_data, test_data = load_data(train_dir, test_dir, target_size, batch_size)
    print("Classes:", train_data.class_indices)

    # Build the model
    num_classes = len(train_data.class_indices)
    model = build_model(input_shape, num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train the model
    history = train_model(model, train_data, test_data, epochs=30)

    # Plot training results
    plot_training_results(history)

    # Save the model and categories
    save_model_and_categories(model, train_data)

if __name__ == "__main__":
    main()