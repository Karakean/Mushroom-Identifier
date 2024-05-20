import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint

from helpers import load_data


def build_model(image_size, classes_number):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   activation='relu', input_shape=(image_size, image_size, 3)))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
    # cnn.add(tf.keras.layers.Dropout(0.5))
    cnn.add(tf.keras.layers.Dense(units=classes_number, activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'f1_score', 'precision_score',
    #                                                                         'recall_score'])

    return cnn


def make_plots(results, name):
    filename = name.replace(" ", "_")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results.history['accuracy'], label='Train Accuracy')
    plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"Accuracy with {name}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(f"{filename}_accuracy.png")

    plt.subplot(1, 2, 2)
    plt.plot(results.history['loss'], label='Train Loss')
    plt.plot(results.history['val_loss'], label='Validation Loss')
    plt.title(f"Loss with {name}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig(f"{filename}_loss.png")
    plt.show()


def train_on_splits():
    base_dir = 'splits'
    splits = ['split1', 'split2', 'split3']
    image_size = 224
    classes_number = 4
    epochs_number = 30
    for i, split in enumerate(splits):
        print(f"Training on {split}...")
        data_dir = os.path.join(base_dir, split)
        train_generator, validation_generator, test_generator = load_data(data_dir, image_size)
        model = build_model(image_size, classes_number)

        model_name = f"MODEL{i + 1}.keras"
        key_metric = 'val_accuracy'
        checkpoint = ModelCheckpoint(model_name, monitor=key_metric, save_best_only=True, mode='max')
        # checkpoint_accuracy = ModelCheckpoint('model_best_accuracy.keras', monitor='accuracy', save_best_only=True,
        #                                       mode='max')
        # checkpoint_recall = ModelCheckpoint('model_best_recall.keras', monitor='recall', save_best_only=True, mode='max')
        # checkpoint_precision = ModelCheckpoint('model_best_precision.keras', monitor='precision', save_best_only=True,
        #                                        mode='max')
        earlystop = tf.keras.callbacks.EarlyStopping(monitor=key_metric, patience=5)
        result = model.fit(train_generator, validation_data=validation_generator, epochs=epochs_number,
                           callbacks=[checkpoint, earlystop])
        # result = model.fit(train_generator, validation_data=validation_generator, epochs=epochs_number,
        #                    callbacks=[checkpoint_accuracy, checkpoint_recall, checkpoint_precision, earlystop])

        make_plots(result, f"MODEL{i + 1} created from {split}")

        model.save(f"MODEL{i + 1}.keras")


def main():
    train_on_splits()


if __name__ == "__main__":
    main()
