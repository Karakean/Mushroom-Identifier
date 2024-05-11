import os

import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt

def main():
    splits = ['split3']
    image_size = 224
    classes_number = 4
    model_path = os.path.join("modele", "MODEL3.keras")
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    splits_dir = "splits/split1"

    # Load the test data
    test_data = test_datagen.flow_from_directory(
        os.path.join(splits_dir, "test"),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_data, batch_size=128)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Generate a bar plot for the test loss and accuracy
    plt.bar(['Test Loss', 'Test Accuracy'], [test_loss, test_accuracy])
    plt.title('Test Loss and Accuracy')
    plt.savefig(f"TEST_VALUES.png")
    # plt.show()


if __name__ == "__main__":
    main()

