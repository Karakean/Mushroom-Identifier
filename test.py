import os

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import numpy as np

from helpers import load_data


def main():
    splits_dir = 'splits'
    splits = ['split1', 'split2', 'split3']
    image_size = 224
    source_dir = "dataset"
    genuses = [genus for genus in os.listdir(source_dir) if not genus.startswith('.')]

    for i, split in enumerate(splits):
        loaded_model = tf.keras.models.load_model(f"MODEL{i + 1}.keras")
        data_dir = os.path.join(splits_dir, split)
        train_generator, validation_generator, test_generator = load_data(data_dir, image_size)

        test_steps = test_generator.samples // test_generator.batch_size
        if test_generator.samples % test_generator.batch_size != 0:
            test_steps += 1

        loss, accuracy = loaded_model.evaluate(test_generator)
        print(f"Test Accuracy for MODEL{i + 1} ({split}): {accuracy}")
        print(f"Test Loss for MODEL{i + 1} ({split}): {loss}")
        predictions = loaded_model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        print(f"Confusion Matrix for MODEL{i + 1} ({split}):")
        print(confusion_matrix(validation_generator.classes, predicted_classes))
        print(f"\nClassification Report for MODEL{i + 1} ({split}):")
        print(classification_report(validation_generator.classes, predicted_classes, target_names=genuses))


if __name__ == "__main__":
    main()
