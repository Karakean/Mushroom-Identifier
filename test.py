import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from helpers import load_data


def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_classification_report(cr, classes, title):
    report_df = pd.DataFrame(cr).transpose()
    report_df = report_df.loc[classes]
    report_df.drop(['support'], axis=1, inplace=True)
    report_df.plot(kind='bar', figsize=(12, 8))
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.ylim(0, 1)
    plt.show()


def plot_metrics(metrics, splits, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.bar(splits, metrics, color='skyblue')
    plt.title(title)
    plt.xlabel('Model Splits')
    plt.ylabel(ylabel)
    plt.xticks(ticks=range(len(splits)), labels=splits)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def test_model(model_name, genuses, generator, additional_text=" for Test Set"):
    loaded_model = tf.keras.models.load_model(model_name)
    loss, accuracy = loaded_model.evaluate(generator)
    predictions = loaded_model.predict(generator)
    predicted_classes = np.argmax(predictions, axis=1)
    cm = confusion_matrix(generator.classes, predicted_classes)
    cr = classification_report(generator.classes, predicted_classes, target_names=genuses, output_dict=True)
    plot_confusion_matrix(cm, genuses, f"Confusion Matrix for {model_name} {additional_text}")
    plot_classification_report(cr, genuses, f"Classification Report for {model_name} {additional_text}")
    return loss, accuracy


def test_basic_models(splits_dir, splits, image_size, genuses):
    accuracies = [[], [], []]
    losses = [[], [], []]
    generator_labels = ['Training', 'Validation', 'Test']
    for i, split in enumerate(splits):
        data_dir = os.path.join(splits_dir, split)
        generators = load_data(data_dir, image_size, shuffle=False)
        for j, generator in enumerate(generators):
            loss, accuracy = test_model(f"MODEL{i + 1}.keras", genuses, generators[j],
                                        f" for {generator_labels[j]} Set")
            losses[j].append(loss)
            accuracies[j].append(accuracy)

    for i, label in enumerate(generator_labels):
        plot_metrics(accuracies[i], splits, f'Testing Accuracy on {label} Set Across Models', 'Accuracy')
        plot_metrics(losses[i], splits, f'Testing Loss on {label} Set Across Models', 'Loss')


def test_ensemble_models(models, genuses, generator):
    predictions = ensemble_predictions(models, generator)
    predicted_classes = np.argmax(predictions, axis=1)
    cm = confusion_matrix(generator.classes, predicted_classes)
    cr = classification_report(generator.classes, predicted_classes, target_names=genuses, output_dict=True)
    plot_confusion_matrix(cm, genuses, f"Confusion Matrix for model ensemble")
    plot_classification_report(cr, genuses, f"Classification Report for model ensemble")


def ensemble_predictions(models, generator):
    total_predictions = None
    for model in models:
        loaded_model = tf.keras.models.load_model(model)
        predictions = loaded_model.predict(generator, verbose=0)
        if total_predictions is None:
            total_predictions = predictions
        else:
            total_predictions += predictions
    average_predictions = total_predictions / len(models)
    return average_predictions


def main():
    splits_dir = 'splits'
    splits = ['split1', 'split2', 'split3']
    image_size = 224
    source_dir = "dataset"
    genuses = [genus for genus in os.listdir(source_dir) if not genus.startswith('.')]
    test_basic_models(splits_dir, splits, image_size, genuses)
    _, _, generator = load_data("splits/split2", image_size, shuffle=False)
    model_ensemble_set = ["MODEL1.keras", "MODEL2.keras", "MODEL3.keras"]
    test_ensemble_models(model_ensemble_set, genuses, generator)
    # test_model("test_model.keras", genuses, generator)


if __name__ == "__main__":
    main()
