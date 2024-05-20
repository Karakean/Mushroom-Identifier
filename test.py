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


def find_hard_examples(model, generator, threshold=0.75):
    hard_examples = []
    steps = generator.samples // generator.batch_size
    if generator.samples % generator.batch_size != 0:
        steps += 1
    for i in range(steps):
        images, labels = generator[i]
        predictions = model.predict(images)
        print(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        for j in range(len(images)):
            if predicted_classes[j] != true_classes[j] or np.max(predictions[j]) < threshold:
                hard_examples.append((images[j], labels[j]))
        print(f"Processed {i + 1}/{steps} batches")

    return hard_examples


def retrain_model_on_hard_examples(model, hard_examples, batch_size=32, epochs=5):
    if not hard_examples:
        return model, False
    hard_images, hard_labels = zip(*hard_examples)
    hard_images = np.array(hard_images)
    hard_labels = np.array(hard_labels)
    model.fit(hard_images, hard_labels, batch_size=batch_size, epochs=epochs)
    return model, True


def make_retraining_plots(loss_before, accuracy_before, loss_after, accuracy_after):
    fig, ax1 = plt.subplots()
    ax1.bar(['Before Retraining', 'After Retraining'], [accuracy_before, accuracy_after], color=['blue', 'orange'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Before and After Hard Mining and Retraining')
    plt.show()
    fig, ax2 = plt.subplots()
    ax2.bar(['Before Retraining', 'After Retraining'], [loss_before, loss_after], color=['blue', 'orange'])
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss Before and After Hard Mining and Retraining')
    plt.show()


def hard_example_mining_and_retraining(model_name, data_dir, image_size):
    train_generator, _, test_generator = load_data(data_dir, image_size)
    model = tf.keras.models.load_model(model_name)
    loss_before, accuracy_before = model.evaluate(test_generator)
    threshold = 0.8
    hard_examples = find_hard_examples(model, train_generator, threshold)
    model, is_retrained = retrain_model_on_hard_examples(model, hard_examples)
    if not is_retrained:
        print("No hard examples found.")
        return
    loss_after, accuracy_after = model.evaluate(test_generator)
    if accuracy_after > accuracy_before and loss_after < loss_before:
        model.save(f"retrained_{model_name}")
    make_retraining_plots(loss_before, accuracy_before, loss_after, accuracy_after)


def main():
    splits_dir = 'splits'
    splits = ['split1', 'split2', 'split3']
    split2_path = "splits/split2"
    image_size = 224
    source_dir = "dataset"
    genuses = [genus for genus in os.listdir(source_dir) if not genus.startswith('.')]
    _, _, generator = load_data(split2_path, image_size, shuffle=False)
    chosen_model = "MODEL2.keras"
    ensemble_models_set = ["MODEL1.keras", "MODEL2.keras", "MODEL3.keras"]

    # test_basic_models(splits_dir, splits, image_size, genuses)
    # test_ensemble_models(ensemble_models_set, genuses, generator)
    # test_model(chosen_model, genuses, generator)
    hard_example_mining_and_retraining(chosen_model, split2_path, image_size)


if __name__ == "__main__":
    main()
