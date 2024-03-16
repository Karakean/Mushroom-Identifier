import os
import shutil
import random
import sys
import helpers

def find_min_images_number(category_path):
    min_images = sys.maxsize
    classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
    for class_dir in classes:
        class_path = os.path.join(category_path, class_dir)
        images_count = len(os.listdir(class_path))
        min_images = min(min_images, images_count)
    return min_images


def equalize_classes_size(source_dir, target_dir):
    categories = [category for category in os.listdir(source_dir) if not category.startswith('.')]
    for category in categories:
        category_path = os.path.join(source_dir, category)
        min_images_number = find_min_images_number(category_path)
        classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
        for class_dir in classes:
            class_path = os.path.join(category_path, class_dir)
            images = [file for file in os.listdir(class_path) if not file.startswith('.')]
            random.shuffle(images)
            equalized_images = images[:min_images_number]
            target_class_dir = os.path.join(target_dir, category, class_dir)
            os.makedirs(target_class_dir)
            for image in equalized_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(target_class_dir, image))


def normalize_classes(categories_directory):
    categories = [category for category in os.listdir(categories_directory) if not category.startswith('.')]
    for category in categories:
        category_path = os.path.join(categories_directory, category)
        classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
        for class_dir in classes:
            class_path = os.path.join(category_path, class_dir)
            helpers.normalize_images_set(class_path)

def augment_classes(source_directory, target_directory):
    aug_count = 5
    categories = [category for category in os.listdir(source_directory) if not category.startswith('.')]
    for category in categories:
        category_path = os.path.join(source_directory, category)
        classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
        for class_dir in classes:
            class_path = os.path.join(category_path, class_dir)
            output_set_path = class_path.replace(source_directory, target_directory)
            os.makedirs(output_set_path)
            helpers.augment_images_set(class_path, output_set_path, aug_count)


source_directory = "splits/split1"
tmp_directory = "splits/tmp"
target_directory = "splits/split2"
if os.path.exists(tmp_directory):
    shutil.rmtree(tmp_directory)
if os.path.exists(target_directory):
    shutil.rmtree(target_directory)
equalize_classes_size(source_directory, tmp_directory)
normalize_classes(tmp_directory)
augment_classes(tmp_directory, target_directory)

