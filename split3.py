import os
import shutil
import random
import sys
import helpers


def find_max_images_number(category_path):
    max_images_num = -1
    classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
    for class_dir in classes:
        class_path = os.path.join(category_path, class_dir)
        images_count = len(os.listdir(class_path))
        max_images_num = max(max_images_num, images_count)
    return max_images_num


def normalize_classes(categories_directory):
    categories = [category for category in os.listdir(categories_directory) if not category.startswith('.')]
    for category in categories:
        category_path = os.path.join(categories_directory, category)
        classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
        for class_dir in classes:
            class_path = os.path.join(category_path, class_dir)
            helpers.normalize_images_set(class_path)


def augment_and_equalize_classes(source_directory, target_directory, aug_count):
    categories = [category for category in os.listdir(source_directory) if not category.startswith('.')]
    for category in categories:
        category_path = os.path.join(source_directory, category)
        max_images_num = find_max_images_number(category_path)
        classes = [class_dir for class_dir in os.listdir(category_path) if not class_dir.startswith('.')]
        for class_dir in classes:
            class_path = os.path.join(category_path, class_dir)
            output_set_path = class_path.replace(source_directory, target_directory)
            os.makedirs(output_set_path)
            helpers.augment_and_equalize_images_set(class_path, output_set_path, aug_count, max_images_num)


def replace_val_set(val_path):
    classes = [class_dir for class_dir in os.listdir(val_path) if not class_dir.startswith('.')]
    for class_name in classes:
        val_class_path = os.path.join(val_path, class_name)
        val_class_images_num = len(os.listdir(val_class_path))
        train_class_path = val_class_path.replace("val", "train")
        train_set = os.listdir(train_class_path)
        shutil.rmtree(val_class_path)
        os.makedirs(val_class_path)
        new_val_set = random.sample(train_set, val_class_images_num)
        for image in new_val_set:
            shutil.copy(os.path.join(train_class_path, image), os.path.join(val_class_path, image))


source_directory = "splits/split1"
tmp_directory = "splits/tmp"
target_directory = "splits/split3"
if os.path.exists(tmp_directory):
    shutil.rmtree(tmp_directory)
if os.path.exists(target_directory):
    shutil.rmtree(target_directory)
shutil.copytree(source_directory, tmp_directory)

replace_val_set(os.path.join(tmp_directory, "val"))
aug_count = 5
augment_and_equalize_classes(source_directory, target_directory, aug_count)
normalize_classes(target_directory)
