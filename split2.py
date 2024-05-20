import os
import shutil
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


source_directory = "splits/split1"
target_directory = "splits/split2"
if os.path.exists(target_directory):
    shutil.rmtree(target_directory)

aug_count = 5
augment_and_equalize_classes(source_directory, target_directory, aug_count)
normalize_classes(target_directory)
