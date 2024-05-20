import os
import random
import shutil


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


source_directory = "splits/split2"
target_directory = "splits/split3"
if os.path.exists(target_directory):
    shutil.rmtree(target_directory)
shutil.copytree(source_directory, target_directory)
replace_val_set(os.path.join(target_directory, "val"))

