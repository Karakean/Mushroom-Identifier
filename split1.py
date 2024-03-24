import os
import shutil
import random

source_dir = "small_dataset"
destination_dir = "splits/split1"
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

genuses = [genus for genus in os.listdir(source_dir) if not genus.startswith('.')]

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for genus in genuses:
    source_genus_dir = os.path.join(source_dir, genus)
    destination_train_dir = os.path.join(destination_dir, "train", genus)
    destination_val_dir = os.path.join(destination_dir, "val", genus)
    destination_test_dir = os.path.join(destination_dir, "test", genus)

    for directory in [destination_train_dir, destination_val_dir, destination_test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    files = os.listdir(source_genus_dir)
    random.shuffle(files)

    num_files = len(files)
    num_val = int(num_files * val_ratio)
    num_test = int(num_files * test_ratio)
    num_train = num_files - num_val - num_test

    train_files = files[:num_train]
    val_files = files[num_train:num_train + num_val]
    test_files = files[num_train + num_val:]

    for file in train_files:
        source_file = os.path.join(source_genus_dir, file)
        destination_file = os.path.join(destination_train_dir, file)
        shutil.copy(source_file, destination_file)

    for file in val_files:
        source_file = os.path.join(source_genus_dir, file)
        destination_file = os.path.join(destination_val_dir, file)
        shutil.copy(source_file, destination_file)

    for file in test_files:
        source_file = os.path.join(source_genus_dir, file)
        destination_file = os.path.join(destination_test_dir, file)
        shutil.copy(source_file, destination_file)

    print(f"Data split and copied for genus {genus}")