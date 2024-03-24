import math
import cv2
import os
import numpy as np
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def normalize_and_save_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array_scaled = image_array * (1.0 / 255.0)
    min_val = np.min(image_array_scaled)
    max_val = np.max(image_array_scaled)
    normalized_image_array = (image_array_scaled - min_val) / (max_val - min_val)
    normalized_image = Image.fromarray((normalized_image_array * 255).astype(np.uint8))
    normalized_image.save(image_path)


def normalize_images_set(image_set_path):
    files = [file for file in os.listdir(image_set_path) if not file.startswith('.')]
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_set_path, file)
            try:
                normalize_and_save_image(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


def augment_and_save_image(image_path, output_dir, aug_count):
    image = Image.open(image_path)
    image_array = np.array(image)
    x = np.expand_dims(image_array, axis=0)
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.75, 1.25],
        channel_shift_range=100.0,
        horizontal_flip=True
    )
    original_image_path = os.path.join(output_dir, "original_" + os.path.basename(image_path))
    image.save(original_image_path)
    for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_format='jpg')):
        if i >= (aug_count - 1):
            break


def augment_and_equalize_images_set(image_set_path, output_set_path, aug_count, max_images_num):
    files = [file for file in os.listdir(image_set_path) if not file.startswith('.')]
    images_num = len(files)
    diff = max_images_num - images_num
    multi = math.floor(diff / images_num)  # it will be more than 0 only if diff is larger than current set size
    diff_leftover = diff - multi * images_num
    for i, file in enumerate(files):
        additional_aug = (multi + int(i < diff_leftover)) * (aug_count + 1)
        img_path = os.path.join(image_set_path, file)
        try:
            augment_and_save_image(img_path, output_set_path, aug_count + additional_aug)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
