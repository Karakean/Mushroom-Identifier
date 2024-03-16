import numpy as np
import os
from PIL import Image
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# print("IMAGE 1")
# print(np.array(Image.open("splits/split1/test/Amanita/002_pJY3-9Ttfto.jpg")))
# print("IMAGE 2")
# print(np.array(Image.open("splits/tmp/test/Amanita/002_pJY3-9Ttfto.jpg")))

# image = Image.open("002_pJY3-9Ttfto.jpg")
# image_array = np.array(image)
# print("IMAGE 1")
# print(image_array)
# mean = np.mean(image_array)
# std_dev = np.std(image_array)
# image_array = (image_array - mean) / std_dev
# print("IMAGE 2")
# print(image_array)
# new_image = Image.fromarray(image_array)
# new_image.save("test.jpg")

image = Image.open("test_img.jpg")
image_array = np.array(image)

x = np.expand_dims(image_array, axis=0)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

original_image_path = os.path.join("tmp", "original_test_img.jpg")
image.save(original_image_path)

for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir="tmp", save_prefix="augmented", save_format='jpg')):
    if i >= 5:
        break


