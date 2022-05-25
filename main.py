import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(
        'training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.5))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=9, activation='softmax'))
    cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=test_set, epochs=20)

    test_image = image.load_img('sample_images/1.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    training_set.class_indices

    if result[0][0] == 1:
        print('Agaricus')
    elif result[0][1] == 1:
        print('Amanita')
    elif result[0][2] == 1:
        print('Boletus')
    elif result[0][3] == 1:
        print('Cortinarius')
    elif result[0][4] == 1:
        print("Entoloma")
    elif result[0][5] == 1:
        print("Hygrocybe")
    elif result[0][6] == 1:
        print("Lactarius")
    elif result[0][7] == 1:
        print("Russula")
    elif result[0][8] == 1:
        print("Suillus")



main()
