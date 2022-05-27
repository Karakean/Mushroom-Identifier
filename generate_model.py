import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(
        'training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.5))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=9, activation='softmax'))
    cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=test_set, epochs=50)
    cnn.save("model.h5")

    test_image = image.load_img('1.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)

    name = ''
    if result[0][0] == 1:
        name = 'Agaricus'
    elif result[0][1] == 1:
        name = 'Amanita'
    elif result[0][2] == 1:
        name = 'Boletus'
    elif result[0][3] == 1:
        name = 'Cortinarius'
    elif result[0][4] == 1:
        name = 'Entoloma'
    elif result[0][5] == 1:
        name = 'Hygrocybe'
    elif result[0][6] == 1:
        name = 'Lactarius'
    elif result[0][7] == 1:
        name = 'Russula'
    elif result[0][8] == 1:
        name = 'Suillus'

    print(name)


generate_model()
