import tensorflow as tf
from PIL import ImageFile
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_plots(results, name):
    filename = name.replace(" ", "_")

    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('Accuracy with '+name)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(filename+'_accuracy.png')
    plt.show()

    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Loss with '+name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(filename+'_loss.png')
    plt.show()


def generate_model_from_scratch(dataset_path, classes_number, image_size, epochs_number):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    training_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                               batch_size=32, class_mode='categorical', subset='training')

    validation_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                                 batch_size=32, class_mode='categorical', subset='validation')

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',
                                   input_shape=(image_size, image_size, 3)))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.5))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=classes_number, activation='softmax'))
    cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    results = cnn.fit(x=training_set, validation_data=validation_set, epochs=epochs_number)
    cnn.save('model_from_scratch.h5')

    make_plots(results, 'model created from scratch')


def generate_pretrained_model(dataset_path, classes_number, image_size, epochs_number):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    training_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                               batch_size=32, class_mode='categorical', subset='training')

    validation_set = datagen.flow_from_directory(dataset_path, target_size=(image_size, image_size),
                                                 batch_size=32, class_mode='categorical', subset='validation')

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
    for layer in vgg.layers:
        layer.trainable = False

    cnn = tf.keras.models.Sequential()
    cnn.add(vgg)
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(classes_number, activation='softmax'))
    cnn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('vgg.h5', monitor='val_accuracy', save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    results = cnn.fit(x=training_set, validation_data=validation_set, epochs=epochs_number,
                      callbacks=[checkpoint, earlystop])
    cnn.save('pretrained_model.h5')

    make_plots(results, 'pretrained model')


def main():
    dataset_path = 'original_dataset'
    classes_number = 7
    image_size = 64
    epochs_number = 5
    # generate_model_from_scratch(dataset_path, classes_number, image_size, epochs_number)
    generate_pretrained_model(dataset_path, classes_number, image_size, epochs_number)


main()
