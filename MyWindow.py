import numpy as np
import tensorflow as tf
from PIL import ImageFile
# from keras.src.legacy.preprocessing import image

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QMovie
from PyQt5.uic import loadUi


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi("gui/gui.ui", self)

        # Set default stuff here
        # --------------------------

        self.setWindowTitle("Mushroom classifier")
        self.out_label.setText("")
        self.image_preview.setPixmap(QtGui.QPixmap("images/logo.png"))

        # --------------------------

        self.setStyleSheet("background-color: rgb(20, 20, 20);"
                           "color: rgb(255, 255, 255);"
                           "font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;")
        self.out_label.setStyleSheet("font-size: 18px;")
        self.img_path = ""

        self.movie = QMovie("gui/wallow.gif")
        self.movie.start()

        self.select_button.clicked.connect(self.select_file)
        self.select_button.setStyleSheet("background-color: rgb(40, 40, 40);")
        self.identify_button.clicked.connect(self.process)
        self.identify_button.setStyleSheet("background-color: rgb(40, 40, 40);")

        app_icon = QtGui.QIcon()
        app_icon.addFile('gui/wallow.gif', QtCore.QSize(16, 16))
        app_icon.addFile('gui/wallow.gif', QtCore.QSize(24, 24))
        app_icon.addFile('gui/wallow.gif', QtCore.QSize(32, 32))
        app_icon.addFile('gui/wallow.gif', QtCore.QSize(48, 48))
        app_icon.addFile('gui/wallow.gif', QtCore.QSize(256, 256))

        self.setWindowIcon(app_icon)
        self.show()

    def process(self):
        if self.img_path == "":
            self.shroom_laugh_l.setMovie(self.movie)
            self.shroom_laugh_r.setMovie(self.movie)
            self.movie.start()
            self.image_preview.setPixmap(QtGui.QPixmap("images/logo.png").scaledToWidth(200))
            self.out_label.setText("🥶🥶🥶🥶🥶")
            return

        self.out_label.setText("Processing image...")
        self.update_image()
        cnn = tf.keras.models.load_model("MODEL3.keras")
        test_image = tf.keras.utils.load_img(self.img_path, target_size=(224, 224))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255
        result = cnn.predict(test_image)
        answer = np.argmax(result, axis=1)

        name = ''
        if answer == 0:
            name = 'Amanita'
        elif answer == 1:
            name = 'Boletus'
        elif answer == 2:
            name = 'Cantharellus'
        elif answer == 3:
            name = 'Lactarius'

        self.out_label.setText(name)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select image", "C:/", "Images (*.jpg *.png *.bmp)")

        self.shroom_laugh_l.setMovie(None)
        self.shroom_laugh_r.setMovie(None)
        self.movie.stop()

        if fname == "":
            return

        self.out_label.setPixmap(QtGui.QPixmap(None))
        self.out_label.setText("Image loaded")
        self.img_path = fname
        self.update_image()

    def update_image(self):
        # Kiedys tu bylo wiecej, moze ta funkcja sie jeszcze przyda
        self.image_preview.setPixmap(QtGui.QPixmap(self.img_path))
