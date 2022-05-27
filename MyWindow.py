import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras_preprocessing import image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QFileDialog, QApplication, QMainWindow
from PyQt5.QtGui import QMovie
from PyQt5.uic import loadUi


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        loadUi("gui.ui", self)

        # Set default stuff here
        # --------------------------

        self.setWindowTitle("My first GUI ;)")
        self.out_label.setText("sample text")
        self.image_preview.setPixmap(QtGui.QPixmap("images/default.png"))

        # --------------------------

        self.setStyleSheet("background-color: rgb(20, 20, 20);"
                           "color: rgb(255, 255, 255);"
                           "font-family: \"Comic Sans MS\", \"Comic Sans\", cursive;")
        self.out_label.setStyleSheet("font-size: 18px;")
        self.img_path = ""

        self.movie = QMovie("images/laugh.gif")
        self.movie.start()

        self.select_button.clicked.connect(self.select_file)
        self.select_button.setStyleSheet("background-color: rgb(40, 40, 40);")
        self.identify_button.clicked.connect(self.process)
        self.identify_button.setStyleSheet("background-color: rgb(40, 40, 40);")

        app_icon = QtGui.QIcon()
        app_icon.addFile('images/default.gif', QtCore.QSize(16, 16))
        app_icon.addFile('images/default.gif', QtCore.QSize(24, 24))
        app_icon.addFile('images/default.gif', QtCore.QSize(32, 32))
        app_icon.addFile('images/default.gif', QtCore.QSize(48, 48))
        app_icon.addFile('images/default.gif', QtCore.QSize(256, 256))

        self.setWindowIcon(app_icon)
        self.show()

    def process(self):
        if self.img_path == "":
            self.shroom_laugh_l.setMovie(self.movie)
            self.shroom_laugh_r.setMovie(self.movie)
            self.movie.start()
            self.image_preview.setPixmap(QtGui.QPixmap("images/mm.png").scaledToWidth(200))
            self.out_label.setText("🥶🥶🥶🥶🥶")
            return

        cnn = tf.keras.models.load_model("model.h5")
        test_image = image.load_img(self.img_path, target_size=(64, 64))
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

        self.out_label.setText(name)

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select image", "C:/", "Images (*.jpg *.png *.bmp)")

        self.shroom_laugh_l.setMovie(None)
        self.shroom_laugh_r.setMovie(None)
        self.movie.stop()

        if fname == "":
            return

        self.file_path.setText(fname)
        self.out_label.setPixmap(QtGui.QPixmap(None))
        self.out_label.setText("Image loaded")
        self.img_path = fname
        self.update_image()

    def update_image(self):
        # Kiedys tu bylo wiecej, moze ta funkcja sie jeszcze przyda
        self.image_preview.setPixmap(QtGui.QPixmap(self.img_path))
