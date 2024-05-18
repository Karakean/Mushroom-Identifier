import numpy as np
import tensorflow as tf
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication
from PyQt5.QtGui import QPixmap, QIcon, QMovie
from PyQt5.uic import loadUi


class MyWindow(QMainWindow):
    def __init__(self, model_path):
        super(MyWindow, self).__init__()
        loadUi("gui/gui.ui", self)
        self.setWindowTitle("Mushroom Classifier")
        self.setStyleSheet(
            "background-color: rgb(20, 20, 20); color: rgb(255, 255, 255); font-family: 'Open Sans', cursive;")
        self.out_label.setStyleSheet("font-size: 18px;")
        self.select_button.setStyleSheet("background-color: rgb(40, 40, 40);")
        self.identify_button.setStyleSheet("background-color: rgb(40, 40, 40);")
        self.movie = None
        self.img_path = None
        self.model = model_path
        self.select_button.clicked.connect(self.select_file)
        self.identify_button.clicked.connect(self.process)
        self.init_ui()

    def init_ui(self):
        self.movie = QMovie("gui/wallow.gif")
        self.image_preview.setMovie(self.movie)
        self.movie.start()
        app_icon = QIcon()
        for size in [16, 24, 32, 48, 256]:
            app_icon.addFile('gui/logo.png', QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)
        self.out_label.setText("Welcome to Mushroom Classifier!\nPlease upload a mushroom image.")
        self.show()

    def process(self):
        if not self.img_path:
            self.out_label.setText("No image selected! Please select an image.")
            return
        self.out_label.setText("Processing image...")
        QApplication.processEvents()
        QTimer.singleShot(25, self.predict)

    def predict(self):
        cnn = tf.keras.models.load_model(self.model)
        test_image = tf.keras.utils.load_img(self.img_path, target_size=(224, 224))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0
        result = cnn.predict(test_image)
        name_map = {0: 'Amanita', 1: 'Boletus', 2: 'Cantharellus', 3: 'Lactarius'}
        name = name_map.get(np.argmax(result), 'Unknown')
        self.out_label.setText(f"Mushroom classified as: {name}\nYou can select another image.")

    def select_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select image", "C:/", "Images (*.jpg *.png *.bmp)")
        if fname:
            self.img_path = fname
            self.image_preview.setPixmap(QPixmap(fname))
            self.movie.stop()
            self.out_label.setText("Image loaded successfully.\nClick \"Identify\" button to identify this mushroom.")
        else:
            self.img_path = None
            self.out_label.setText("No image selected")
            self.image_preview.setMovie(self.movie)
            self.movie.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyWindow()
    sys.exit(app.exec_())
