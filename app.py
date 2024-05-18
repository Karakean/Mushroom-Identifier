import sys

from PyQt5.QtWidgets import QApplication
import MyWindow


def window():
    model = "MODEL3.keras"
    app = QApplication(sys.argv)
    win = MyWindow.MyWindow(model)
    sys.exit(app.exec_())


if __name__ == '__main__':
    window()

