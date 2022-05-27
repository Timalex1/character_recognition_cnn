# cnn-project.py
# Generated from c:\Users\Timilehin Vincent\Desktop\Desktop\project\CNN_AlphabetRecognition-master\cnn-project.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
from enum import Enum

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

# IMPORTS:
# from . import cnn-project


class cnnproject(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)  # Call the inherited classes __init__ method
        self.ui = uic.loadUi("cnn-project.ui", self)  # Load the .ui file

        self.ui.pickImage.click.connect(self.b2_clicked)
        self.initUI()
        self.show()  # Show the GUI

    def b2_clicked(self):
        print('yellow')

    def initUI(self):
        pass

    def __str__(self):
        return


# if __name__ == '__main__':
#     app = QtGui.QGuiApplication(sys.argv)
#     window=cnnproject()
#     window.show()
#     sys.exit(app.exec_())
