## \file Example_GUI_3.py
#  \brief This GUI makes a basic demo of Qt events
#  \brief Every interaction the user has with a Qt application is an event.
#  There are many types of event, each representing a different type of interaction.
#  Qt represents these events using event objects which package up information about what happened.
#  These events are passed to specific event handlers on the widget where the interaction occurred.
#  One of the main events which widgets receive is the QMouseEvent.
#  QMouseEvent events are created for each and every mouse movement and button click on a widget
#  \see https://www.pythonguis.com/tutorials/pyqt-signals-slots-events/
#  \copyright 2021 IOT
#  \author AQ 2021


import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.label = QLabel("Click in this window")
        self.setCentralWidget(self.label)

    def mouseMoveEvent(self, e):
        self.label.setText("mouseMoveEvent")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # handle the left-button press in here
            self.label.setText("mousePressEvent LEFT")

        elif e.button() == Qt.MiddleButton:
            # handle the middle-button press in here.
            self.label.setText("mousePressEvent MIDDLE")

        elif e.button() == Qt.RightButton:
            # handle the right-button press in here.
            self.label.setText("mousePressEvent RIGHT")

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.label.setText("mouseReleaseEvent LEFT")

        elif e.button() == Qt.MiddleButton:
            self.label.setText("mouseReleaseEvent MIDDLE")

        elif e.button() == Qt.RightButton:
            self.label.setText("mouseReleaseEvent RIGHT")

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.label.setText("mouseDoubleClickEvent LEFT")

        elif e.button() == Qt.MiddleButton:
            self.label.setText("mouseDoubleClickEvent MIDDLE")

        elif e.button() == Qt.RightButton:
            self.label.setText("mouseDoubleClickEvent RIGHT")


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
