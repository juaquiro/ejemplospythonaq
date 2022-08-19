## \file Example_GUI_2.py
#  \details This GUI makes a basic demo of signals and slots en PYQt5
#  \brief This GUI makes a basic demo of signals and slots en PYQt5
#  \see https://www.pythonguis.com/tutorials/pyqt-signals-slots-events/
#  \copyright 2021 IOT
#  \author AQ 2021

import sys
from random import choice
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QVBoxLayout, QWidget

# list of window titles
window_titles = [
    'My App',
    'My App',
    'Still My App',
    'Still My App',
    'What on earth',
    'What on earth',
    'This is surprising',
    'This is surprising',
    'Something went wrong']


class MainWindow(QMainWindow):
    """!
     @brief MainWindow for the app, inherits from QMainWindow.
     @details A main window provides a framework for building an application’s user interface.
     Qt has QMainWindow and its related classes for main window management.
     QMainWindow has its own layout to which you can add QToolBars, QDockWidgets, a QMenuBar , and a QStatusBar .
     The layout has a center area that can be occupied by any kind of widget.
     @see https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QMainWindow.html#more
     @see https://doc.qt.io/qtforpython-5/_images/mainwindowlayout.png
     @author AQ 2021
     """

    def __init__(self):
        # first call superclass constructor
        super().__init__()

        # second initialize class
        self.button_is_checked = False
        self.n_times_clicked = 0

        self.setWindowTitle("My App")

        # NOTA: si no especificamos self.button, button no seria un miembro de la clase
        # seria una variable local de __init__
        self.button = QPushButton("Press Me!")

        self.button.setObjectName("Button_1")
        self.button.setCheckable(True)

        self.button.clicked.connect(self.the_button_was_clicked)
        self.button.clicked.connect(self.the_button_was_toggled)

        self.windowTitleChanged.connect(self.the_window_title_changed)

        # Connecting widgets together directly
        #
        # So far we've seen examples of connecting widget signals to Python methods.
        # When a signal is fired from the widget, our Python method is called and receives the data from the signal.
        # But you don't always need to use a Python function to handle signals
        # -- you can also connect Qt widgets directly to one another.
        self.label = QLabel()
        self.input = QLineEdit()
        # link input.textChanged signal with label.setText slot
        self.input.textChanged.connect(self.label.setText)

        # reate a layout and add the button, input and label
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.input)
        layout.addWidget(self.label)

        # reate a container and set layout
        container = QWidget()
        container.setLayout(layout)

        # Set the central widget of the Window.
        # CentralWidget can be anu widget for example button: self.setCentralWidget(self.button)
        self.setCentralWidget(container)

    def the_button_was_clicked(self):
        """!
        @brief slot for signal button.clicked
        @see
        @author AQ 2021
        """
        # The sender object:
        sender = self.sender()
        # The sender object's name:
        senderName = sender.objectName()
        print(senderName + " Clicked!")

        new_window_title = choice(window_titles)
        print("Setting title:  %s" % new_window_title)
        self.setWindowTitle(new_window_title)

    def the_button_was_toggled(self, checked):
        """!
           @brief slot for signal button.clicked, this time we check for the
           @details signals can also send data to provide more information about what has just happened.
           The .clicked signal is no exception, also providing a checked (or toggled) state for the button.
           For normal buttons this is always False. For this we make our button checkable and see the effect.
           @author AQ 2021
           """
        # The sender object:
        sender = self.sender()
        # The sender object's name:
        senderName = sender.objectName()

        self.button_is_checked = checked
        print(senderName + " Checked?", self.button_is_checked)

    def the_window_title_changed(self, window_title):
        """!
            @brief slot for signal windowTitleChanged, this signal is launched only if the title changes.
            @author AQ 2021
            """
        print("Window title changed: %s" % window_title)

        if window_title == 'Something went wrong':
            # NOTA: OJO en Python si button no es un miembro, el interprete no se queja, se va a crear empty y
            # se va a llamar a setDisabled() provocando un crash
            self.button.setDisabled(True)


# start PyQt QApplication
app = QApplication(sys.argv)
# set margin for all QPushButton widgets
app.setStyleSheet("QPushButton { margin: 3ex; }")

# start window as a MainWindow widget
# see https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QMainWindow.html#more
window = MainWindow()
window.show()

app.exec()
