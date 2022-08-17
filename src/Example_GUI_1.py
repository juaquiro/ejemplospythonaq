## \file Example_GUI_1.py
#  \details This GUI makes a basic demo of PYQt5, including creating an app, seting layout, ad buttons and set and
#  connect a signal/slot pair
#  \brief This GUI makes a basic demo of PYQt5
#  \see https://build-system.fman.io/pyqt5-tutorial
#  \copyright 2021 IOT
#  \author AQ 2021
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox


# slot example (callback function). the term slot is important when using Qt from C++, because slots must be declared in a
# special way in C++.
# In Python however, any function can be a slot.
# For this reason, the distinction between slots and "normal" functions has little relevance in Python.
# This is simply a function that gets called when a signal occurs
def on_button_clicked():
    """!
    @brief example of
    @see testFPAUtils.py for unit tests
    @author AQ 2021
    """
    alert = QMessageBox()
    alert.setText('You clicked the button!')
    alert.exec()


# we create a QApplication with the command:
# This is a requirement of Qt: Every GUI app must have exactly one instance of QApplication.
# Many parts of Qt don't work until you have executed the line.
# You will therefore need it in virtually every (Py)Qt app you write.
app = QApplication([])

# Built-in styles
# The coarsest way to change the appearance of your application is to set the global Style.
# The available styles depend on your platform.
# Usually are 'Fusion', 'Windows', 'WindowsVista' (Windows only) and 'Macintosh' (Mac only).
app.setStyle('Fusion')

# In addition to the above, you can change the appearance of your application via style sheets.
# This is Qt's analogue of CSS. We can use this for example to add some spacing:
app.setStyleSheet("QPushButton { margin: 3ex; }")

# Everything you see in a (Py)Qt app is a widget: Buttons, labels, windows, dialogs, progress bars etc.
# Like HTML elements, widgets are often nested. For example, a window can contain a button,
# which in turn contains a label.
window = QWidget()

# your GUI will most likely consist of multiple widgets.
# In this case, you need to tell Qt how to position them.
# For instance, you can use QVBoxLayout to stack widgets vertically
layout = QVBoxLayout()

buttonTop = QPushButton('Top')
layout.addWidget(buttonTop)

buttonBottom = QPushButton('Bottom')
layout.addWidget(buttonBottom)
window.setLayout(layout)

# signal connection to a slot
# QPushButton.clicked is a signal. connect(...) lets us install the slot on it.
# the connected slot gets called when the signal occurs
buttonTop.clicked.connect(on_button_clicked)

# tell Qt to show the window widget on the screen:
window.show()

# The last step is to hand control over to Qt and ask it to "run the application until the user closes it".
app.exec()
