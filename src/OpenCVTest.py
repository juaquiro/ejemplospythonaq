## \file OpenCVTest.py
#  \brief demo file for using openCV from a python console
#  \details NA
#  \copyright 2021 IOT
#  \author AQ 2021

import cv2
import numpy as np
from matplotlib import pyplot as plt


##
#  \brief openCV filter a complex phasor
#  \details NA
#  \copyright 2021 IOT
#  \author AQ 2021
def testOpenCV001():
    # check version
    print(cv2.__version__)

    # setup meshgrid
    NR, NC, NP = (512, 511, 1)
    x = np.arange(0, NC)
    x = x - 0.5 * NC
    y = np.arange(0, NR)
    y = y - 0.5 * NR

    X, Y = np.meshgrid(x, y)

    # fringe period in Px
    Tx, Ty = (30, 30)

    # phase, background and modulation
    phi = 2 * np.pi * (X / Tx + Y / Ty)
    b = 100  # GV
    sigma_x, sigma_y = (0.25 * NC, 0.25 * NR)  # px
    m = np.exp(-X ** 2 / (2 * sigma_x ** 2) - Y ** 2 / (2 * sigma_y ** 2))
    z = m * (np.cos(phi) + 1j * np.sin(phi))

    # a veces las referencias a un modulo importado da un warning: Cannot find reference 'xxxx' in '__init__.py | __init__.py'
    # para solucionarlo:
    # Right-click on source directory
    # Mark Directory As --> Source Root
    # File --> Invalidate Caches / Restart... -> Invalidate and Restart

    filter_size = (19, 19)  # px
    h = np.ones(filter_size) / (filter_size[0] * filter_size[1])
    z_filt = cv2.filter2D(z.real, -1, h) + 1j * cv2.filter2D(z.imag, -1, h)

    plt.figure()
    plt.imshow(np.abs(z_filt))
    plt.show()


if __name__ == '__main__':
    testOpenCV001()
