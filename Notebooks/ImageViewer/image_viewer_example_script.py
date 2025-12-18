# my_image_ops.py
"""
Minimal image-processing function for debugging from a Jupyter notebook in VS Code.
"""


import numpy as np
import matplotlib.pyplot as plt
from image_viewer_app import launch_image_viewer


def enhance_contrast(img, gain=1.5, bias=0.1):
    """
    Simple contrast "enhancement" for a grayscale image in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        2D image array with values in [0, 1].
    gain : float
        Multiplicative factor.
    bias : float
        Additive offset.

    Returns
    -------
    np.ndarray
        Processed image, still in [0, 1].
    """
    # Put a BREAKPOINT on the next line in VS Code
    stretched = img * gain + bias

    stretched = np.clip(stretched, 0.0, 1.0)
    return stretched


if __name__ == '__main__':

    # Example: synthetic test image
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    X, Y = np.meshgrid(x, y)
    img = np.sin(10 * np.pi * X) * np.cos(10 * np.pi * Y)
    img_stretched=enhance_contrast(img, gain=2, bias=1)


    # Show in notebook just to check (optional)
    plt.imshow(img, cmap="gray")
    plt.title("In-notebook preview")
    plt.show(block=False)
    plt.pause(2)   # or 0.01

    # Launch external viewer (non-blocking, own process)
    launch_image_viewer(img, title="Original image")

    launch_image_viewer(np.log(2+img), title="Log original image")