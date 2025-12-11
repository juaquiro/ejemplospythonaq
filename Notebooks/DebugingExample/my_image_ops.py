# my_image_ops.py
"""
Minimal image-processing function for debugging from a Jupyter notebook in VS Code.
"""

import numpy as np


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
