#!/usr/bin/env python
"""
example_viewer_usage.py

Simple usage example for the PyQt6-only image viewer.
Run:
    python example_viewer_usage.py
"""
import numpy as np
from image_viewer_app import launch_image_viewer

def main():
    # Synthetic test image
    x = np.linspace(0, 1, 512)
    y = np.linspace(0, 1, 512)
    X, Y = np.meshgrid(x, y)
    img = np.sin(10 * np.pi * X) * np.cos(10 * np.pi * Y)

    # Open two independent viewer instances
    launch_image_viewer(img, title="Example 1: sincos")
    launch_image_viewer(np.log1p(2 + img), title="Example 2: log1p(2+img)")

if __name__ == "__main__":
    main()
