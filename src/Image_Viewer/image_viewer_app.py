#!/usr/bin/env python
"""
image_viewer_app.py (PyQt6-only)

Standalone image viewer inspired by MATLAB's Image Viewer app.

Key properties
--------------
- Runs as its own GUI process when launched via `launch_image_viewer(...)` (non-modal, non-blocking).
- Multiple instances can run simultaneously.
- PyQt6-only: raises ImportError if PyQt6 is not installed.

Features
--------
a) Histogram panel (dockable)
b) Line profile between two user-selected points (opens in separate non-modal window)
   - Endpoints are draggable; profile updates live
   - Optional profile averaging width ±k pixels (perpendicular averaging)
c) Live X/Y profiles at cursor position (with optional averaging ±k pixels):
   - Vertical profile shown on the RIGHT
   - Horizontal profile shown BELOW
d) Image export: PNG, TIFF, BMP, JPEG (exports the current view "as shown")
e) Dynamic crosshair (vertical + horizontal) through cursor point while hovering the image
f) Fine cursor movement with keyboard arrows (when cursor is on the image)

Requirements
------------
- numpy
- matplotlib
- PyQt6
(Optional, for TIFF/JPEG support depending on your setup): Pillow

Install (conda-forge recommended)
--------------------------------
conda install -c conda-forge pyqt matplotlib numpy

Usage
-----
From Python / Jupyter (NumPy array):
    from image_viewer_app import launch_image_viewer
    launch_image_viewer(img, title="My image")
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---- PyQt6-only enforcement (must happen BEFORE importing matplotlib Qt backends) ----
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
except Exception as e:
    raise ImportError(
        "PyQt6 is required for this viewer. Install with:\n"
        "  conda install -c conda-forge pyqt\n"
        "or\n"
        "  pip install PyQt6\n"
    ) from e

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_float_image(image: np.ndarray) -> np.ndarray:
    """Convert to float32 for internal processing, without copying if possible."""
    if image.dtype == np.float32:
        return image
    return image.astype(np.float32, copy=False)

def _is_rgb_like(image: np.ndarray) -> bool:
    return image.ndim == 3 and image.shape[2] in (3, 4)

def _intensity(image: np.ndarray) -> np.ndarray:
    """
    Get a scalar intensity image for profiles/histogram.
    - Grayscale: itself
    - RGB/RGBA: average of RGB channels (simple, dependency-free)
    """
    if image.ndim == 2:
        return image
    if _is_rgb_like(image):
        rgb = image[..., :3]
        return np.mean(rgb, axis=2)
    raise ValueError("Unsupported image shape for intensity.")

def _clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0)

def _bilinear_sample(img2d: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Bilinear sampling of a 2D image at floating-point coordinates (x,y).
    xs, ys are arrays of same shape in pixel coordinates (0..W-1, 0..H-1).
    Returns sampled values, with coordinates clamped to bounds.
    """
    h, w = img2d.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    dx = xs - x0
    dy = ys - y0

    Ia = img2d[y0, x0]
    Ib = img2d[y0, x1]
    Ic = img2d[y1, x0]
    Id = img2d[y1, x1]

    return (Ia * (1 - dx) * (1 - dy) +
            Ib * dx * (1 - dy) +
            Ic * (1 - dx) * dy +
            Id * dx * dy)

def _load_image_from_path(path: str) -> np.ndarray:
    import matplotlib.image as mpimg
    if path.lower().endswith(".npy"):
        return np.load(path)
    return mpimg.imread(path)

def _qt_orientation_horizontal() -> QtCore.Qt.Orientation:
    return QtCore.Qt.Orientation.Horizontal


# ---------------------------------------------------------------------
# Auxiliary windows
# ---------------------------------------------------------------------

class PlotWindow(QtWidgets.QMainWindow):
    """A simple non-modal window hosting a Matplotlib figure + toolbar."""
    def __init__(self, title: str):
        super().__init__()
        self.setWindowTitle(title)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)
        self.setCentralWidget(central)

    def plot_line(self, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()


class LineProfileWindow(PlotWindow):
    """Non-modal window that can be updated (for draggable endpoints)."""
    def __init__(self, title: str = "Line profile"):
        super().__init__(title)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot([], [])
        self.ax.set_xlabel("Distance (px)")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def update_profile(self, dist: np.ndarray, prof: np.ndarray, title: str = "Line profile"):
        self.ax.set_title(title)
        self.line.set_data(dist, prof)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()


# ---------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------

@dataclass
class _LineProfileState:
    active: bool = False
    p1: Optional[Tuple[float, float]] = None  # (x,y) in image coords
    p2: Optional[Tuple[float, float]] = None
    line_artist: Optional[object] = None      # matplotlib Line2D
    p1_artist: Optional[object] = None        # endpoint marker
    p2_artist: Optional[object] = None
    dragging: Optional[str] = None            # "p1" | "p2" | None
    window: Optional[LineProfileWindow] = None


class ImageViewer(QtWidgets.QMainWindow):
    def __init__(self, image: np.ndarray, title: str = "Image Viewer"):
        super().__init__()
        self.setWindowTitle(title)

        if not (image.ndim == 2 or _is_rgb_like(image)):
            raise ValueError("Only 2D grayscale or 3D RGB/RGBA images are supported.")

        self.image = _as_float_image(image)
        self.is_grayscale = (self.image.ndim == 2)

        # Scalar image for histogram/profiles
        self.scalar = _as_float_image(_intensity(self.image))

        self._data_min = float(np.min(self.scalar))
        self._data_max = float(np.max(self.scalar))
        if self._data_max == self._data_min:
            self._data_max = self._data_min + 1.0

        # Display controls
        self._vmin = self._data_min
        self._vmax = self._data_max
        self._current_cmap = "gray"

        # Cursor state
        self._last_cursor_ij: Optional[Tuple[int, int]] = None
        self._cursor_in_image: bool = False

        # Line profile state
        self._line_profile = _LineProfileState()

        self._build_ui()
        self._connect_events()
        self._update_all()

    # ---------------- UI ----------------

    def _build_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        act_export = QtGui.QAction("Export view...", self)
        act_export.triggered.connect(self._export_view)
        file_menu.addAction(act_export)

        act_close = QtGui.QAction("Close", self)
        act_close.triggered.connect(self.close)
        file_menu.addAction(act_close)

        tools_menu = menubar.addMenu("&Tools")
        self.act_line_profile = QtGui.QAction("Line profile (2 clicks)", self)
        self.act_line_profile.setCheckable(True)
        self.act_line_profile.triggered.connect(self._toggle_line_profile_mode)
        tools_menu.addAction(self.act_line_profile)

        self.act_clear_line = QtGui.QAction("Clear line profile overlay", self)
        self.act_clear_line.triggered.connect(self._clear_line_profile_overlay)
        tools_menu.addAction(self.act_clear_line)

        # Central layout
        central = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        # Image canvas
        self.fig_img = Figure(figsize=(6, 5), dpi=100)
        self.canvas_img = FigureCanvasQTAgg(self.fig_img)
        self.canvas_img.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_axis_off()
        self.toolbar = NavigationToolbar2QT(self.canvas_img, self)

        img_box = QtWidgets.QVBoxLayout()
        img_box.addWidget(self.toolbar)
        img_box.addWidget(self.canvas_img, stretch=1)
        top_row.addLayout(img_box, stretch=3)

        # Vertical profile (right)
        self.fig_v = Figure(figsize=(3, 5), dpi=100)
        self.canvas_v = FigureCanvasQTAgg(self.fig_v)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title("Vertical profile")
        self.ax_v.set_xlabel("Value")
        self.ax_v.set_ylabel("y (px)")
        self.ax_v.grid(True, alpha=0.3)
        top_row.addWidget(self.canvas_v, stretch=1)

        outer.addLayout(top_row, stretch=3)

        # Horizontal profile (bottom)
        self.fig_h = Figure(figsize=(8, 2.5), dpi=100)
        self.canvas_h = FigureCanvasQTAgg(self.fig_h)
        self.ax_h = self.fig_h.add_subplot(111)
        self.ax_h.set_title("Horizontal profile")
        self.ax_h.set_xlabel("x (px)")
        self.ax_h.set_ylabel("Value")
        self.ax_h.grid(True, alpha=0.3)
        outer.addWidget(self.canvas_h, stretch=1)

        # Controls
        controls = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_group = QtWidgets.QGroupBox("Display (scalar channel)")
        gl = QtWidgets.QGridLayout(display_group)

        self.slider_min = QtWidgets.QSlider(_qt_orientation_horizontal())
        self.slider_max = QtWidgets.QSlider(_qt_orientation_horizontal())
        self.slider_min.setRange(0, 1000)
        self.slider_max.setRange(0, 1000)
        self.slider_min.setValue(0)
        self.slider_max.setValue(1000)

        self.label_min = QtWidgets.QLabel(f"Min: {self._vmin:.3g}")
        self.label_max = QtWidgets.QLabel(f"Max: {self._vmax:.3g}")

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "plasma", "inferno", "magma", "jet", "hot", "cool", "bone"])
        self.cmap_combo.setCurrentText("gray")

        gl.addWidget(QtWidgets.QLabel("Lower"), 0, 0)
        gl.addWidget(self.slider_min, 0, 1)
        gl.addWidget(self.label_min, 0, 2)

        gl.addWidget(QtWidgets.QLabel("Upper"), 1, 0)
        gl.addWidget(self.slider_max, 1, 1)
        gl.addWidget(self.label_max, 1, 2)

        gl.addWidget(QtWidgets.QLabel("Colormap"), 2, 0)
        gl.addWidget(self.cmap_combo, 2, 1, 1, 2)

        gl.addWidget(QtWidgets.QLabel("Profile avg ±k (px)"), 3, 0)
        self.spin_avg = QtWidgets.QSpinBox()
        self.spin_avg.setRange(0, 50)
        self.spin_avg.setValue(0)
        self.spin_avg.setToolTip("Averaging half-width in pixels for X/Y profiles and line profile.")
        gl.addWidget(self.spin_avg, 3, 1, 1, 2)

        controls_layout.addWidget(display_group)
        controls_layout.addStretch(1)
        outer.addWidget(controls, stretch=0)

        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # Histogram dock
        self.hist_dock = QtWidgets.QDockWidget("Histogram", self)
        self.hist_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

        hist_widget = QtWidgets.QWidget(self.hist_dock)
        hist_layout = QtWidgets.QVBoxLayout(hist_widget)
        hist_layout.setContentsMargins(4, 4, 4, 4)

        self.fig_hist = Figure(figsize=(4, 3), dpi=100)
        self.canvas_hist = FigureCanvasQTAgg(self.fig_hist)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_hist.set_title("Histogram (scalar)")
        self.ax_hist.set_xlabel("Value")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(True, alpha=0.3)

        bins_row = QtWidgets.QHBoxLayout()
        bins_row.addWidget(QtWidgets.QLabel("Bins:"))
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(16, 4096)
        self.spin_bins.setValue(256)
        bins_row.addWidget(self.spin_bins)
        bins_row.addStretch(1)

        hist_layout.addLayout(bins_row)
        hist_layout.addWidget(self.canvas_hist, stretch=1)

        self.hist_dock.setWidget(hist_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.hist_dock)

    # ---------------- events ----------------

    def _connect_events(self):
        self.slider_min.valueChanged.connect(self._on_contrast_change)
        self.slider_max.valueChanged.connect(self._on_contrast_change)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_change)
        self.spin_bins.valueChanged.connect(lambda _: self._update_histogram())
        self.spin_avg.valueChanged.connect(lambda _: self._refresh_profiles())

        self.canvas_img.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas_img.mpl_connect("button_press_event", self._on_click)
        self.canvas_img.mpl_connect("button_release_event", self._on_release)
        self.canvas_img.mpl_connect("figure_leave_event", self._on_leave)
        self.canvas_img.mpl_connect("key_press_event", self._on_mpl_key_press)

    # ---------------- updates ----------------

    def _update_all(self):
        self._update_image()
        self._update_histogram()
        i0 = self.scalar.shape[0] // 2
        j0 = self.scalar.shape[1] // 2
        self._set_cursor(i0, j0, update_status=False)
        self._update_cursor_profiles(i0, j0)

    def _update_image(self):
        self.ax_img.clear()
        self.ax_img.set_axis_off()

        if self.is_grayscale:
            self._im = self.ax_img.imshow(
                self.scalar,
                cmap=self._current_cmap,
                vmin=self._vmin,
                vmax=self._vmax,
                interpolation="nearest",
                origin="upper",
            )
        else:
            img = self.image
            if img.max() > 1.0:
                img = img / 255.0
            self._im = self.ax_img.imshow(_clip01(img), interpolation="nearest", origin="upper")

        self.ax_img.set_xlim(0, self.scalar.shape[1])
        self.ax_img.set_ylim(self.scalar.shape[0], 0)

        # Crosshair
        self._cross_v = self.ax_img.axvline(0, linestyle="--", linewidth=1, visible=False)
        self._cross_h = self.ax_img.axhline(0, linestyle="--", linewidth=1, visible=False)

        # Restore line profile overlay if present
        if self._line_profile.p1 and self._line_profile.p2:
            self._ensure_line_profile_artists()

        self.canvas_img.draw_idle()

    def _avg_k(self) -> int:
        return int(self.spin_avg.value())

    def _update_histogram(self):
        bins = int(self.spin_bins.value())
        data = self.scalar.ravel()

        self.ax_hist.clear()
        self.ax_hist.set_title("Histogram (scalar)")
        self.ax_hist.set_xlabel("Value")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(True, alpha=0.3)

        self.ax_hist.hist(data, bins=bins)
        if self.is_grayscale:
            self.ax_hist.axvline(self._vmin, color="red", linewidth=1.5)
            self.ax_hist.axvline(self._vmax, color="red", linewidth=1.5)

        self.canvas_hist.draw_idle()

    def _refresh_profiles(self):
        if self._last_cursor_ij:
            i, j = self._last_cursor_ij
            self._update_cursor_profiles(i, j)
        if self._line_profile.p1 and self._line_profile.p2 and self._line_profile.window:
            self._compute_and_update_line_profile()

    def _update_cursor_profiles(self, i: int, j: int):
        i = int(np.clip(i, 0, self.scalar.shape[0] - 1))
        j = int(np.clip(j, 0, self.scalar.shape[1] - 1))
        self._last_cursor_ij = (i, j)

        k = self._avg_k()

        i0 = max(0, i - k)
        i1 = min(self.scalar.shape[0] - 1, i + k)
        row = np.mean(self.scalar[i0:i1 + 1, :], axis=0).astype(np.float32, copy=False)
        x = np.arange(row.size, dtype=np.float32)

        j0 = max(0, j - k)
        j1 = min(self.scalar.shape[1] - 1, j + k)
        col = np.mean(self.scalar[:, j0:j1 + 1], axis=1).astype(np.float32, copy=False)
        y = np.arange(col.size, dtype=np.float32)

        self.ax_h.clear()
        ht = f"Horizontal profile @ y={i}"
        if k > 0:
            ht += f" (avg ±{k})"
        self.ax_h.set_title(ht)
        self.ax_h.set_xlabel("x (px)")
        self.ax_h.set_ylabel("Value")
        self.ax_h.grid(True, alpha=0.3)
        self.ax_h.plot(x, row)
        self.ax_h.axvline(j, linestyle="--", linewidth=1)
        self.canvas_h.draw_idle()

        self.ax_v.clear()
        vt = f"Vertical profile @ x={j}"
        if k > 0:
            vt += f" (avg ±{k})"
        self.ax_v.set_title(vt)
        self.ax_v.set_xlabel("Value")
        self.ax_v.set_ylabel("y (px)")
        self.ax_v.grid(True, alpha=0.3)
        self.ax_v.plot(col, y)
        self.ax_v.axhline(i, linestyle="--", linewidth=1)
        self.ax_v.invert_yaxis()
        self.canvas_v.draw_idle()

    # ---------------- crosshair + cursor ----------------

    def _set_cursor(self, i: int, j: int, update_status: bool = True):
        i = int(np.clip(i, 0, self.scalar.shape[0] - 1))
        j = int(np.clip(j, 0, self.scalar.shape[1] - 1))
        self._last_cursor_ij = (i, j)

        self._cross_v.set_xdata([j, j])
        self._cross_h.set_ydata([i, i])
        self._cross_v.set_visible(True)
        self._cross_h.set_visible(True)
        self.canvas_img.draw_idle()

        if update_status:
            val = self.scalar[i, j]
            self.status.showMessage(f"x={j}, y={i}, value={val:.6g}")

    def _on_leave(self, _event):
        self._cursor_in_image = False
        self._cross_v.set_visible(False)
        self._cross_h.set_visible(False)
        self.canvas_img.draw_idle()

    def _on_motion(self, event):
        if event.inaxes != self.ax_img:
            self._cursor_in_image = False
            return
        if event.xdata is None or event.ydata is None:
            return

        self._cursor_in_image = True
        self.canvas_img.setFocus()

        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if 0 <= i < self.scalar.shape[0] and 0 <= j < self.scalar.shape[1]:
            self._set_cursor(i, j, update_status=True)
            if self._last_cursor_ij != (i, j):
                self._update_cursor_profiles(i, j)

            if self._line_profile.dragging is not None:
                self._drag_endpoint_to(i, j)

    # ---------------- line profile overlay & dragging ----------------

    def _ensure_line_profile_artists(self):
        p1, p2 = self._line_profile.p1, self._line_profile.p2
        if p1 is None or p2 is None:
            return
        x1, y1 = p1
        x2, y2 = p2

        for a in (self._line_profile.line_artist, self._line_profile.p1_artist, self._line_profile.p2_artist):
            try:
                if a is not None:
                    a.remove()
            except Exception:
                pass

        (self._line_profile.line_artist,) = self.ax_img.plot([x1, x2], [y1, y2], "-", linewidth=2)
        (self._line_profile.p1_artist,) = self.ax_img.plot([x1], [y1], "o", markersize=8)
        (self._line_profile.p2_artist,) = self.ax_img.plot([x2], [y2], "o", markersize=8)
        self.canvas_img.draw_idle()

    def _nearest_endpoint(self, x: float, y: float, thresh_px: float = 10.0) -> Optional[str]:
        if self._line_profile.p1 is None or self._line_profile.p2 is None:
            return None
        x1, y1 = self._line_profile.p1
        x2, y2 = self._line_profile.p2
        d1 = (x - x1) ** 2 + (y - y1) ** 2
        d2 = (x - x2) ** 2 + (y - y2) ** 2
        if d1 <= thresh_px ** 2 and d1 <= d2:
            return "p1"
        if d2 <= thresh_px ** 2:
            return "p2"
        return None

    def _on_click(self, event):
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)

        ep = self._nearest_endpoint(x, y)
        if ep is not None:
            self._line_profile.dragging = ep
            return

        if self._line_profile.active:
            if self._line_profile.p1 is None:
                self._line_profile.p1 = (x, y)
                self.status.showMessage("Line profile: first point set. Click second point.")
            else:
                self._line_profile.p2 = (x, y)
                self._ensure_line_profile_artists()
                self._create_line_profile_window()
                self._compute_and_update_line_profile()
                self._line_profile.active = False
                self.act_line_profile.setChecked(False)
        else:
            self._set_cursor(int(round(y)), int(round(x)), update_status=True)
            self._update_cursor_profiles(int(round(y)), int(round(x)))

    def _on_release(self, _event):
        self._line_profile.dragging = None

    def _drag_endpoint_to(self, i: int, j: int):
        if self._line_profile.dragging is None:
            return
        p = (float(j), float(i))
        if self._line_profile.dragging == "p1":
            self._line_profile.p1 = p
        else:
            self._line_profile.p2 = p
        self._ensure_line_profile_artists()
        self._compute_and_update_line_profile()

    def _toggle_line_profile_mode(self, checked: bool):
        self._line_profile.active = bool(checked)
        if checked:
            self._line_profile.p1 = None
            self._line_profile.p2 = None
            self.status.showMessage("Line profile mode ON: click two points on the image.")
        else:
            self.status.showMessage("Line profile mode OFF.")

    def _clear_line_profile_overlay(self):
        self._line_profile = _LineProfileState()
        self.act_line_profile.setChecked(False)
        self._update_image()

    def _create_line_profile_window(self):
        if self._line_profile.window is None:
            w = LineProfileWindow("Line profile")
            w.resize(700, 450)
            w.show()
            self._line_profile.window = w

    def _compute_and_update_line_profile(self):
        if self._line_profile.p1 is None or self._line_profile.p2 is None or self._line_profile.window is None:
            return
        (x1, y1) = self._line_profile.p1
        (x2, y2) = self._line_profile.p2

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            return

        n = int(max(2, min(4096, np.ceil(length * 2))))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        xs = x1 + t * dx
        ys = y1 + t * dy

        k = self._avg_k()
        if k <= 0:
            prof = _bilinear_sample(self.scalar, xs, ys)
            title = "Line profile"
        else:
            nx = -dy / length
            ny = dx / length
            offs = np.arange(-k, k + 1, dtype=np.float32)
            stack = [_bilinear_sample(self.scalar, xs + o * nx, ys + o * ny) for o in offs]
            prof = np.mean(np.stack(stack, axis=0), axis=0)
            title = f"Line profile (avg ±{k})"

        dist = t * length
        self._line_profile.window.update_profile(dist, prof, title=title)

    # ---------------- keyboard arrows ----------------

    def _on_mpl_key_press(self, event):
        if event.key in ("up", "down", "left", "right"):
            self._handle_arrow(event.key)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key_map = {
            QtCore.Qt.Key.Key_Up: "up",
            QtCore.Qt.Key.Key_Down: "down",
            QtCore.Qt.Key.Key_Left: "left",
            QtCore.Qt.Key.Key_Right: "right",
        }
        if event.key() in key_map:
            self._handle_arrow(key_map[event.key()])
            return
        super().keyPressEvent(event)

    def _handle_arrow(self, direction: str):
        if not self._cursor_in_image or self._last_cursor_ij is None:
            return
        i, j = self._last_cursor_ij
        if direction == "up":
            i -= 1
        elif direction == "down":
            i += 1
        elif direction == "left":
            j -= 1
        elif direction == "right":
            j += 1
        i = int(np.clip(i, 0, self.scalar.shape[0] - 1))
        j = int(np.clip(j, 0, self.scalar.shape[1] - 1))
        self._set_cursor(i, j, update_status=True)
        self._update_cursor_profiles(i, j)

    # ---------------- display controls ----------------

    def _on_contrast_change(self):
        vmin_frac = self.slider_min.value() / 1000.0
        vmax_frac = self.slider_max.value() / 1000.0
        if vmin_frac >= vmax_frac:
            vmax_frac = min(vmin_frac + 0.001, 1.0)
            self.slider_max.blockSignals(True)
            self.slider_max.setValue(int(vmax_frac * 1000))
            self.slider_max.blockSignals(False)

        self._vmin = self._data_min + vmin_frac * (self._data_max - self._data_min)
        self._vmax = self._data_min + vmax_frac * (self._data_max - self._data_min)
        self.label_min.setText(f"Min: {self._vmin:.3g}")
        self.label_max.setText(f"Max: {self._vmax:.3g}")

        if self.is_grayscale:
            self._im.set_clim(self._vmin, self._vmax)
            self.canvas_img.draw_idle()
        self._update_histogram()

    def _on_cmap_change(self, text: str):
        self._current_cmap = text
        if self.is_grayscale:
            self._im.set_cmap(text)
            self.canvas_img.draw_idle()

    # ---------------- export ----------------

    def _export_view(self):
        filters = "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Export view", "", filters)
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            if selected.startswith("PNG"):
                path += ".png"
            elif selected.startswith("TIFF"):
                path += ".tif"
            elif selected.startswith("JPEG"):
                path += ".jpg"
            elif selected.startswith("BMP"):
                path += ".bmp"

        fig = Figure(figsize=(6, 6), dpi=150)
        _ = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        if self.is_grayscale:
            ax.imshow(self.scalar, cmap=self._current_cmap, vmin=self._vmin, vmax=self._vmax,
                      interpolation="nearest", origin="upper")
        else:
            img = self.image
            if img.max() > 1.0:
                img = img / 255.0
            ax.imshow(_clip01(img), interpolation="nearest", origin="upper")

        try:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
            self.status.showMessage(f"Exported view to: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{e}")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def _run_gui(image_path: str, title: Optional[str] = None):
    img = _load_image_from_path(image_path)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    viewer = ImageViewer(img, title=title or os.path.basename(image_path))
    viewer.resize(1300, 900)
    viewer.show()

    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(description="Standalone PyQt6 image viewer app.")
    parser.add_argument("image_path", help="Path to image file or .npy array.")
    parser.add_argument("--title", default=None, help="Optional window title.")
    args = parser.parse_args()
    _run_gui(args.image_path, title=args.title)


# ---------------------------------------------------------------------
# Public API: launch_image_viewer (spawns separate process)
# ---------------------------------------------------------------------

def launch_image_viewer(image: np.ndarray, title: str = "Image Viewer") -> subprocess.Popen:
    """
    Launch the image viewer in a separate Python process (non-blocking).

    Returns
    -------
    subprocess.Popen
        Handle to the spawned process.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    tmp_dir = tempfile.gettempdir()
    tmp_name = next(tempfile._get_candidate_names()) + ".npy"
    tmp_path = os.path.join(tmp_dir, tmp_name)
    np.save(tmp_path, image)

    cmd = [sys.executable, os.path.abspath(__file__), tmp_path, "--title", title]
    return subprocess.Popen(cmd, start_new_session=True)


if __name__ == "__main__":
    main()
