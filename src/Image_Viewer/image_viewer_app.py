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
c) Live X/Y profiles at cursor position:
   - Vertical profile shown on the RIGHT
   - Horizontal profile shown BELOW
d) Image export: PNG, TIFF, BMP, JPEG (exports the current view "as shown")
e) PyQt6-only enforcement

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

From command line:
    python image_viewer_app.py path/to/image.png --title "My image"
    python image_viewer_app.py path/to/array.npy --title "My array"
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
import matplotlib

# Qt backend (works with PyQt6)
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# ---- PyQt6-only enforcement ----
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
except Exception as e:
    raise ImportError(
        "PyQt6 is required for this viewer. Install with:\n"
        "  conda install -c conda-forge pyqt\n"
        "or\n"
        "  pip install PyQt6\n"
    ) from e


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

def _qt_orientation_vertical() -> QtCore.Qt.Orientation:
    return QtCore.Qt.Orientation.Vertical


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


# ---------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------

@dataclass
class _LineProfileState:
    active: bool = False
    p1: Optional[Tuple[float, float]] = None  # (x,y) in image coords
    p2: Optional[Tuple[float, float]] = None
    artist: Optional[object] = None  # matplotlib Line2D


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

        # Display controls (for grayscale rendering)
        self._vmin = self._data_min
        self._vmax = self._data_max
        self._current_cmap = "gray"

        # Live cursor state
        self._last_cursor_ij: Optional[Tuple[int, int]] = None

        # Line profile tool state
        self._line_profile = _LineProfileState()

        self._profile_windows: list[PlotWindow] = []

        self._build_ui()
        self._connect_events()
        self._update_all()

    # ---------------- UI ----------------

    def _build_ui(self):
        # Menu
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

        # Central layout:
        #   [ image + vertical profile ] stacked over [ horizontal profile ]
        central = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)

        # --- Image canvas ---
        self.fig_img = Figure(figsize=(6, 5), dpi=100)
        self.canvas_img = FigureCanvasQTAgg(self.fig_img)
        self.ax_img = self.fig_img.add_subplot(111)
        self.ax_img.set_axis_off()
        self.toolbar = NavigationToolbar2QT(self.canvas_img, self)

        img_box = QtWidgets.QVBoxLayout()
        img_box.addWidget(self.toolbar)
        img_box.addWidget(self.canvas_img, stretch=1)

        top_row.addLayout(img_box, stretch=3)

        # --- Vertical profile canvas (right) ---
        self.fig_v = Figure(figsize=(3, 5), dpi=100)
        self.canvas_v = FigureCanvasQTAgg(self.fig_v)
        self.ax_v = self.fig_v.add_subplot(111)
        self.ax_v.set_title("Vertical profile")
        self.ax_v.set_xlabel("Value")
        self.ax_v.set_ylabel("y (px)")
        self.ax_v.grid(True, alpha=0.3)

        top_row.addWidget(self.canvas_v, stretch=1)

        outer.addLayout(top_row, stretch=3)

        # --- Horizontal profile canvas (bottom) ---
        self.fig_h = Figure(figsize=(8, 2.5), dpi=100)
        self.canvas_h = FigureCanvasQTAgg(self.fig_h)
        self.ax_h = self.fig_h.add_subplot(111)
        self.ax_h.set_title("Horizontal profile")
        self.ax_h.set_xlabel("x (px)")
        self.ax_h.set_ylabel("Value")
        self.ax_h.grid(True, alpha=0.3)

        outer.addWidget(self.canvas_h, stretch=1)

        # --- Controls (contrast + colormap) ---
        controls = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Contrast group applies to scalar/grayscale display; for RGB we still allow export/profiles/hist.
        contrast_group = QtWidgets.QGroupBox("Display (scalar channel)")
        gl = QtWidgets.QGridLayout(contrast_group)

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

        controls_layout.addWidget(contrast_group)
        controls_layout.addStretch(1)
        outer.addWidget(controls, stretch=0)

        self.setCentralWidget(central)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # --- Histogram dock ---
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

        # Optional: bins control
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

        self.cid_motion = self.canvas_img.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_click = self.canvas_img.mpl_connect("button_press_event", self._on_click)

    # ---------------- updates ----------------

    def _update_all(self):
        self._update_image()
        self._update_histogram()
        # initialize profiles with center pixel
        i0 = self.scalar.shape[0] // 2
        j0 = self.scalar.shape[1] // 2
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
            # display RGB as is (scaled to 0..1 if needed)
            img = self.image
            if img.max() > 1.0:
                img_disp = img / 255.0
            else:
                img_disp = img
            img_disp = _clip01(img_disp)
            self._im = self.ax_img.imshow(
                img_disp,
                interpolation="nearest",
                origin="upper",
            )

        self.ax_img.set_xlim(0, self.scalar.shape[1])
        self.ax_img.set_ylim(self.scalar.shape[0], 0)

        # If line overlay exists, redraw it on top
        if self._line_profile.artist is not None and self._line_profile.p1 and self._line_profile.p2:
            (x1, y1) = self._line_profile.p1
            (x2, y2) = self._line_profile.p2
            self._line_profile.artist, = self.ax_img.plot([x1, x2], [y1, y2], "-", linewidth=2)

        self.canvas_img.draw_idle()

    def _update_histogram(self):
        bins = int(self.spin_bins.value())
        data = self.scalar.ravel()

        # If contrast window is active, optionally show both full and windowed
        self.ax_hist.clear()
        self.ax_hist.set_title("Histogram (scalar)")
        self.ax_hist.set_xlabel("Value")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(True, alpha=0.3)

        self.ax_hist.hist(data, bins=bins)
        # Mark display window for grayscale
        if self.is_grayscale:
            self.ax_hist.axvline(self._vmin)
            self.ax_hist.axvline(self._vmax)

        self.canvas_hist.draw_idle()

    def _update_cursor_profiles(self, i: int, j: int):
        i = int(np.clip(i, 0, self.scalar.shape[0] - 1))
        j = int(np.clip(j, 0, self.scalar.shape[1] - 1))
        self._last_cursor_ij = (i, j)

        # Horizontal profile: row i
        row = self.scalar[i, :].astype(np.float32, copy=False)
        x = np.arange(row.size, dtype=np.float32)

        # Vertical profile: column j
        col = self.scalar[:, j].astype(np.float32, copy=False)
        y = np.arange(col.size, dtype=np.float32)

        # Update horizontal plot
        self.ax_h.clear()
        self.ax_h.set_title(f"Horizontal profile @ y={i}")
        self.ax_h.set_xlabel("x (px)")
        self.ax_h.set_ylabel("Value")
        self.ax_h.grid(True, alpha=0.3)
        self.ax_h.plot(x, row)
        self.ax_h.axvline(j, linestyle="--", linewidth=1)
        self.canvas_h.draw_idle()

        # Update vertical plot (value vs y)
        self.ax_v.clear()
        self.ax_v.set_title(f"Vertical profile @ x={j}")
        self.ax_v.set_xlabel("Value")
        self.ax_v.set_ylabel("y (px)")
        self.ax_v.grid(True, alpha=0.3)
        self.ax_v.plot(col, y)
        self.ax_v.axhline(i, linestyle="--", linewidth=1)
        # keep y increasing downward like image coordinates
        self.ax_v.invert_yaxis()
        self.canvas_v.draw_idle()

    # ---------------- callbacks ----------------

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
            try:
                self._im.set_clim(self._vmin, self._vmax)
            except Exception:
                pass
            self.canvas_img.draw_idle()

        self._update_histogram()

    def _on_cmap_change(self, text: str):
        self._current_cmap = text
        if self.is_grayscale and hasattr(self, "_im"):
            self._im.set_cmap(self._current_cmap)
            self.canvas_img.draw_idle()

    def _on_motion(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if i < 0 or j < 0 or i >= self.scalar.shape[0] or j >= self.scalar.shape[1]:
            return

        # Status message
        val = self.scalar[i, j]
        self.status.showMessage(f"x={j}, y={i}, value={val:.6g}")

        # Live profiles (throttle: only update if pixel changed)
        if self._last_cursor_ij != (i, j):
            self._update_cursor_profiles(i, j)

    def _on_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # If line profile mode is enabled, capture points
        if self._line_profile.active:
            if self._line_profile.p1 is None:
                self._line_profile.p1 = (x, y)
                self.status.showMessage("Line profile: first point set. Click second point.")
            else:
                self._line_profile.p2 = (x, y)
                self.status.showMessage("Line profile: second point set. Computing profile...")
                self._compute_and_show_line_profile()
                # keep overlay, but exit mode
                self._line_profile.active = False
                self.act_line_profile.setChecked(False)

    def _toggle_line_profile_mode(self, checked: bool):
        self._line_profile.active = bool(checked)
        if checked:
            self._line_profile.p1 = None
            self._line_profile.p2 = None
            self.status.showMessage("Line profile mode ON: click two points on the image.")
        else:
            self.status.showMessage("Line profile mode OFF.")

    def _clear_line_profile_overlay(self):
        self._line_profile.p1 = None
        self._line_profile.p2 = None
        self._line_profile.active = False
        self.act_line_profile.setChecked(False)
        self._line_profile.artist = None
        self._update_image()

    def _compute_and_show_line_profile(self):
        if self._line_profile.p1 is None or self._line_profile.p2 is None:
            return
        (x1, y1) = self._line_profile.p1
        (x2, y2) = self._line_profile.p2

        # Draw overlay
        try:
            if self._line_profile.artist is not None:
                self._line_profile.artist.remove()
        except Exception:
            pass
        self._line_profile.artist, = self.ax_img.plot([x1, x2], [y1, y2], "-", linewidth=2)
        self.canvas_img.draw_idle()

        # Sample along the segment
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 1e-9:
            return

        n = int(max(2, min(4096, np.ceil(length * 2))))  # ~2 samples per pixel
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        xs = x1 + t * dx
        ys = y1 + t * dy

        prof = _bilinear_sample(self.scalar, xs, ys)
        dist = t * length

        w = PlotWindow("Line profile")
        w.plot_line(dist, prof, xlabel="Distance (px)", ylabel="Value")
        w.resize(700, 450)
        w.show()

        # Keep reference to prevent GC closing it
        self._profile_windows.append(w)
        self.status.showMessage(f"Line profile computed: {n} samples, length={length:.2f}px")

    # ---------------- export ----------------

    def _export_view(self):
        filters = "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export view",
            "",
            filters,
        )
        if not path:
            return

        # Infer extension if missing
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

        try:
            # Export "as shown": the image axes only
            # Create a temporary figure to avoid including toolbars/docks
            fig = Figure(figsize=(6, 6), dpi=150)
            canvas = FigureCanvasQTAgg(fig)  # needed by some backends before savefig
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            if self.is_grayscale:
                ax.imshow(
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
                ax.imshow(_clip01(img), interpolation="nearest", origin="upper")

            fig.savefig(path, bbox_inches="tight", pad_inches=0.0)
            self.status.showMessage(f"Exported view to: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Export failed",
                f"Could not export image.\n\nPath: {path}\n\nError:\n{e}",
            )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def _run_gui(image_path: str, title: Optional[str] = None):
    img = _load_image_from_path(image_path)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    viewer = ImageViewer(img, title=title or os.path.basename(image_path))
    viewer.resize(1200, 850)
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

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        tmp_path,
        "--title",
        title,
    ]

    # Detach reasonably well across platforms
    return subprocess.Popen(cmd, start_new_session=True)


if __name__ == "__main__":
    main()
