import traceback

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

from PySide6.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QProgressBar, QSizePolicy,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool

from orbit.image import QPTiffImage
from orbit.fov import RandomFOVGenerator


COLOR_MAPS = {
    "Gray": (1, 1, 1), "Red": (1, 0, 0), "Green": (0, 1, 0),
    "Blue": (0, 0, 1), "Cyan": (0, 1, 1),
    "Magenta": (1, 0, 1), "Yellow": (1, 1, 0),
}


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    low, high = np.percentile(arr, (1, 99))
    arr = np.clip(arr, low, high)
    return ((arr - low) / (high - low + 1e-8) * 255).astype(np.uint8)


def array_to_qpixmap(
    marker_arr: np.ndarray,
    marker_color: str = "Green",
    dapi_arr: np.ndarray | None = None,
    show_dapi: bool = True,
    segmentation_boundary: np.ndarray | None = None,
) -> QPixmap:
    marker = normalize_channel(marker_arr)
    r_scale, g_scale, b_scale = COLOR_MAPS[marker_color]
    rgb = np.zeros((*marker.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.maximum(rgb[:, :, 0], marker * r_scale)
    rgb[:, :, 1] = np.maximum(rgb[:, :, 1], marker * g_scale)
    rgb[:, :, 2] = np.maximum(rgb[:, :, 2], marker * b_scale)

    if show_dapi and dapi_arr is not None:
        rgb[:, :, 2] = np.maximum(rgb[:, :, 2], normalize_channel(dapi_arr))

    if segmentation_boundary is not None:
        boundary = segmentation_boundary.astype(bool)
        rgb[boundary, 0] = 255
        rgb[boundary, 1] = 0
        rgb[boundary, 2] = 0

    h, w, _ = rgb.shape
    qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class WorkerSignals(QObject):
    finished = Signal(object, object)
    error = Signal(str)


class FOVLoadWorker(QRunnable):
    def __init__(self, fov_generator, y0, x0, size, channel, dapi_channel=0):
        super().__init__()
        self.fov_generator = fov_generator
        self.y0, self.x0, self.size = y0, x0, size
        self.channel, self.dapi_channel = channel, dapi_channel
        self.signals = WorkerSignals()

    def run(self):
        try:
            marker_fov = self.fov_generator.get_fov(
                y0=self.y0, x0=self.x0, size=self.size, channel=self.channel
            )
            dapi_fov = self.fov_generator.get_fov(
                y0=self.y0, x0=self.x0, size=self.size,
                channel=self.dapi_channel,
            )
            self.signals.finished.emit(marker_fov, dapi_fov)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class OrbitFOVViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ORBIT Random FOV Viewer")

        self.img = None
        self.fov_generator = None
        self.thread_pool = QThreadPool.globalInstance()
        self.fov_size = 512
        self.current_y0 = self.current_x0 = None
        self.current_fov = self.current_dapi_fov = None
        self.current_pixmap = None
        self.is_loading = False

        # Segmentation data remain in whole-slide pixel coordinates. Only the
        # current FOV is cropped and converted to a boundary overlay.
        self.cell_data = None
        self.segmentation_masks = None
        self.cell_data_path = None
        self.segmentation_mask_path = None

        self.image_label = QLabel("Select a QPTIFF image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(700, 700)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("""
            QLabel { background-color: black; border: 1px solid #333;
                     color: white; font-size: 16px; }
        """)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.spinner = QProgressBar()
        self.spinner.setRange(0, 0)
        self.spinner.setTextVisible(False)
        self.spinner.setMaximumHeight(8)
        self.spinner.hide()

        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_qptiff)
        self.load_segmentation_button = QPushButton("Load Segmentation")
        self.load_segmentation_button.clicked.connect(self.load_segmentation)
        self.load_segmentation_button.setEnabled(False)
        self.generate_button = QPushButton("Generate FOV")
        self.generate_button.clicked.connect(self.generate_fov)
        self.generate_button.setEnabled(False)
        self.regenerate_button = QPushButton("Regenerate")
        self.regenerate_button.clicked.connect(self.generate_fov)
        self.regenerate_button.setEnabled(False)

        self.channel_dropdown = QComboBox()
        self.channel_dropdown.currentIndexChanged.connect(self.reload_current_fov)
        self.channel_dropdown.setEnabled(False)
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(COLOR_MAPS.keys())
        self.color_dropdown.setCurrentText("Green")
        self.color_dropdown.currentTextChanged.connect(self.update_display)
        self.color_dropdown.setEnabled(False)
        self.dapi_checkbox = QCheckBox("DAPI")
        self.dapi_checkbox.setChecked(True)
        self.dapi_checkbox.stateChanged.connect(self.update_display)
        self.dapi_checkbox.setEnabled(False)
        self.segmentation_checkbox = QCheckBox("Segmentation")
        self.segmentation_checkbox.setChecked(True)
        self.segmentation_checkbox.stateChanged.connect(self.update_display)
        self.segmentation_checkbox.setEnabled(False)

        toolbar = QHBoxLayout()
        for widget in (
            self.open_button, self.load_segmentation_button,
            self.generate_button, self.regenerate_button,
        ):
            toolbar.addWidget(widget)
        toolbar.addSpacing(20)
        for widget in (
            self.channel_dropdown, self.color_dropdown, self.dapi_checkbox,
            self.segmentation_checkbox,
        ):
            toolbar.addWidget(widget)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.spinner)
        layout.addWidget(self.status_label)
        layout.addLayout(toolbar)
        self.setLayout(layout)
        self.resize(1200, 1000)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            self.display_pixmap()

    def display_pixmap(self):
        if self.current_pixmap is None:
            return
        self.image_label.setPixmap(self.current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def set_loading(self, loading: bool, message: str = ""):
        self.is_loading = loading
        self.status_label.setText(message)
        self.spinner.setVisible(loading)
        self.open_button.setEnabled(not loading)
        self.load_segmentation_button.setEnabled(not loading and self.img is not None)
        self.generate_button.setEnabled(not loading and self.img is not None)
        self.regenerate_button.setEnabled(not loading and self.current_y0 is not None)
        self.channel_dropdown.setEnabled(not loading and self.img is not None)
        self.color_dropdown.setEnabled(not loading and self.img is not None)
        self.dapi_checkbox.setEnabled(not loading and self.img is not None)
        self.segmentation_checkbox.setEnabled(
            not loading and self.segmentation_masks is not None
        )

    def open_qptiff(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select QPTIFF image", "", "QPTIFF files (*.qptiff *.tif *.tiff)"
        )
        if not path:
            return
        self.set_loading(True, "Loading QPTIFF metadata...")
        try:
            self.img = QPTiffImage(path)
            self.fov_generator = RandomFOVGenerator(self.img)
            self.channel_dropdown.blockSignals(True)
            self.channel_dropdown.clear()
            self.channel_dropdown.addItems(self.img.get_channel_names())
            self.channel_dropdown.blockSignals(False)
            self.current_y0 = self.current_x0 = None
            self.current_fov = self.current_dapi_fov = None
            self.current_pixmap = None
            self.cell_data = self.segmentation_masks = None
            self.segmentation_checkbox.setEnabled(False)
            self.image_label.clear()
            self.image_label.setText("Image loaded.\n\nClick 'Generate FOV'.")
            self.status_label.setText("")
        except Exception:
            self.img = self.fov_generator = None
            self.image_label.setText("Failed to load QPTIFF.")
            self.status_label.setText(traceback.format_exc())
        finally:
            self.set_loading(False)

    def load_segmentation(self):
        if self.img is None:
            return
        cell_path, _ = QFileDialog.getOpenFileName(
            self, "Select cell data", "",
            "Cell data (*.tsv *.txt *.csv);;All files (*)",
        )
        if not cell_path:
            return
        mask_path, _ = QFileDialog.getOpenFileName(
            self, "Select annotation mask", "",
            "TIFF masks (*.tif *.tiff);;All files (*)",
        )
        if not mask_path:
            return

        self.set_loading(True, "Loading segmentation...")
        try:
            separator = "," if cell_path.lower().endswith(".csv") else "\t"
            cell_data = pd.read_csv(cell_path, sep=separator)
            if cell_data.empty:
                raise ValueError("The selected cell-data file contains no rows.")

            masks = np.squeeze(tifffile.imread(mask_path))
            if masks.ndim != 2:
                raise ValueError(
                    f"Expected a two-dimensional annotation mask; got {masks.shape}."
                )

            self.cell_data = cell_data
            self.segmentation_masks = masks
            self.cell_data_path = cell_path
            self.segmentation_mask_path = mask_path
            self.segmentation_checkbox.setChecked(True)
            self.status_label.setText(
                f"Loaded {len(cell_data):,} cells and a "
                f"{masks.shape[1]} x {masks.shape[0]} mask."
            )
            self.update_display()
        except Exception:
            self.status_label.setText(traceback.format_exc())
        finally:
            self.set_loading(False, self.status_label.text())

    def generate_fov(self):
        if self.fov_generator is None:
            return
        self.current_y0, self.current_x0 = self.fov_generator.random_position(
            size=self.fov_size, seed=None
        )
        self.reload_current_fov()

    def reload_current_fov(self):
        if self.is_loading or self.current_y0 is None:
            return
        self.set_loading(True, "Loading field of view...")
        worker = FOVLoadWorker(
            self.fov_generator, self.current_y0, self.current_x0,
            self.fov_size, self.channel_dropdown.currentIndex(), 0,
        )
        worker.signals.finished.connect(self.on_fov_loaded)
        worker.signals.error.connect(self.on_fov_error)
        self.thread_pool.start(worker)

    def on_fov_loaded(self, marker_fov, dapi_fov):
        self.current_fov, self.current_dapi_fov = marker_fov, dapi_fov
        self.set_loading(False)
        self.regenerate_button.setEnabled(True)
        self.update_display()

    def on_fov_error(self, error_message: str):
        self.set_loading(False)
        self.image_label.setText("Failed to generate FOV.")
        self.status_label.setText(error_message)

    def current_segmentation_boundary(self):
        if self.segmentation_masks is None or self.current_y0 is None:
            return None
        y0, x0 = int(self.current_y0), int(self.current_x0)
        height, width = self.current_fov.shape[:2]
        y1, x1 = y0 + height, x0 + width
        mask_height, mask_width = self.segmentation_masks.shape
        if y0 < 0 or x0 < 0 or y1 > mask_height or x1 > mask_width:
            raise ValueError(
                "The segmentation mask does not cover the current image field "
                f"({mask_width} x {mask_height} mask; requested x={x0}:{x1}, "
                f"y={y0}:{y1})."
            )
        mask_fov = self.segmentation_masks[y0:y1, x0:x1]
        return find_boundaries(mask_fov, connectivity=1, mode="inner")

    def update_display(self):
        if self.current_fov is None:
            return
        try:
            boundary = None
            if self.segmentation_checkbox.isChecked():
                boundary = self.current_segmentation_boundary()
            self.current_pixmap = array_to_qpixmap(
                marker_arr=self.current_fov,
                marker_color=self.color_dropdown.currentText(),
                dapi_arr=self.current_dapi_fov,
                show_dapi=self.dapi_checkbox.isChecked(),
                segmentation_boundary=boundary,
            )
            self.display_pixmap()
        except Exception:
            self.status_label.setText(traceback.format_exc())
