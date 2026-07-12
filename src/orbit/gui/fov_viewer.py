import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import joblib
from skimage.segmentation import find_boundaries
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from PySide6.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QProgressBar, QSizePolicy, QLineEdit, QGroupBox,
    QFormLayout, QMessageBox, QMenuBar,
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QAction
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool

from orbit.image import QPTiffImage
from orbit.fov import RandomFOVGenerator


COLOR_MAPS = {
    "Gray": (1, 1, 1), "Red": (1, 0, 0), "Green": (0, 1, 0),
    "Blue": (0, 0, 1), "Cyan": (0, 1, 1),
    "Magenta": (1, 0, 1), "Yellow": (1, 1, 0),
}

DEFAULT_PIXEL_SIZE_UM = 0.5055
MODEL_FORMAT = "ORBIT phenotype model"
MODEL_VERSION = 1


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
    annotation_markers: list[dict] | None = None,
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
    pixmap = QPixmap.fromImage(qimage.copy())

    if annotation_markers:
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        for marker in annotation_markers:
            radius = 3 if marker.get("source") == "model" else 5
            color = QColor("#00e640" if marker["label"] == "positive" else "#ff3030")
            painter.setPen(QPen(QColor("black"), 1))
            painter.setBrush(color)
            x, y = round(marker["x"]), round(marker["y"])
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()

    return pixmap


class ClickableImageLabel(QLabel):
    """QLabel that reports clicks as fractions of the displayed pixmap."""

    image_clicked = Signal(float, float)

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if event.button() == Qt.LeftButton and pixmap is not None:
            left = (self.width() - pixmap.width()) / 2
            top = (self.height() - pixmap.height()) / 2
            x = event.position().x() - left
            y = event.position().y() - top
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                self.image_clicked.emit(x / pixmap.width(), y / pixmap.height())
        super().mousePressEvent(event)


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
        self.image_path = None
        self.project_path = None
        self.annotations = {}
        self.training_navigation_indices = {"positive": -1, "negative": -1}
        self.loaded_images = []
        self.current_image_index = -1
        self.model_bundle = None

        # Segmentation data remain in whole-slide pixel coordinates. Only the
        # current FOV is cropped and converted to a boundary overlay.
        self.cell_data = None
        self.segmentation_masks = None
        self.cell_data_path = None
        self.segmentation_mask_path = None

        self.image_label = ClickableImageLabel("Select a QPTIFF image")
        self.image_label.image_clicked.connect(self.label_clicked_cell)
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

        self.open_button = QPushButton("Add Image")
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

        self.phenotype_name = QLineEdit()
        self.phenotype_name.setPlaceholderText("e.g. CD8-positive")
        self.positive_annotations_checkbox = QCheckBox("Show Positive")
        self.positive_annotations_checkbox.setChecked(True)
        self.positive_annotations_checkbox.setStyleSheet("color: #00b832;")
        self.positive_annotations_checkbox.stateChanged.connect(self.update_display)
        self.negative_annotations_checkbox = QCheckBox("Show Negative")
        self.negative_annotations_checkbox.setChecked(True)
        self.negative_annotations_checkbox.setStyleSheet("color: #e02020;")
        self.negative_annotations_checkbox.stateChanged.connect(self.update_display)
        self.positive_count_label = QLabel("Positive: 0")
        self.negative_count_label = QLabel("Negative: 0")
        self.previous_positive_button = QPushButton("←")
        self.previous_positive_button.setToolTip("Previous positive training cell")
        self.previous_positive_button.setFixedWidth(36)
        self.previous_positive_button.clicked.connect(
            lambda: self.navigate_training("positive", -1)
        )
        self.next_positive_button = QPushButton("→")
        self.next_positive_button.setToolTip("Next positive training cell")
        self.next_positive_button.setFixedWidth(36)
        self.next_positive_button.clicked.connect(
            lambda: self.navigate_training("positive", 1)
        )
        self.positive_position_label = QLabel("0 / 0")
        self.positive_position_label.setAlignment(Qt.AlignCenter)
        self.previous_negative_button = QPushButton("←")
        self.previous_negative_button.setToolTip("Previous negative training cell")
        self.previous_negative_button.setFixedWidth(36)
        self.previous_negative_button.clicked.connect(
            lambda: self.navigate_training("negative", -1)
        )
        self.next_negative_button = QPushButton("→")
        self.next_negative_button.setToolTip("Next negative training cell")
        self.next_negative_button.setFixedWidth(36)
        self.next_negative_button.clicked.connect(
            lambda: self.navigate_training("negative", 1)
        )
        self.negative_position_label = QLabel("0 / 0")
        self.negative_position_label.setAlignment(Qt.AlignCenter)

        positive_navigation_layout = QHBoxLayout()
        positive_navigation_layout.addWidget(self.previous_positive_button)
        positive_navigation_layout.addWidget(self.positive_position_label, stretch=1)
        positive_navigation_layout.addWidget(self.next_positive_button)
        negative_navigation_layout = QHBoxLayout()
        negative_navigation_layout.addWidget(self.previous_negative_button)
        negative_navigation_layout.addWidget(self.negative_position_label, stretch=1)
        negative_navigation_layout.addWidget(self.next_negative_button)

        training_panel = QGroupBox("Phenotype Training")
        training_panel.setMinimumWidth(220)
        training_panel.setMaximumWidth(300)
        training_layout = QVBoxLayout()
        name_layout = QFormLayout()
        name_layout.addRow("Phenotype:", self.phenotype_name)
        training_layout.addLayout(name_layout)
        training_layout.addWidget(self.positive_annotations_checkbox)
        training_layout.addWidget(self.negative_annotations_checkbox)
        training_layout.addSpacing(10)
        training_layout.addWidget(self.positive_count_label)
        training_layout.addLayout(positive_navigation_layout)
        training_layout.addWidget(self.negative_count_label)
        training_layout.addLayout(negative_navigation_layout)
        training_layout.addStretch()
        training_panel.setLayout(training_layout)

        self.model_status_label = QLabel("No model trained or loaded.")
        self.model_status_label.setWordWrap(True)
        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self.train_model)
        self.apply_model_button = QPushButton("Apply to Loaded Images")
        self.apply_model_button.clicked.connect(self.apply_model)
        self.modelled_phenotypes_checkbox = QCheckBox(
            "Show Modelled Phenotypes"
        )
        self.modelled_phenotypes_checkbox.setChecked(True)
        self.modelled_phenotypes_checkbox.stateChanged.connect(self.update_display)
        self.model_positive_count_label = QLabel("Model positive: 0")
        self.model_negative_count_label = QLabel("Model negative: 0")
        self.export_cell_phenotypes_button = QPushButton(
            "Export Cell Phenotypes"
        )
        self.export_cell_phenotypes_button.setToolTip(
            "Export every original Cellpose TSV column and the "
            "Positive/Negative phenotype labels for every loaded image"
        )
        self.export_cell_phenotypes_button.clicked.connect(
            self.export_cell_phenotypes
        )

        model_panel = QGroupBox("Machine-Learning Model")
        model_panel.setMinimumWidth(220)
        model_panel.setMaximumWidth(300)
        model_layout = QVBoxLayout()
        model_layout.addWidget(self.model_status_label)
        model_layout.addWidget(self.train_model_button)
        model_layout.addWidget(self.apply_model_button)
        model_layout.addWidget(self.modelled_phenotypes_checkbox)
        model_layout.addWidget(self.model_positive_count_label)
        model_layout.addWidget(self.model_negative_count_label)
        model_layout.addStretch()
        model_layout.addWidget(self.export_cell_phenotypes_button)
        model_panel.setLayout(model_layout)

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

        self.menu_bar = QMenuBar()
        file_menu = self.menu_bar.addMenu("&File")
        self.new_project_action = QAction("&New Project", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action = QAction("&Open...", self)
        self.open_project_action.setShortcut("Ctrl+O")
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action = QAction("&Save", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action = QAction("Save &As...", self)
        self.save_project_as_action.setShortcut("Ctrl+Shift+S")
        self.save_project_as_action.triggered.connect(self.save_project_as)
        self.import_model_action = QAction("&Import Model...", self)
        self.import_model_action.triggered.connect(self.import_model)
        self.export_model_action = QAction("&Export Model...", self)
        self.export_model_action.triggered.connect(self.export_model)
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_model_action)
        file_menu.addAction(self.export_model_action)
        layout.setMenuBar(self.menu_bar)

        viewer_layout = QHBoxLayout()
        viewer_layout.addWidget(self.image_label, stretch=1)
        right_panel = QVBoxLayout()
        right_panel.addWidget(training_panel)
        right_panel.addWidget(model_panel)
        viewer_layout.addLayout(right_panel)
        layout.addLayout(viewer_layout, stretch=1)
        layout.addWidget(self.spinner)
        layout.addWidget(self.status_label)
        layout.addLayout(toolbar)

        carousel = QGroupBox("Loaded Images")
        carousel_layout = QHBoxLayout()
        self.previous_image_button = QPushButton("←")
        self.previous_image_button.setToolTip("Previous loaded image")
        self.previous_image_button.setFixedWidth(44)
        self.previous_image_button.clicked.connect(lambda: self.cycle_image(-1))
        self.image_carousel = QComboBox()
        self.image_carousel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_carousel.currentIndexChanged.connect(self.switch_image)
        self.next_image_button = QPushButton("→")
        self.next_image_button.setToolTip("Next loaded image")
        self.next_image_button.setFixedWidth(44)
        self.next_image_button.clicked.connect(lambda: self.cycle_image(1))
        carousel_layout.addWidget(self.previous_image_button)
        carousel_layout.addWidget(self.image_carousel, stretch=1)
        carousel_layout.addWidget(self.next_image_button)
        carousel.setLayout(carousel_layout)
        layout.addWidget(carousel)

        self.setLayout(layout)
        self.resize(1200, 1000)
        self.update_training_navigation_controls()
        self.update_image_carousel_controls()
        self.update_model_controls()

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
        self.update_training_navigation_controls()
        self.update_image_carousel_controls()
        self.update_model_controls()

    def open_qptiff(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Select QPTIFF image", "", "QPTIFF files (*.qptiff *.tif *.tiff)"
        )
        if not image_path:
            return
        cell_path, _ = QFileDialog.getOpenFileName(
            self, "Select corresponding cell data", "",
            "Cell data (*.tsv *.txt *.csv);;All files (*)",
        )
        if not cell_path:
            return
        mask_path, _ = QFileDialog.getOpenFileName(
            self, "Select corresponding annotation mask", "",
            "TIFF masks (*.tif *.tiff);;All files (*)",
        )
        if not mask_path:
            return

        self.set_loading(True, "Loading image and segmentation...")
        try:
            state = self._create_image_state(image_path, cell_path, mask_path)
            self._capture_current_image_state()
            self.loaded_images.append(state)
            self._refresh_image_carousel()
            self._activate_image(len(self.loaded_images) - 1)
            self.status_label.setText(
                f"Loaded {Path(state['image_path']).name} "
                f"({len(self.loaded_images)} image(s) in the carousel)."
            )
        except Exception:
            self.status_label.setText(traceback.format_exc())
        finally:
            self.set_loading(False, self.status_label.text())

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
            cell_data, masks, cell_path, mask_path = self._read_segmentation(
                cell_path, mask_path, self.img
            )
            self.cell_data = cell_data
            self.segmentation_masks = masks
            self.cell_data_path = cell_path
            self.segmentation_mask_path = mask_path
            self.annotations.clear()
            self.training_navigation_indices = {"positive": -1, "negative": -1}
            self.loaded_images[self.current_image_index]["model_predictions"] = None
            self.loaded_images[self.current_image_index]["centroid_cache"] = None
            self._capture_current_image_state()
            self.update_annotation_counts()
            self.segmentation_checkbox.setChecked(True)
            self.status_label.setText(
                f"Loaded {len(self.cell_data):,} cells and a "
                f"{self.segmentation_masks.shape[1]} x "
                f"{self.segmentation_masks.shape[0]} mask."
            )
            self.update_display()
        except Exception:
            self.status_label.setText(traceback.format_exc())
        finally:
            self.set_loading(False, self.status_label.text())

    def _create_image_state(self, image_path, cell_path=None, mask_path=None):
        image_path = str(Path(image_path).expanduser().resolve())
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = QPTiffImage(image_path)
        if cell_path or mask_path:
            if not cell_path or not mask_path:
                raise ValueError("Both cell data and annotation mask paths are required.")
            cell_data, masks, cell_path, mask_path = self._read_segmentation(
                cell_path, mask_path, image
            )
        else:
            cell_data = masks = cell_path = mask_path = None
        return {
            "image_path": image_path,
            "cell_data_path": cell_path,
            "segmentation_mask_path": mask_path,
            "img": image,
            "fov_generator": RandomFOVGenerator(image),
            "cell_data": cell_data,
            "segmentation_masks": masks,
            "annotations": {},
            "current_y0": None,
            "current_x0": None,
            "current_fov": None,
            "current_dapi_fov": None,
            "channel_index": 0,
            "centroid_cache": None,
            "model_predictions": None,
        }

    def _read_segmentation(self, cell_path, mask_path, image=None):
        cell_path = str(Path(cell_path).expanduser().resolve())
        mask_path = str(Path(mask_path).expanduser().resolve())
        if not Path(cell_path).is_file():
            raise FileNotFoundError(f"Cell-data file not found: {cell_path}")
        if not Path(mask_path).is_file():
            raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")

        separator = "," if cell_path.lower().endswith(".csv") else "\t"
        cell_data = pd.read_csv(cell_path, sep=separator)
        if cell_data.empty:
            raise ValueError("The selected cell-data file contains no rows.")

        try:
            # Keep large, uncompressed masks on disk when possible so adding
            # several images does not require loading every mask into RAM.
            masks = np.squeeze(tifffile.memmap(mask_path))
        except (ValueError, TypeError):
            masks = np.squeeze(tifffile.imread(mask_path))
        if masks.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional annotation mask; got {masks.shape}."
            )
        if image is not None:
            _, image_height, image_width = image.get_shape()
            if masks.shape != (image_height, image_width):
                raise ValueError(
                    "The annotation mask dimensions do not match the image "
                    f"({masks.shape[1]} x {masks.shape[0]} mask; "
                    f"{image_width} x {image_height} image)."
                )
        return cell_data, masks, cell_path, mask_path

    def _capture_current_image_state(self):
        if not (0 <= self.current_image_index < len(self.loaded_images)):
            return
        state = self.loaded_images[self.current_image_index]
        state.update({
            "image_path": self.image_path,
            "cell_data_path": self.cell_data_path,
            "segmentation_mask_path": self.segmentation_mask_path,
            "img": self.img,
            "fov_generator": self.fov_generator,
            "cell_data": self.cell_data,
            "segmentation_masks": self.segmentation_masks,
            "annotations": self.annotations,
            "current_y0": self.current_y0,
            "current_x0": self.current_x0,
            "current_fov": self.current_fov,
            "current_dapi_fov": self.current_dapi_fov,
            "channel_index": self.channel_dropdown.currentIndex(),
        })

    def _activate_image(self, index):
        if not (0 <= index < len(self.loaded_images)):
            return
        state = self.loaded_images[index]
        self.current_image_index = index
        self.image_path = state["image_path"]
        self.cell_data_path = state["cell_data_path"]
        self.segmentation_mask_path = state["segmentation_mask_path"]
        self.img = state["img"]
        self.fov_generator = state["fov_generator"]
        self.cell_data = state["cell_data"]
        self.segmentation_masks = state["segmentation_masks"]
        self.annotations = state["annotations"]
        self.current_y0 = state["current_y0"]
        self.current_x0 = state["current_x0"]
        self.current_fov = state["current_fov"]
        self.current_dapi_fov = state["current_dapi_fov"]
        self.current_pixmap = None
        self.channel_dropdown.blockSignals(True)
        self.channel_dropdown.clear()
        self.channel_dropdown.addItems(self.img.get_channel_names())
        channel_index = min(
            max(int(state.get("channel_index", 0)), 0),
            max(self.channel_dropdown.count() - 1, 0),
        )
        self.channel_dropdown.setCurrentIndex(channel_index)
        self.channel_dropdown.blockSignals(False)
        self.image_carousel.blockSignals(True)
        self.image_carousel.setCurrentIndex(index)
        self.image_carousel.blockSignals(False)
        self.update_annotation_counts()
        self.update_model_prediction_counts()

        if self.current_fov is None:
            self.image_label.clear()
            self.image_label.setText(
                f"{Path(self.image_path).name} loaded.\n\nClick 'Generate FOV'."
            )
        else:
            self.update_display()
        self.update_image_carousel_controls()
        self.update_model_controls()

    def switch_image(self, index):
        if self.is_loading or index == self.current_image_index:
            return
        self._capture_current_image_state()
        self._activate_image(index)
        self.set_loading(False, f"Showing {Path(self.image_path).name}")

    def cycle_image(self, step):
        if self.is_loading or not self.loaded_images:
            return
        index = (self.current_image_index + step) % len(self.loaded_images)
        self.switch_image(index)

    def _refresh_image_carousel(self):
        self.image_carousel.blockSignals(True)
        self.image_carousel.clear()
        for index, state in enumerate(self.loaded_images, start=1):
            self.image_carousel.addItem(
                f"{index} / {len(self.loaded_images)} — "
                f"{Path(state['image_path']).name}"
            )
        if self.current_image_index >= 0:
            self.image_carousel.setCurrentIndex(self.current_image_index)
        self.image_carousel.blockSignals(False)
        self.update_image_carousel_controls()

    def update_image_carousel_controls(self):
        if not hasattr(self, "previous_image_button"):
            return
        has_images = bool(self.loaded_images)
        can_cycle = len(self.loaded_images) > 1 and not self.is_loading
        self.previous_image_button.setEnabled(can_cycle)
        self.next_image_button.setEnabled(can_cycle)
        self.image_carousel.setEnabled(has_images and not self.is_loading)

    def label_clicked_cell(self, x_fraction, y_fraction):
        if self.current_fov is None or self.segmentation_masks is None:
            return

        height, width = self.current_fov.shape[:2]
        local_x = min(int(x_fraction * width), width - 1)
        local_y = min(int(y_fraction * height), height - 1)
        global_x = int(self.current_x0) + local_x
        global_y = int(self.current_y0) + local_y

        if not (0 <= global_y < self.segmentation_masks.shape[0]
                and 0 <= global_x < self.segmentation_masks.shape[1]):
            return

        raw_cell_id = self.segmentation_masks[global_y, global_x]
        if raw_cell_id == 0:
            self.status_label.setText("No segmented cell at the selected location.")
            return

        cell_id = str(raw_cell_id.item() if hasattr(raw_cell_id, "item") else raw_cell_id)
        centroid_x, centroid_y = self._cell_centroid_near_click(
            raw_cell_id, global_x, global_y
        )
        state = self.loaded_images[self.current_image_index]
        try:
            row_index = self._find_cell_row_index(
                state,
                {
                    "cell_id": cell_id,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                },
            )
        except ValueError:
            row_index = None
        phenotype = self.phenotype_name.text().strip() or "unnamed phenotype"

        message = QMessageBox(self)
        message.setWindowTitle("Label cell")
        message.setText(f"Label cell {cell_id} for {phenotype}:")
        positive_button = message.addButton("Positive", QMessageBox.AcceptRole)
        negative_button = message.addButton("Negative", QMessageBox.RejectRole)
        exclude_button = message.addButton("Do not train", QMessageBox.DestructiveRole)
        message.addButton(QMessageBox.Cancel)
        message.exec()

        selected = message.clickedButton()
        if selected is positive_button:
            label = "positive"
        elif selected is negative_button:
            label = "negative"
        elif selected is exclude_button:
            self.annotations.pop(cell_id, None)
            self.update_annotation_counts()
            self.update_display()
            return
        else:
            return

        self.annotations[cell_id] = {
            "cell_id": cell_id,
            "label": label,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "row_index": row_index,
        }
        self.update_annotation_counts()
        self.update_display()

    def _cell_centroid_near_click(self, cell_id, global_x, global_y):
        # Cells are compact relative to a 129 x 129 pixel search area. Keeping
        # this local avoids scanning a whole-slide mask after every click.
        radius = 64
        y0 = max(global_y - radius, 0)
        y1 = min(global_y + radius + 1, self.segmentation_masks.shape[0])
        x0 = max(global_x - radius, 0)
        x1 = min(global_x + radius + 1, self.segmentation_masks.shape[1])
        window = self.segmentation_masks[y0:y1, x0:x1]
        rows, columns = np.nonzero(window == cell_id)
        if not len(rows):
            return float(global_x), float(global_y)
        return float(x0 + columns.mean()), float(y0 + rows.mean())

    def update_annotation_counts(self):
        positive = len(self.training_annotations("positive"))
        negative = len(self.training_annotations("negative"))
        self.positive_count_label.setText(f"Positive: {positive}")
        self.negative_count_label.setText(f"Negative: {negative}")
        for label, count in (("positive", positive), ("negative", negative)):
            if self.training_navigation_indices[label] >= count:
                self.training_navigation_indices[label] = -1
        self.update_training_navigation_controls()
        self.update_model_controls()

    def training_annotations(self, label):
        return [
            (image_index, annotation)
            for image_index, state in enumerate(self.loaded_images)
            for annotation in state["annotations"].values()
            if annotation["label"] == label
        ]

    def update_training_navigation_controls(self):
        if not hasattr(self, "previous_positive_button"):
            return
        for label, previous_button, next_button, position_label in (
            (
                "positive", self.previous_positive_button,
                self.next_positive_button, self.positive_position_label,
            ),
            (
                "negative", self.previous_negative_button,
                self.next_negative_button, self.negative_position_label,
            ),
        ):
            count = len(self.training_annotations(label))
            index = self.training_navigation_indices[label]
            enabled = count > 0 and self.img is not None and not self.is_loading
            previous_button.setEnabled(enabled)
            next_button.setEnabled(enabled)
            position_label.setText(
                f"{index + 1} / {count}" if index >= 0 else f"— / {count}"
            )

    def navigate_training(self, label, step):
        annotations = self.training_annotations(label)
        if not annotations or self.img is None or self.is_loading:
            return

        index = self.training_navigation_indices[label]
        if index < 0:
            index = 0 if step > 0 else len(annotations) - 1
        else:
            index = (index + step) % len(annotations)
        self.training_navigation_indices[label] = index

        image_index, annotation = annotations[index]
        if image_index != self.current_image_index:
            self.switch_image(image_index)
        centroid_x = float(annotation["centroid_x"])
        centroid_y = float(annotation["centroid_y"])
        _, image_height, image_width = self.img.get_shape()
        maximum_x0 = max(int(image_width) - self.fov_size, 0)
        maximum_y0 = max(int(image_height) - self.fov_size, 0)
        self.current_x0 = min(
            max(round(centroid_x - self.fov_size / 2), 0), maximum_x0
        )
        self.current_y0 = min(
            max(round(centroid_y - self.fov_size / 2), 0), maximum_y0
        )

        if label == "positive":
            self.positive_annotations_checkbox.setChecked(True)
        else:
            self.negative_annotations_checkbox.setChecked(True)
        self.update_training_navigation_controls()
        self.reload_current_fov()

    def current_annotation_markers(self):
        if self.current_fov is None:
            return []
        x0, y0 = float(self.current_x0), float(self.current_y0)
        height, width = self.current_fov.shape[:2]
        markers = []
        for annotation in self.annotations.values():
            label = annotation["label"]
            if label == "positive" and not self.positive_annotations_checkbox.isChecked():
                continue
            if label == "negative" and not self.negative_annotations_checkbox.isChecked():
                continue
            x = float(annotation["centroid_x"]) - x0
            y = float(annotation["centroid_y"]) - y0
            if 0 <= x < width and 0 <= y < height:
                markers.append({
                    "x": x, "y": y, "label": label, "source": "manual"
                })
        return markers

    @staticmethod
    def _centroid_columns(cell_data):
        def find_axis(axis):
            exact = [
                f"Centroid {axis.upper()} µm",
                f"Centroid {axis.upper()} μm",
                f"Centroid {axis.upper()} um",
                f"Centroid {axis.upper()} px",
                f"Centroid {axis.upper()}",
            ]
            for candidate in exact:
                if candidate in cell_data.columns:
                    return candidate
            for column in cell_data.columns:
                normalized = str(column).lower().replace("_", " ").replace("-", " ")
                has_centroid = "centroid" in normalized or "center" in normalized
                has_axis = f" {axis} " in f" {normalized} " or normalized.endswith(axis)
                if has_centroid and has_axis:
                    return column
            return None

        x_column, y_column = find_axis("x"), find_axis("y")
        if x_column is None or y_column is None:
            raise ValueError(
                "Cell data must contain X and Y centroid columns to display "
                "model predictions."
            )
        return x_column, y_column

    @staticmethod
    def _coordinates_are_microns(column):
        name = str(column).lower()
        return any(unit in name for unit in ("µm", "μm", " um", "micron"))

    def _cell_centroid_cache(self, state):
        if state.get("centroid_cache") is not None:
            return state["centroid_cache"]
        x_column, y_column = self._centroid_columns(state["cell_data"])
        x = pd.to_numeric(state["cell_data"][x_column], errors="coerce").to_numpy(
            dtype=float
        )
        y = pd.to_numeric(state["cell_data"][y_column], errors="coerce").to_numpy(
            dtype=float
        )
        if self._coordinates_are_microns(x_column):
            x = x / DEFAULT_PIXEL_SIZE_UM
        if self._coordinates_are_microns(y_column):
            y = y / DEFAULT_PIXEL_SIZE_UM
        state["centroid_cache"] = {
            "x": x,
            "y": y,
            "x_column": x_column,
            "y_column": y_column,
        }
        return state["centroid_cache"]

    def _find_cell_row_index(self, state, annotation):
        row_index = annotation.get("row_index")
        if row_index is not None:
            try:
                row_index = int(row_index)
            except (TypeError, ValueError):
                row_index = None
            if row_index is not None and 0 <= row_index < len(state["cell_data"]):
                return row_index

        cache = self._cell_centroid_cache(state)
        x = cache["x"]
        y = cache["y"]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.any():
            distance_squared = (
                (x - float(annotation["centroid_x"])) ** 2
                + (y - float(annotation["centroid_y"])) ** 2
            )
            distance_squared[~valid] = np.inf
            nearest = int(np.argmin(distance_squared))
            if distance_squared[nearest] <= 75 ** 2:
                return nearest

        try:
            sequential_index = int(float(annotation["cell_id"])) - 1
        except (TypeError, ValueError):
            return None
        if 0 <= sequential_index < len(state["cell_data"]):
            return sequential_index
        return None

    @staticmethod
    def _excluded_feature(column):
        name = str(column).lower().replace("_", " ").replace("-", " ")
        excluded = (
            "centroid", "object id", "cell id", "label", "classification",
            "geometry", "polygon", "bounding", "bbox", "roi", "x min",
            "x max", "y min", "y max",
        )
        return any(fragment in name for fragment in excluded)

    def _shared_numeric_features(self):
        if not self.loaded_images:
            return []
        per_image = []
        for state in self.loaded_images:
            numeric = set()
            for column in state["cell_data"].columns:
                if self._excluded_feature(column):
                    continue
                values = pd.to_numeric(state["cell_data"][column], errors="coerce")
                if values.notna().any():
                    numeric.add(column)
            per_image.append(numeric)
        shared = set.intersection(*per_image) if per_image else set()
        return [
            column for column in self.loaded_images[0]["cell_data"].columns
            if column in shared
        ]

    def train_model(self):
        try:
            features = self._shared_numeric_features()
            if not features:
                raise ValueError(
                    "No shared numeric measurement columns were found across "
                    "the loaded cell-data files."
                )

            rows, targets = [], []
            skipped = 0
            for state in self.loaded_images:
                for annotation in state["annotations"].values():
                    row_index = self._find_cell_row_index(state, annotation)
                    if row_index is None:
                        skipped += 1
                        continue
                    annotation["row_index"] = row_index
                    rows.append(state["cell_data"].iloc[row_index][features])
                    targets.append(1 if annotation["label"] == "positive" else 0)

            if set(targets) != {0, 1}:
                raise ValueError(
                    "Label at least one Positive and one Negative cell before training."
                )
            training_data = pd.DataFrame(rows, columns=features).apply(
                pd.to_numeric, errors="coerce"
            )
            features = [
                column for column in features
                if training_data[column].notna().any()
                and training_data[column].nunique(dropna=True) > 1
            ]
            if not features:
                raise ValueError(
                    "The labelled cells do not vary in any shared measurement column."
                )

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
            pipeline.fit(training_data[features], np.asarray(targets, dtype=np.uint8))
            self.model_bundle = {
                "format": MODEL_FORMAT,
                "version": MODEL_VERSION,
                "phenotype_name": self.phenotype_name.text().strip(),
                "feature_columns": features,
                "algorithm": "RandomForestClassifier",
                "training_samples": len(targets),
                "pipeline": pipeline,
            }
            for state in self.loaded_images:
                state["model_predictions"] = None
            self.model_status_label.setText(
                f"Random forest trained on {len(targets)} cells and "
                f"{len(features)} features"
                + (f"; {skipped} labels skipped." if skipped else ".")
            )
            self.update_model_prediction_counts()
            self.update_model_controls()
            self.update_display()
        except Exception as error:
            QMessageBox.warning(self, "Could not train model", str(error))

    def apply_model(self):
        if self.model_bundle is None:
            return
        try:
            features = list(self.model_bundle["feature_columns"])
            prepared = []
            for state in self.loaded_images:
                if state["cell_data"] is None:
                    raise ValueError(
                        f"{Path(state['image_path']).name} has no cell-data file."
                    )
                missing = [
                    column for column in features
                    if column not in state["cell_data"].columns
                ]
                if missing:
                    raise ValueError(
                        f"{Path(state['image_path']).name} is missing model "
                        f"features: {', '.join(missing[:8])}"
                    )
                measurements = state["cell_data"][features].apply(
                    pd.to_numeric, errors="coerce"
                )
                centroids = self._cell_centroid_cache(state)
                prepared.append((state, measurements, centroids))

            for state, measurements, centroids in prepared:
                prediction = np.asarray(
                    self.model_bundle["pipeline"].predict(measurements),
                    dtype=np.uint8,
                )
                state["model_predictions"] = {
                    "x": centroids["x"],
                    "y": centroids["y"],
                    "positive": prediction.astype(bool),
                }
            self.modelled_phenotypes_checkbox.setChecked(True)
            self.update_model_prediction_counts()
            self.update_model_controls()
            self.update_display()
            self.status_label.setText(
                f"Applied model to {len(self.loaded_images)} loaded image(s)."
            )
        except Exception as error:
            QMessageBox.warning(self, "Could not apply model", str(error))

    def current_model_markers(self):
        if (
            not self.modelled_phenotypes_checkbox.isChecked()
            or not (0 <= self.current_image_index < len(self.loaded_images))
            or self.current_fov is None
        ):
            return []
        predictions = self.loaded_images[self.current_image_index].get(
            "model_predictions"
        )
        if predictions is None:
            return []
        x0, y0 = float(self.current_x0), float(self.current_y0)
        height, width = self.current_fov.shape[:2]
        x, y = predictions["x"], predictions["y"]
        visible = (
            np.isfinite(x) & np.isfinite(y)
            & (x >= x0) & (x < x0 + width)
            & (y >= y0) & (y < y0 + height)
        )
        indices = np.flatnonzero(visible)
        return [
            {
                "x": float(x[index] - x0),
                "y": float(y[index] - y0),
                "label": (
                    "positive" if predictions["positive"][index] else "negative"
                ),
                "source": "model",
            }
            for index in indices
        ]

    def update_model_prediction_counts(self):
        positive = negative = 0
        for state in self.loaded_images:
            predictions = state.get("model_predictions")
            if predictions is None:
                continue
            positive += int(np.count_nonzero(predictions["positive"]))
            negative += int(len(predictions["positive"]) - np.count_nonzero(
                predictions["positive"]
            ))
        self.model_positive_count_label.setText(f"Model positive: {positive:,}")
        self.model_negative_count_label.setText(f"Model negative: {negative:,}")

    def update_model_controls(self):
        if not hasattr(self, "train_model_button"):
            return
        has_both_labels = bool(
            self.training_annotations("positive")
            and self.training_annotations("negative")
        )
        all_have_cell_data = bool(self.loaded_images) and all(
            state["cell_data"] is not None for state in self.loaded_images
        )
        self.train_model_button.setEnabled(
            has_both_labels and all_have_cell_data and not self.is_loading
        )
        self.apply_model_button.setEnabled(
            self.model_bundle is not None
            and all_have_cell_data
            and not self.is_loading
        )
        has_predictions = any(
            state.get("model_predictions") is not None
            for state in self.loaded_images
        )
        all_have_predictions = bool(self.loaded_images) and all(
            state.get("model_predictions") is not None
            for state in self.loaded_images
        )
        self.modelled_phenotypes_checkbox.setEnabled(has_predictions)
        self.export_cell_phenotypes_button.setEnabled(
            self.model_bundle is not None
            and all_have_predictions
            and not self.is_loading
        )
        self.export_model_action.setEnabled(self.model_bundle is not None)
        self.import_model_action.setEnabled(not self.is_loading)

    @staticmethod
    def _unique_export_column(preferred_name, existing_columns):
        if preferred_name not in existing_columns:
            return preferred_name
        orbit_name = f"ORBIT {preferred_name}"
        if orbit_name not in existing_columns:
            return orbit_name
        suffix = 2
        while f"{orbit_name} {suffix}" in existing_columns:
            suffix += 1
        return f"{orbit_name} {suffix}"

    def export_cell_phenotypes(self):
        if self.model_bundle is None:
            QMessageBox.warning(
                self,
                "Export cell phenotypes",
                "Train or import and apply a model before exporting.",
            )
            return
        if not self.loaded_images or any(
            state.get("model_predictions") is None
            for state in self.loaded_images
        ):
            QMessageBox.warning(
                self,
                "Export cell phenotypes",
                "Apply the model to all loaded images before exporting.",
            )
            return

        phenotype_name = (
            self.phenotype_name.text().strip()
            or self.model_bundle.get("phenotype_name", "").strip()
            or "Phenotype"
        )
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in phenotype_name
        ).strip("_") or "phenotype"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export cell phenotypes",
            f"{safe_name}_cell_phenotypes.tsv",
            "Tab-separated values (*.tsv);;Comma-separated values (*.csv)",
        )
        if not path:
            return

        export_csv = (
            path.lower().endswith(".csv")
            or selected_filter.startswith("Comma-separated")
        )
        if not Path(path).suffix:
            path += ".csv" if export_csv else ".tsv"
        separator = "," if export_csv else "\t"
        features = list(self.model_bundle["feature_columns"])
        original_columns = []
        seen_columns = set()
        for state in self.loaded_images:
            for column in state["cell_data"].columns:
                if column not in seen_columns:
                    original_columns.append(column)
                    seen_columns.add(column)

        image_column = self._unique_export_column(
            "Image Name", seen_columns
        )
        seen_columns.add(image_column)
        row_column = self._unique_export_column("Cell Row", seen_columns)
        seen_columns.add(row_column)
        label_column = self._unique_export_column(
            f"{phenotype_name} Label", seen_columns
        )
        seen_columns.add(label_column)
        source_column = self._unique_export_column(
            "Label Source", seen_columns
        )
        temporary_path = Path(f"{path}.tmp")
        exported_rows = 0

        try:
            for image_index, state in enumerate(self.loaded_images):
                predictions = state["model_predictions"]
                cell_data = state["cell_data"]
                if len(predictions["positive"]) != len(cell_data):
                    raise ValueError(
                        f"Prediction count does not match the cell data for "
                        f"{Path(state['image_path']).name}. Reapply the model."
                    )
                missing = [column for column in features if column not in cell_data]
                if missing:
                    raise ValueError(
                        f"{Path(state['image_path']).name} is missing exported "
                        f"features: {', '.join(missing[:8])}"
                    )

                # Reindex to the union of all source schemas so every original
                # Cellpose column is retained and image blocks append cleanly.
                export_data = cell_data.reindex(columns=original_columns).copy()
                export_data.insert(
                    0, row_column, np.arange(1, len(cell_data) + 1)
                )
                export_data.insert(
                    0, image_column, Path(state["image_path"]).name
                )

                labels = np.where(
                    predictions["positive"], "Positive", "Negative"
                ).astype(object)
                label_sources = np.full(len(cell_data), "Model", dtype=object)
                for annotation in state["annotations"].values():
                    row_index = self._find_cell_row_index(state, annotation)
                    if row_index is None:
                        continue
                    labels[row_index] = (
                        "Positive"
                        if annotation["label"] == "positive"
                        else "Negative"
                    )
                    label_sources[row_index] = "Manual Training"
                export_data[label_column] = labels
                export_data[source_column] = label_sources

                export_data.to_csv(
                    temporary_path,
                    sep=separator,
                    index=False,
                    mode="w" if image_index == 0 else "a",
                    header=image_index == 0,
                )
                exported_rows += len(export_data)

            temporary_path.replace(Path(path))
            self.status_label.setText(
                f"Exported {exported_rows:,} cells to {Path(path).resolve()}"
            )
        except Exception as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
            QMessageBox.warning(
                self, "Could not export cell phenotypes", str(error)
            )

    def export_model(self):
        if self.model_bundle is None:
            QMessageBox.warning(self, "Export model", "Train or import a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ORBIT model", "", "ORBIT model (*.orbitmodel)"
        )
        if not path:
            return
        if not path.lower().endswith(".orbitmodel"):
            path += ".orbitmodel"
        try:
            joblib.dump(self.model_bundle, path)
            self.status_label.setText(f"Exported model: {Path(path).resolve()}")
        except Exception:
            QMessageBox.critical(self, "Could not export model", traceback.format_exc())

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import ORBIT model", "", "ORBIT model (*.orbitmodel);;All files (*)"
        )
        if not path:
            return
        try:
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
                raise ValueError("The selected file is not an ORBIT phenotype model.")
            if bundle.get("version") != MODEL_VERSION:
                raise ValueError(
                    f"Unsupported ORBIT model version: {bundle.get('version')}"
                )
            if not bundle.get("feature_columns") or not hasattr(
                bundle.get("pipeline"), "predict"
            ):
                raise ValueError("The ORBIT model file is incomplete.")
            self.model_bundle = bundle
            for state in self.loaded_images:
                state["model_predictions"] = None
            if not self.phenotype_name.text().strip():
                self.phenotype_name.setText(bundle.get("phenotype_name", ""))
            self.model_status_label.setText(
                f"Loaded {bundle.get('algorithm', 'model')} with "
                f"{len(bundle['feature_columns'])} features."
            )
            self.update_model_prediction_counts()
            self.update_model_controls()
            self.update_display()
        except Exception as error:
            QMessageBox.warning(self, "Could not import model", str(error))

    def new_project(self):
        if self.loaded_images:
            choice = QMessageBox.question(
                self,
                "New project",
                "Close the current project? Save it first if you want to keep "
                "its training labels.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
        for state in self.loaded_images:
            try:
                state["img"].tif.close()
            except Exception:
                pass
        self.loaded_images = []
        self.current_image_index = -1
        self.img = self.fov_generator = None
        self.image_path = self.cell_data_path = None
        self.segmentation_mask_path = None
        self.cell_data = self.segmentation_masks = None
        self.current_y0 = self.current_x0 = None
        self.current_fov = self.current_dapi_fov = None
        self.current_pixmap = None
        self.annotations = {}
        self.model_bundle = None
        self.project_path = None
        self.training_navigation_indices = {"positive": -1, "negative": -1}
        self.phenotype_name.clear()
        self.channel_dropdown.clear()
        self.image_label.clear()
        self.image_label.setText("Select a QPTIFF image")
        self.model_status_label.setText("No model trained or loaded.")
        self._refresh_image_carousel()
        self.update_annotation_counts()
        self.update_model_prediction_counts()
        self.update_model_controls()
        self.set_loading(False, "New project")

    def project_data(self):
        if not self.loaded_images:
            raise ValueError("Load an image before saving a project.")
        self._capture_current_image_state()
        images = []
        for state in self.loaded_images:
            images.append({
                "paths": {
                    "image": state["image_path"],
                    "cell_data": state["cell_data_path"],
                    "segmentation_mask": state["segmentation_mask_path"],
                },
                "annotations": list(state["annotations"].values()),
                "viewer": {
                    "current_x0": (
                        None if state["current_x0"] is None
                        else int(state["current_x0"])
                    ),
                    "current_y0": (
                        None if state["current_y0"] is None
                        else int(state["current_y0"])
                    ),
                    "channel_index": int(state.get("channel_index", 0)),
                },
            })
        return {
            "format": "ORBIT phenotype training session",
            "version": 2,
            "images": images,
            "current_image_index": self.current_image_index,
            "phenotype": {
                "name": self.phenotype_name.text(),
                "show_positive": self.positive_annotations_checkbox.isChecked(),
                "show_negative": self.negative_annotations_checkbox.isChecked(),
            },
            "viewer": {
                "fov_size": self.fov_size,
                "color": self.color_dropdown.currentText(),
                "show_dapi": self.dapi_checkbox.isChecked(),
                "show_segmentation": self.segmentation_checkbox.isChecked(),
            },
        }

    def save_project(self):
        if not self.project_path:
            self.save_project_as()
            return
        self._write_project(self.project_path)

    def save_project_as(self):
        if not self.image_path:
            QMessageBox.warning(self, "Save project", "Load an image before saving.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ORBIT project", "", "ORBIT project (*.orbit.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".orbit.json"):
            path += ".orbit.json"
        self._write_project(path)

    def _write_project(self, path):
        try:
            with open(path, "w", encoding="utf-8") as project_file:
                json.dump(self.project_data(), project_file, indent=2)
            self.project_path = str(Path(path).resolve())
            self.status_label.setText(f"Saved project: {self.project_path}")
        except Exception:
            QMessageBox.critical(self, "Could not save project", traceback.format_exc())

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open ORBIT project", "", "ORBIT project (*.orbit.json *.json)"
        )
        if not path:
            return

        self.set_loading(True, "Opening project...")
        try:
            with open(path, "r", encoding="utf-8") as project_file:
                data = json.load(project_file)
            if data.get("format") != "ORBIT phenotype training session":
                raise ValueError("The selected file is not an ORBIT training session.")
            version = data.get("version")
            if version not in {1, 2}:
                raise ValueError(f"Unsupported ORBIT project version: {data.get('version')}")
            phenotype = data.get("phenotype", {})
            viewer = data.get("viewer", {})
            if version == 1:
                image_entries = [{
                    "paths": data["paths"],
                    "annotations": phenotype.get("annotations", []),
                    "viewer": {
                        "current_x0": viewer.get("current_x0"),
                        "current_y0": viewer.get("current_y0"),
                        "channel_index": viewer.get("channel_index", 0),
                    },
                }]
                target_index = 0
            else:
                image_entries = data.get("images", [])
                target_index = int(data.get("current_image_index", 0))
            if not image_entries:
                raise ValueError("The project does not contain any images.")

            loaded_states = []
            for entry in image_entries:
                paths = entry.get("paths", {})
                if not paths.get("image"):
                    raise ValueError("A project image is missing its image path.")
                state = self._create_image_state(
                    paths["image"],
                    paths.get("cell_data"),
                    paths.get("segmentation_mask"),
                )
                state["annotations"] = {
                    str(annotation["cell_id"]): annotation
                    for annotation in entry.get("annotations", [])
                    if annotation.get("label") in {"positive", "negative"}
                }
                image_viewer = entry.get("viewer", {})
                state["current_x0"] = image_viewer.get("current_x0")
                state["current_y0"] = image_viewer.get("current_y0")
                state["channel_index"] = int(
                    image_viewer.get("channel_index", 0)
                )
                loaded_states.append(state)

            for old_state in self.loaded_images:
                try:
                    old_state["img"].tif.close()
                except Exception:
                    pass
            self.loaded_images = loaded_states
            self.current_image_index = -1
            self.model_bundle = None
            self.model_status_label.setText("No model trained or loaded.")
            self.fov_size = int(viewer.get("fov_size", 512))
            self.phenotype_name.setText(phenotype.get("name", ""))
            self.positive_annotations_checkbox.setChecked(
                phenotype.get("show_positive", True)
            )
            self.negative_annotations_checkbox.setChecked(
                phenotype.get("show_negative", True)
            )
            color = viewer.get("color", "Green")
            if color in COLOR_MAPS:
                self.color_dropdown.setCurrentText(color)
            self.dapi_checkbox.setChecked(viewer.get("show_dapi", True))
            self.segmentation_checkbox.setChecked(
                viewer.get("show_segmentation", True)
            )
            target_index = min(max(target_index, 0), len(self.loaded_images) - 1)
            self._refresh_image_carousel()
            self._activate_image(target_index)
            self.project_path = str(Path(path).resolve())
            has_fov = self.current_x0 is not None and self.current_y0 is not None
            message = f"Opened project: {self.project_path}"
        except Exception:
            self.set_loading(False)
            QMessageBox.critical(self, "Could not open project", traceback.format_exc())
            return

        self.set_loading(False, message)
        if has_fov:
            self.reload_current_fov()

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
                annotation_markers=(
                    self.current_model_markers()
                    + self.current_annotation_markers()
                ),
            )
            self.display_pixmap()
        except Exception:
            self.status_label.setText(traceback.format_exc())
