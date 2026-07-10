import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

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
        radius = 5
        for marker in annotation_markers:
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
        training_layout.addWidget(self.negative_count_label)
        training_layout.addStretch()
        training_panel.setLayout(training_layout)

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
        self.open_project_action = QAction("&Open...", self)
        self.open_project_action.setShortcut("Ctrl+O")
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action = QAction("&Save", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.triggered.connect(self.save_project)
        self.save_project_as_action = QAction("Save &As...", self)
        self.save_project_as_action.setShortcut("Ctrl+Shift+S")
        self.save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_project_as_action)
        layout.setMenuBar(self.menu_bar)

        viewer_layout = QHBoxLayout()
        viewer_layout.addWidget(self.image_label, stretch=1)
        viewer_layout.addWidget(training_panel)
        layout.addLayout(viewer_layout, stretch=1)
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
            self._load_image_path(path)
            self.project_path = None
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
            self._load_segmentation_paths(cell_path, mask_path)
            self.annotations.clear()
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

    def _load_image_path(self, path):
        path = str(Path(path).expanduser().resolve())
        if not Path(path).is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        self.img = QPTiffImage(path)
        self.fov_generator = RandomFOVGenerator(self.img)
        self.image_path = path
        self.channel_dropdown.blockSignals(True)
        self.channel_dropdown.clear()
        self.channel_dropdown.addItems(self.img.get_channel_names())
        self.channel_dropdown.blockSignals(False)
        self.current_y0 = self.current_x0 = None
        self.current_fov = self.current_dapi_fov = None
        self.current_pixmap = None
        self.cell_data = self.segmentation_masks = None
        self.cell_data_path = self.segmentation_mask_path = None
        self.annotations.clear()
        self.update_annotation_counts()
        self.segmentation_checkbox.setEnabled(False)
        self.image_label.clear()
        self.image_label.setText("Image loaded.\n\nClick 'Generate FOV'.")

    def _load_segmentation_paths(self, cell_path, mask_path):
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

        masks = np.squeeze(tifffile.imread(mask_path))
        if masks.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional annotation mask; got {masks.shape}."
            )

        self.cell_data = cell_data
        self.segmentation_masks = masks
        self.cell_data_path = cell_path
        self.segmentation_mask_path = mask_path

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
        positive = sum(a["label"] == "positive" for a in self.annotations.values())
        negative = sum(a["label"] == "negative" for a in self.annotations.values())
        self.positive_count_label.setText(f"Positive: {positive}")
        self.negative_count_label.setText(f"Negative: {negative}")

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
                markers.append({"x": x, "y": y, "label": label})
        return markers

    def project_data(self):
        if not self.image_path:
            raise ValueError("Load an image before saving a project.")
        return {
            "format": "ORBIT phenotype training session",
            "version": 1,
            "paths": {
                "image": self.image_path,
                "cell_data": self.cell_data_path,
                "segmentation_mask": self.segmentation_mask_path,
            },
            "phenotype": {
                "name": self.phenotype_name.text(),
                "annotations": list(self.annotations.values()),
                "show_positive": self.positive_annotations_checkbox.isChecked(),
                "show_negative": self.negative_annotations_checkbox.isChecked(),
            },
            "viewer": {
                "fov_size": self.fov_size,
                "current_x0": (
                    None if self.current_x0 is None else int(self.current_x0)
                ),
                "current_y0": (
                    None if self.current_y0 is None else int(self.current_y0)
                ),
                "channel_index": self.channel_dropdown.currentIndex(),
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
            if data.get("version") != 1:
                raise ValueError(f"Unsupported ORBIT project version: {data.get('version')}")

            paths = data["paths"]
            if not paths.get("image"):
                raise ValueError("The project does not contain an image path.")
            self._load_image_path(paths["image"])
            if paths.get("cell_data") or paths.get("segmentation_mask"):
                if not paths.get("cell_data") or not paths.get("segmentation_mask"):
                    raise ValueError("The project contains incomplete segmentation paths.")
                self._load_segmentation_paths(
                    paths["cell_data"], paths["segmentation_mask"]
                )

            phenotype = data.get("phenotype", {})
            self.phenotype_name.setText(phenotype.get("name", ""))
            annotations = phenotype.get("annotations", [])
            self.annotations = {
                str(annotation["cell_id"]): annotation
                for annotation in annotations
                if annotation.get("label") in {"positive", "negative"}
            }
            self.positive_annotations_checkbox.setChecked(
                phenotype.get("show_positive", True)
            )
            self.negative_annotations_checkbox.setChecked(
                phenotype.get("show_negative", True)
            )
            self.update_annotation_counts()

            viewer = data.get("viewer", {})
            self.fov_size = int(viewer.get("fov_size", 512))
            self.current_x0 = viewer.get("current_x0")
            self.current_y0 = viewer.get("current_y0")
            channel_index = int(viewer.get("channel_index", 0))
            self.channel_dropdown.setCurrentIndex(
                min(max(channel_index, 0), self.channel_dropdown.count() - 1)
            )
            color = viewer.get("color", "Green")
            if color in COLOR_MAPS:
                self.color_dropdown.setCurrentText(color)
            self.dapi_checkbox.setChecked(viewer.get("show_dapi", True))
            self.segmentation_checkbox.setChecked(
                viewer.get("show_segmentation", True)
            )
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
                annotation_markers=self.current_annotation_markers(),
            )
            self.display_pixmap()
        except Exception:
            self.status_label.setText(traceback.format_exc())
