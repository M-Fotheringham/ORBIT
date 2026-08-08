"""Embedded Napari canvas for ORBIT's existing Qt interface."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import napari
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget


NAPARI_COLORMAPS = {
    "Gray": "gray",
    "Red": "red",
    "Green": "green",
    "Blue": "blue",
    "Cyan": "cyan",
    "Magenta": "magenta",
    "Yellow": "yellow",
}

OVERVIEW_COLOR_SCALES = {
    "Gray": (1.0, 1.0, 1.0),
    "Red": (1.0, 0.0, 0.0),
    "Green": (0.0, 1.0, 0.0),
    "Blue": (0.0, 0.0, 1.0),
    "Cyan": (0.0, 1.0, 1.0),
    "Magenta": (1.0, 0.0, 1.0),
    "Yellow": (1.0, 1.0, 0.0),
}

# The original QLabel canvas uses Qt.SmoothTransformation when enlarging a
# 512-pixel FOV. Lanczos is Napari/Vispy's highest-quality photographic-image
# resampler and preserves substantially more apparent detail than nearest
# neighbour at ORBIT's normal 700+ pixel canvas size. Binary overlays must stay
# nearest-neighbour so masks and one-pixel boundaries are not blurred.
IMAGE_INTERPOLATION = "lanczos"
MASK_INTERPOLATION = "nearest"


def _contrast_limits(data: np.ndarray) -> tuple[float, float]:
    values = np.asarray(data)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    low, high = _contrast_limits(values)
    normalized = np.asarray(values, dtype=np.float32)
    normalized = np.nan_to_num(
        normalized,
        nan=low,
        posinf=high,
        neginf=low,
    )
    normalized = np.clip((normalized - low) / (high - low), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


class OverviewNavigator(QWidget):
    """Top-left whole-image navigator with a full-resolution FOV outline."""

    clicked = Signal(float, float)
    BORDER = 3
    MAXIMUM_IMAGE_SIZE = 250

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._full_shape = None
        self._fov_rect = None
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to center the field of view at this location")
        self.hide()

    @property
    def has_image(self):
        return self._pixmap is not None

    def set_image(self, rgb, full_shape, fov_rect=None):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"The overview navigator requires RGB data; got {rgb.shape}."
            )
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data,
            width,
            height,
            3 * width,
            QImage.Format_RGB888,
        ).copy()
        source = QPixmap.fromImage(qimage)
        self._pixmap = source.scaled(
            self.MAXIMUM_IMAGE_SIZE,
            self.MAXIMUM_IMAGE_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._full_shape = tuple(int(value) for value in full_shape)
        self._fov_rect = fov_rect
        self.setFixedSize(
            self._pixmap.width() + 2 * self.BORDER,
            self._pixmap.height() + 2 * self.BORDER,
        )
        self.update()

    def set_fov_rect(self, fov_rect):
        self._fov_rect = fov_rect
        self.update()

    def clear(self):
        self._pixmap = None
        self._full_shape = None
        self._fov_rect = None
        self.hide()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101010"))
        painter.drawPixmap(self.BORDER, self.BORDER, self._pixmap)

        if self._fov_rect is not None and self._full_shape is not None:
            full_height, full_width = self._full_shape
            x0, y0, fov_width, fov_height = self._fov_rect
            left = self.BORDER + x0 / full_width * self._pixmap.width()
            top = self.BORDER + y0 / full_height * self._pixmap.height()
            width = max(fov_width / full_width * self._pixmap.width(), 2.0)
            height = max(fov_height / full_height * self._pixmap.height(), 2.0)
            painter.setPen(QPen(QColor("#ff2020"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                round(left),
                round(top),
                max(round(width), 2),
                max(round(height), 2),
            )
        painter.end()

    def mousePressEvent(self, event):
        if (
            self._pixmap is not None
            and event.button() == Qt.LeftButton
        ):
            x = event.position().x() - self.BORDER
            y = event.position().y() - self.BORDER
            if 0 <= x < self._pixmap.width() and 0 <= y < self._pixmap.height():
                self.clicked.emit(
                    float(x / self._pixmap.width()),
                    float(y / self._pixmap.height()),
                )
                event.accept()
                return
        super().mousePressEvent(event)


class NapariImageCanvas(QWidget):
    """Napari-powered drop-in replacement for ORBIT's clickable image label.

    Only the canvas is embedded. ORBIT keeps its menus, panels, carousel,
    project model and phenotyping controls unchanged.
    """

    image_clicked = Signal(float, float)
    image_hovered = Signal(float, float, object)
    image_hover_left = Signal()
    overview_clicked = Signal(float, float)

    def __init__(self, message: str = "", parent=None):
        super().__init__(parent)
        self._image_shape: tuple[int, int] | None = None
        self._marker_identity = None

        self.viewer = napari.Viewer(show=False, title="ORBIT image canvas")
        self.viewer.axes.visible = False
        self.viewer.scale_bar.visible = False
        self.viewer.mouse_drag_callbacks.append(self._mouse_drag_callback)
        self.viewer.mouse_move_callbacks.append(self._mouse_move_callback)

        qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        canvas = getattr(qt_viewer, "canvas", None)
        self._native_canvas = getattr(canvas, "native", None)
        if self._native_canvas is None:
            raise RuntimeError(
                "This Napari version does not expose an embeddable Qt canvas."
            )
        self._native_canvas.setParent(self)
        self._native_canvas.installEventFilter(self)

        self._message_label = QLabel(message, self)
        self._message_label.setAlignment(Qt.AlignCenter)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet(
            "QLabel { background: black; color: white; font-size: 16px; }"
        )

        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStackingMode(QStackedLayout.StackAll)
        layout.addWidget(self._native_canvas)
        layout.addWidget(self._message_label)
        self.setLayout(layout)

        self._overview_visible = True
        self._overview = OverviewNavigator(self)
        self._overview.clicked.connect(self.overview_clicked.emit)
        self._overview.move(12, 12)

    def set_scene(
        self,
        marker_arr: np.ndarray,
        marker_color: str = "Green",
        dapi_arr: np.ndarray | None = None,
        show_dapi: bool = True,
        segmentation_boundary: np.ndarray | None = None,
        threshold_highlight: np.ndarray | None = None,
        annotation_markers: list[dict] | None = None,
    ):
        marker = np.asarray(marker_arr)
        if marker.ndim != 2:
            raise ValueError(f"Napari canvas requires a 2D FOV; got {marker.shape}.")
        self._image_shape = tuple(int(value) for value in marker.shape)
        is_new_fov = id(marker_arr) != self._marker_identity
        self._marker_identity = id(marker_arr)
        self._message_label.hide()

        self._set_image_layer(
            "Marker",
            marker,
            colormap=NAPARI_COLORMAPS.get(marker_color, "green"),
            visible=True,
            blending="additive",
            contrast_limits=_contrast_limits(marker),
            interpolation=IMAGE_INTERPOLATION,
        )
        if dapi_arr is not None:
            dapi = np.asarray(dapi_arr)
            self._set_image_layer(
                "DAPI",
                dapi,
                colormap="blue",
                visible=show_dapi,
                blending="additive",
                contrast_limits=_contrast_limits(dapi),
                interpolation=IMAGE_INTERPOLATION,
            )
        else:
            self._remove_layer("DAPI")

        self._set_binary_layer(
            "Threshold highlight",
            threshold_highlight,
            colormap="yellow",
            opacity=0.75,
        )
        self._set_binary_layer(
            "Segmentation boundary",
            segmentation_boundary,
            colormap="red",
            opacity=1.0,
        )
        self._set_marker_layers(annotation_markers or [])

        if is_new_fov:
            self.viewer.reset_view()
        self._overview.raise_()

    def set_overview_scene(
        self,
        marker_arr: np.ndarray,
        marker_color: str,
        dapi_arr: np.ndarray | None,
        show_dapi: bool,
        full_shape: tuple[int, int],
        fov_rect=None,
    ):
        marker = _normalize_to_uint8(marker_arr)
        red, green, blue = OVERVIEW_COLOR_SCALES.get(
            marker_color,
            OVERVIEW_COLOR_SCALES["Green"],
        )
        rgb = np.zeros((*marker.shape, 3), dtype=np.uint8)
        rgb[..., 0] = marker * red
        rgb[..., 1] = marker * green
        rgb[..., 2] = marker * blue
        if show_dapi and dapi_arr is not None:
            dapi = _normalize_to_uint8(dapi_arr)
            if dapi.shape != marker.shape:
                raise ValueError(
                    "The marker and DAPI overview dimensions do not match "
                    f"({marker.shape} versus {dapi.shape})."
                )
            rgb[..., 2] = np.maximum(rgb[..., 2], dapi)
        self._overview.set_image(rgb, full_shape, fov_rect=fov_rect)
        self._overview.setVisible(self._overview_visible)
        self._overview.raise_()

    def update_overview_fov(self, fov_rect):
        self._overview.set_fov_rect(fov_rect)
        self._overview.raise_()

    def set_overview_visible(self, visible: bool):
        self._overview_visible = bool(visible)
        self._overview.setVisible(
            self._overview_visible and self._overview.has_image
        )
        if self._overview.isVisible():
            self._overview.raise_()

    def _set_image_layer(
        self,
        name,
        data,
        *,
        colormap,
        visible,
        blending,
        contrast_limits,
        interpolation,
    ):
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            layer.data = data
            layer.colormap = colormap
            layer.visible = visible
            layer.blending = blending
            layer.contrast_limits = contrast_limits
        else:
            layer = self.viewer.add_image(
                data,
                name=name,
                colormap=colormap,
                visible=visible,
                blending=blending,
                contrast_limits=contrast_limits,
            )
        self._set_interpolation(layer, interpolation)

    @staticmethod
    def _set_interpolation(layer, interpolation):
        """Use high-quality image scaling without softening binary masks."""
        if hasattr(layer, "interpolation2d"):
            layer.interpolation2d = interpolation
        elif hasattr(layer, "interpolation"):
            # Compatibility with older Napari releases.
            layer.interpolation = interpolation

    def _set_binary_layer(self, name, data, *, colormap, opacity):
        if data is None:
            self._remove_layer(name)
            return
        self._set_image_layer(
            name,
            np.asarray(data, dtype=np.uint8),
            colormap=colormap,
            visible=True,
            blending="additive",
            contrast_limits=(0.0, 1.0),
            interpolation=MASK_INTERPOLATION,
        )
        self.viewer.layers[name].opacity = float(opacity)

    def _set_marker_layers(self, markers: list[dict]):
        grouped = defaultdict(list)
        for marker in markers:
            grouped[(marker.get("source", "manual"), marker["label"])].append(
                (float(marker["y"]), float(marker["x"]))
            )

        expected_names = set()
        for (source, label), coordinates in grouped.items():
            name = f"{source.title()} {label.title()}"
            expected_names.add(name)
            data = np.asarray(coordinates, dtype=float)
            color = "#00eeff" if label == "positive" else "#ff3030"
            size = 10 if source == "manual" else 6
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                layer.data = data
                layer.size = size
                layer.face_color = color
                layer.visible = True
            else:
                self.viewer.add_points(
                    data,
                    name=name,
                    size=size,
                    face_color=color,
                    border_color="black",
                    border_width=0.15,
                    border_width_is_relative=True,
                    opacity=1.0,
                )

        for layer in list(self.viewer.layers):
            if (
                getattr(layer, "metadata", {}).get("orbit_annotation_layer")
                and layer.name not in expected_names
            ):
                self.viewer.layers.remove(layer)
        for name in expected_names:
            self.viewer.layers[name].metadata["orbit_annotation_layer"] = True

    def _remove_layer(self, name: str):
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)

    def _fraction_at_cursor(self):
        if self._image_shape is None:
            return None
        position = tuple(float(value) for value in self.viewer.cursor.position)
        if len(position) < 2:
            return None
        y, x = position[-2:]
        height, width = self._image_shape
        if not (0 <= x < width and 0 <= y < height):
            return None
        return x / width, y / height

    def _mouse_drag_callback(self, _viewer, event):
        if getattr(event, "type", None) != "mouse_press":
            return
        if getattr(event, "button", 1) not in {1, "left"}:
            return
        fraction = self._fraction_at_cursor()
        if fraction is not None:
            self.image_clicked.emit(*fraction)

    def _mouse_move_callback(self, _viewer, _event):
        fraction = self._fraction_at_cursor()
        if fraction is None:
            self.image_hover_left.emit()
            return
        self.image_hovered.emit(fraction[0], fraction[1], QCursor.pos())

    def eventFilter(self, watched, event):
        if watched is self._native_canvas and event.type() == QEvent.Leave:
            self.image_hover_left.emit()
        return super().eventFilter(watched, event)

    def clear(self):
        self.viewer.layers.clear()
        self._image_shape = None
        self._marker_identity = None
        self._overview.clear()
        self._message_label.show()

    def setText(self, text: str):
        self._message_label.setText(text)
        self._message_label.show()

    def setAlignment(self, alignment):
        self._message_label.setAlignment(alignment)

    def setPixmap(self, _pixmap: QPixmap):
        """Compatibility no-op; Napari owns rendering rather than QPixmap."""

    def closeEvent(self, event):
        try:
            self.viewer.close()
        finally:
            super().closeEvent(event)


__all__ = ["NapariImageCanvas"]
