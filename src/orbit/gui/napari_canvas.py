"""Embedded Napari canvas for ORBIT's existing Qt interface."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import napari
from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QCursor, QPixmap
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


class NapariImageCanvas(QWidget):
    """Napari-powered drop-in replacement for ORBIT's clickable image label.

    Only the canvas is embedded. ORBIT keeps its menus, panels, carousel,
    project model and phenotyping controls unchanged.
    """

    image_clicked = Signal(float, float)
    image_hovered = Signal(float, float, object)
    image_hover_left = Signal()

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
