import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import joblib
from skimage.segmentation import find_boundaries

from PySide6.QtWidgets import (
    QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QProgressBar, QSizePolicy, QLineEdit, QGroupBox,
    QFormLayout, QMessageBox, QMenuBar, QSlider, QStackedWidget, QScrollArea,
    QToolTip,
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QAction, QActionGroup,
)
from PySide6.QtCore import (
    Qt, QObject, Signal, QRunnable, QThreadPool, QTimer, QPoint,
)

from orbit.image import QPTiffImage
from orbit.fov import RandomFOVGenerator
from orbit.gui.napari_canvas import NapariImageCanvas
from orbit.models.random_forest import (
    MODEL_FORMAT,
    MODEL_VERSION,
    RANDOM_FOREST_ALGORITHM,
    fit_random_forest,
    model_calls_and_positive_probabilities,
)
from orbit.models.automated import (
    AUTOMATED_INTENSITY_SLIDER_VALUE,
    AUTOMATED_LOW_NEGATIVE_TRAINING_COUNT,
    AUTOMATED_MID_NEGATIVE_TRAINING_COUNT,
    AUTOMATED_NEGATIVE_TRAINING_COUNT,
    AUTOMATED_POSITIVE_PIXEL_PERCENT,
    AUTOMATED_POSITIVE_TRAINING_COUNT,
    AUTOMATED_RANDOM_SEED,
    AUTOMATED_REFINEMENT_FLUORESCENCE_FRACTION,
    AUTOMATED_REFINEMENT_MAX_CALL_PROBABILITY,
    AUTOMATED_REFINEMENT_TRAINING_COUNT,
    AUTOMATED_SECOND_REFINEMENT_MAX_CALL_PROBABILITY,
    AUTOMATED_TOP_POSITIVE_FRACTION,
    random_seed_for_stage,
    select_automated_refinement_indices,
    select_automated_training_indices,
)
from orbit.models.cellpose_segmentation import (
    CELLPOSE_SAM_MODEL,
    cuda_compatible_gpu_available,
    dapi_channel_name,
    membrane_marker_names,
    output_paths_for_image,
    segment_project_images,
)
from orbit.threshold import (
    cell_statistics_by_threshold,
    intensity_threshold_from_slider,
    phenotype_cells_by_threshold,
)


COLOR_MAPS = {
    "Gray": (1, 1, 1), "Red": (1, 0, 0), "Green": (0, 1, 0),
    "Blue": (0, 0, 1), "Cyan": (0, 1, 1),
    "Magenta": (1, 0, 1), "Yellow": (1, 1, 0),
}

DEFAULT_PIXEL_SIZE_UM = 0.5064
THRESHOLD_HISTOGRAM_BINS = 30
THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM = 10
DEFAULT_INWARD_BUFFER_UM = 2.0
MAXIMUM_INWARD_BUFFER_UM = 5.0
DEFAULT_INWARD_BUFFER_SLIDER_VALUE = int(
    DEFAULT_INWARD_BUFFER_UM * THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM
)
MAXIMUM_INWARD_BUFFER_SLIDER_VALUE = int(
    MAXIMUM_INWARD_BUFFER_UM * THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM
)
CELL_PROBABILITY_HOVER_DELAY_MS = 2000


def buffer_microns_from_slider(value):
    return float(value) / THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM


def buffer_pixels_from_slider(value):
    return max(
        int(round(buffer_microns_from_slider(value) / DEFAULT_PIXEL_SIZE_UM)),
        0,
    )


def buffer_slider_from_pixels(pixels):
    microns = np.clip(
        float(pixels) * DEFAULT_PIXEL_SIZE_UM,
        0.0,
        MAXIMUM_INWARD_BUFFER_UM,
    )
    return int(round(microns * THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM))


def buffer_distance_label(slider_value):
    microns = buffer_microns_from_slider(slider_value)
    pixels = buffer_pixels_from_slider(slider_value)
    return f"Inward boundary distance: {microns:.1f} µm ({pixels} px)"


class CellHistogramWidget(QWidget):
    """Compact dependency-free histogram with an optional threshold marker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.array([], dtype=float)
        self._minimum = 0.0
        self._maximum = 1.0
        self._marker = None
        self._message = "Load an image and segmentation"
        self.setMinimumHeight(86)
        self.setMaximumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, values, marker=None, value_range=None):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        self._marker = marker
        if values.size == 0:
            self._counts = np.array([], dtype=float)
            self._message = "No cells with pixels in this compartment"
            self.update()
            return

        if value_range is None:
            minimum = float(values.min())
            maximum = float(values.max())
            if np.isclose(minimum, maximum):
                padding = max(abs(minimum) * 0.05, 0.5)
                minimum -= padding
                maximum += padding
        else:
            minimum, maximum = map(float, value_range)
        self._minimum = minimum
        self._maximum = maximum
        self._counts, _ = np.histogram(
            values,
            bins=THRESHOLD_HISTOGRAM_BINS,
            range=(self._minimum, self._maximum),
        )
        self._message = f"n = {values.size:,} cells"
        self.update()

    def set_marker(self, value):
        self._marker = value
        self.update()

    def set_message(self, message):
        self._counts = np.array([], dtype=float)
        self._message = message
        self.update()

    def set_status(self, message):
        self._message = message
        self.update()

    @staticmethod
    def _format_axis_value(value):
        absolute = abs(value)
        if absolute >= 10000 or (0 < absolute < 0.01):
            return f"{value:.1e}"
        return f"{value:.3g}"

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        left, top = 7, 6
        right, bottom = self.width() - 7, self.height() - 20
        plot_width = max(right - left, 1)
        plot_height = max(bottom - top, 1)
        painter.setPen(QPen(QColor("#b0b0b0"), 1))
        painter.drawRect(left, top, plot_width, plot_height)

        if self._counts.size:
            maximum_count = max(float(self._counts.max()), 1.0)
            bar_width = plot_width / self._counts.size
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#6f9fd1"))
            for index, count in enumerate(self._counts):
                if count <= 0:
                    continue
                height = max(int((count / maximum_count) * (plot_height - 2)), 1)
                x0 = left + int(index * bar_width) + 1
                x1 = left + int((index + 1) * bar_width)
                painter.drawRect(
                    x0,
                    bottom - height,
                    max(x1 - x0, 1),
                    height,
                )

            if self._marker is not None and np.isfinite(self._marker):
                position = np.clip(
                    (float(self._marker) - self._minimum)
                    / (self._maximum - self._minimum),
                    0.0,
                    1.0,
                )
                marker_x = left + round(position * plot_width)
                painter.setPen(QPen(QColor("#e67e22"), 2))
                painter.drawLine(marker_x, top, marker_x, bottom)

            painter.setPen(QColor("#555555"))
            painter.drawText(
                left,
                self.height() - 4,
                self._format_axis_value(self._minimum),
            )
            maximum_text = self._format_axis_value(self._maximum)
            maximum_width = painter.fontMetrics().horizontalAdvance(maximum_text)
            painter.drawText(
                max(right - maximum_width, left),
                self.height() - 4,
                maximum_text,
            )

        painter.setPen(QColor("#555555"))
        painter.drawText(
            left + 3,
            top + 3,
            plot_width - 6,
            plot_height - 6,
            Qt.AlignTop | Qt.AlignRight,
            self._message,
        )
        painter.end()


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


class SegmentationWorkerSignals(QObject):
    finished = Signal(object)
    progress = Signal(str)
    error = Signal(str)


class CellposeSegmentationWorker(QRunnable):
    """Run Cellpose-SAM across all loaded images away from the UI thread."""

    def __init__(self, images, marker_names, pixel_size_um):
        super().__init__()
        self.images = list(images)
        self.marker_names = list(marker_names)
        self.pixel_size_um = float(pixel_size_um)
        self.signals = SegmentationWorkerSignals()

    def run(self):
        try:
            results = segment_project_images(
                self.images,
                self.marker_names,
                pixel_size_um=self.pixel_size_um,
                progress_callback=self.signals.progress.emit,
            )
            self.signals.finished.emit(results)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class CudaDetectionWorker(QRunnable):
    """Detect CUDA without delaying construction of the main window."""

    def __init__(self):
        super().__init__()
        self.signals = SegmentationWorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(cuda_compatible_gpu_available())
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class ThresholdWorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)


class ThresholdHistogramSignals(QObject):
    finished = Signal(object)
    error = Signal(object)


class ThresholdHistogramWorker(QRunnable):
    def __init__(
        self,
        request_id,
        image_index,
        image_state,
        channel_name,
        intensity_threshold,
        compartment,
        inward_buffer_pixels,
    ):
        super().__init__()
        self.request_id = request_id
        self.image_index = image_index
        self.image_state = image_state
        self.channel_name = channel_name
        self.intensity_threshold = intensity_threshold
        self.compartment = compartment
        self.inward_buffer_pixels = inward_buffer_pixels
        self.signals = ThresholdHistogramSignals()

    def run(self):
        try:
            channel_names = list(self.image_state["img"].get_channel_names())
            if self.channel_name not in channel_names:
                raise ValueError(
                    f"The image does not contain channel '{self.channel_name}'."
                )
            channel = self.image_state["img"].get_channel(
                channel_names.index(self.channel_name)
            )
            centroid_cache = self.image_state["centroid_cache"]
            result = cell_statistics_by_threshold(
                channel=channel,
                masks=self.image_state["segmentation_masks"],
                cell_data=self.image_state["cell_data"],
                centroid_x=centroid_cache["x"],
                centroid_y=centroid_cache["y"],
                intensity_threshold=self.intensity_threshold,
                compartment=self.compartment,
                inward_buffer_pixels=self.inward_buffer_pixels,
            )
            result.update({
                "request_id": self.request_id,
                "image_index": self.image_index,
                "channel_name": self.channel_name,
                "intensity_threshold": self.intensity_threshold,
                "compartment": self.compartment,
                "inward_buffer_pixels": self.inward_buffer_pixels,
                "total_cells": len(self.image_state["cell_data"]),
            })
            self.signals.finished.emit(result)
        except Exception:
            self.signals.error.emit({
                "request_id": self.request_id,
                "error": traceback.format_exc(),
            })


class ThresholdApplyWorker(QRunnable):
    def __init__(
        self,
        image_states,
        channel_name,
        intensity_threshold,
        positive_pixel_fraction,
        compartment,
        inward_buffer_pixels,
    ):
        super().__init__()
        self.image_states = image_states
        self.channel_name = channel_name
        self.intensity_threshold = intensity_threshold
        self.positive_pixel_fraction = positive_pixel_fraction
        self.compartment = compartment
        self.inward_buffer_pixels = inward_buffer_pixels
        self.signals = ThresholdWorkerSignals()

    def run(self):
        try:
            results = []
            for state in self.image_states:
                channel_names = list(state["img"].get_channel_names())
                if self.channel_name not in channel_names:
                    raise ValueError(
                        f"{Path(state['image_path']).name} does not contain the "
                        f"channel '{self.channel_name}'."
                    )
                channel_index = channel_names.index(self.channel_name)
                channel = state["img"].get_channel(channel_index)
                centroid_cache = state["centroid_cache"]
                result = phenotype_cells_by_threshold(
                    channel=channel,
                    masks=state["segmentation_masks"],
                    cell_data=state["cell_data"],
                    centroid_x=centroid_cache["x"],
                    centroid_y=centroid_cache["y"],
                    intensity_threshold=self.intensity_threshold,
                    positive_pixel_fraction=self.positive_pixel_fraction,
                    compartment=self.compartment,
                    inward_buffer_pixels=self.inward_buffer_pixels,
                )
                result.update({
                    "x": centroid_cache["x"],
                    "y": centroid_cache["y"],
                    "channel_name": self.channel_name,
                    "intensity_threshold": self.intensity_threshold,
                    "positive_pixel_fraction": self.positive_pixel_fraction,
                    "compartment": self.compartment,
                    "inward_buffer_pixels": self.inward_buffer_pixels,
                })
                results.append(result)
            self.signals.finished.emit(results)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class AutomatedPhenotypeWorker(QRunnable):
    """Threshold, select training cells, fit, and apply a random forest."""

    def __init__(
        self,
        image_states,
        channel_name,
        intensity_threshold,
        positive_pixel_fraction,
        compartment,
        inward_buffer_pixels,
        feature_columns,
        phenotype_name,
        manual_training,
        excluded_rows,
        random_seed,
    ):
        super().__init__()
        self.image_states = image_states
        self.channel_name = channel_name
        self.intensity_threshold = intensity_threshold
        self.positive_pixel_fraction = positive_pixel_fraction
        self.compartment = compartment
        self.inward_buffer_pixels = inward_buffer_pixels
        self.feature_columns = feature_columns
        self.phenotype_name = phenotype_name
        self.manual_training = manual_training
        self.excluded_rows = excluded_rows
        self.random_seed = random_seed
        self.signals = ThresholdWorkerSignals()

    def run(self):
        try:
            threshold_results = []
            offsets = [0]
            for state in self.image_states:
                channel_names = list(state["img"].get_channel_names())
                if self.channel_name not in channel_names:
                    raise ValueError(
                        f"{Path(state['image_path']).name} does not contain the "
                        f"channel '{self.channel_name}'."
                    )
                channel = state["img"].get_channel(
                    channel_names.index(self.channel_name)
                )
                centroids = state["centroid_cache"]
                result = phenotype_cells_by_threshold(
                    channel=channel,
                    masks=state["segmentation_masks"],
                    cell_data=state["cell_data"],
                    centroid_x=centroids["x"],
                    centroid_y=centroids["y"],
                    intensity_threshold=self.intensity_threshold,
                    positive_pixel_fraction=self.positive_pixel_fraction,
                    compartment=self.compartment,
                    inward_buffer_pixels=self.inward_buffer_pixels,
                )
                result.update({
                    "x": centroids["x"],
                    "y": centroids["y"],
                    "channel_name": self.channel_name,
                    "intensity_threshold": self.intensity_threshold,
                    "positive_pixel_fraction": self.positive_pixel_fraction,
                    "compartment": self.compartment,
                    "inward_buffer_pixels": self.inward_buffer_pixels,
                })
                threshold_results.append(result)
                offsets.append(offsets[-1] + len(state["cell_data"]))

            all_fractions = np.concatenate([
                result["positive_fraction"] for result in threshold_results
            ])
            all_fluorescence = np.concatenate([
                result["mean_intensity"] for result in threshold_results
            ])
            excluded_global = {
                offsets[image_index] + int(row_index)
                for image_index, row_index in self.excluded_rows
            }
            manual_global = []
            for item in self.manual_training:
                global_index = offsets[item["image_index"]] + item["row_index"]
                manual_global.append((global_index, item["label"]))
                excluded_global.add(global_index)

            manual_positive = sum(
                label == "positive" for _, label in manual_global
            )
            manual_negative = sum(
                label == "negative" for _, label in manual_global
            )
            requested_positive = max(
                AUTOMATED_POSITIVE_TRAINING_COUNT - manual_positive, 0
            )
            requested_negative = max(
                AUTOMATED_NEGATIVE_TRAINING_COUNT - manual_negative, 0
            )
            requested_low_negative = int(round(
                requested_negative
                * AUTOMATED_LOW_NEGATIVE_TRAINING_COUNT
                / AUTOMATED_NEGATIVE_TRAINING_COUNT
            ))
            requested_low_negative = min(
                requested_low_negative,
                AUTOMATED_LOW_NEGATIVE_TRAINING_COUNT,
            )
            requested_mid_negative = (
                requested_negative - requested_low_negative
            )
            selected = select_automated_training_indices(
                all_fractions,
                positive_pixel_fraction=self.positive_pixel_fraction,
                low_negative_count=requested_low_negative,
                mid_negative_count=requested_mid_negative,
                positive_count=requested_positive,
                top_positive_fraction=AUTOMATED_TOP_POSITIVE_FRACTION,
                fluorescence_values=all_fluorescence,
                excluded_indices=np.asarray(sorted(excluded_global), dtype=np.int64),
                random_seed=self.random_seed,
            )

            image_for_global = np.concatenate([
                np.full(len(state["cell_data"]), image_index, dtype=np.int64)
                for image_index, state in enumerate(self.image_states)
            ])
            row_for_global = np.concatenate([
                np.arange(len(state["cell_data"]), dtype=np.int64)
                for state in self.image_states
            ])
            auto_annotations = [dict() for _ in self.image_states]
            training_references = list(manual_global)

            def add_automated_annotations(label, global_indices):
                for global_index in global_indices:
                    global_index = int(global_index)
                    image_index = int(image_for_global[global_index])
                    row_index = int(row_for_global[global_index])
                    threshold_result = threshold_results[image_index]
                    cell_id = str(int(threshold_result["mask_label"][row_index]))
                    auto_annotations[image_index][cell_id] = {
                        "cell_id": cell_id,
                        "label": label,
                        "centroid_x": float(threshold_result["x"][row_index]),
                        "centroid_y": float(threshold_result["y"][row_index]),
                        "row_index": row_index,
                        "source": "automated",
                    }
                    training_references.append((global_index, label))

            for label, global_indices in (
                ("negative", selected["negative"]),
                ("positive", selected["positive"]),
            ):
                add_automated_annotations(label, global_indices)

            def fit_automated_model(references):
                rows = []
                targets = []
                for global_index, label in references:
                    image_index = int(image_for_global[global_index])
                    row_index = int(row_for_global[global_index])
                    rows.append(
                        self.image_states[image_index]["cell_data"].iloc[
                            row_index
                        ][self.feature_columns]
                    )
                    targets.append(1 if label == "positive" else 0)
                if set(targets) != {0, 1}:
                    raise ValueError(
                        "Automated phenotyping requires both positive and "
                        "negative training cells."
                    )

                training_data = pd.DataFrame(
                    rows, columns=self.feature_columns
                )
                pipeline, usable_features = fit_random_forest(
                    training_data=training_data,
                    targets=targets,
                    feature_columns=self.feature_columns,
                    random_seed=self.random_seed,
                )
                return pipeline, usable_features, targets

            # First fit: the original balanced 25-positive/25-negative set.
            initial_pipeline, initial_features, _ = fit_automated_model(
                training_references
            )
            initial_calls = []
            initial_positive_probabilities = []
            for state in self.image_states:
                measurements = state["cell_data"][initial_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                calls, probabilities = model_calls_and_positive_probabilities(
                    initial_pipeline, measurements
                )
                initial_calls.append(calls)
                initial_positive_probabilities.append(probabilities)

            first_refinement_excluded = excluded_global | {
                int(global_index) for global_index, _ in training_references
            }
            first_refinement_seed = random_seed_for_stage(
                1, self.random_seed
            )
            first_refinement = select_automated_refinement_indices(
                predicted_positive=np.concatenate(initial_calls),
                positive_probabilities=np.concatenate(
                    initial_positive_probabilities
                ),
                fluorescence_values=all_fluorescence,
                positive_count=AUTOMATED_REFINEMENT_TRAINING_COUNT,
                negative_count=AUTOMATED_REFINEMENT_TRAINING_COUNT,
                maximum_call_probability=(
                    AUTOMATED_REFINEMENT_MAX_CALL_PROBABILITY
                ),
                fluorescence_tail_fraction=(
                    AUTOMATED_REFINEMENT_FLUORESCENCE_FRACTION
                ),
                excluded_indices=np.asarray(
                    sorted(first_refinement_excluded), dtype=np.int64
                ),
                random_seed=first_refinement_seed,
            )
            # High-fluorescence uncertain negatives teach the positive class;
            # low-fluorescence uncertain positives teach the negative class.
            add_automated_annotations(
                "positive", first_refinement["positive"]
            )
            add_automated_annotations(
                "negative", first_refinement["negative"]
            )

            # Second fit: retrain with 30 examples per class, then identify a
            # second set of harder examples using a stricter 55% confidence.
            second_pipeline, second_features, _ = fit_automated_model(
                training_references
            )
            second_calls = []
            second_positive_probabilities = []
            for state in self.image_states:
                measurements = state["cell_data"][second_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                calls, probabilities = model_calls_and_positive_probabilities(
                    second_pipeline, measurements
                )
                second_calls.append(calls)
                second_positive_probabilities.append(probabilities)

            second_refinement_excluded = excluded_global | {
                int(global_index) for global_index, _ in training_references
            }
            second_refinement_seed = random_seed_for_stage(
                2, self.random_seed
            )
            second_refinement = select_automated_refinement_indices(
                predicted_positive=np.concatenate(second_calls),
                positive_probabilities=np.concatenate(
                    second_positive_probabilities
                ),
                fluorescence_values=all_fluorescence,
                positive_count=AUTOMATED_REFINEMENT_TRAINING_COUNT,
                negative_count=AUTOMATED_REFINEMENT_TRAINING_COUNT,
                maximum_call_probability=(
                    AUTOMATED_SECOND_REFINEMENT_MAX_CALL_PROBABILITY
                ),
                fluorescence_tail_fraction=(
                    AUTOMATED_REFINEMENT_FLUORESCENCE_FRACTION
                ),
                excluded_indices=np.asarray(
                    sorted(second_refinement_excluded), dtype=np.int64
                ),
                random_seed=second_refinement_seed,
            )
            add_automated_annotations(
                "positive", second_refinement["positive"]
            )
            add_automated_annotations(
                "negative", second_refinement["negative"]
            )

            # Third and final fit: train from scratch on 35 examples per class.
            pipeline, usable_features, targets = fit_automated_model(
                training_references
            )
            model_bundle = {
                "format": MODEL_FORMAT,
                "version": MODEL_VERSION,
                "phenotype_name": self.phenotype_name,
                "feature_columns": usable_features,
                "algorithm": RANDOM_FOREST_ALGORITHM,
                "training_samples": len(targets),
                "pipeline": pipeline,
                "automated": {
                    "channel_name": self.channel_name,
                    "intensity_threshold": self.intensity_threshold,
                    "positive_pixel_fraction": self.positive_pixel_fraction,
                    "compartment": self.compartment,
                    "inward_buffer_pixels": self.inward_buffer_pixels,
                    "negative_low_fluorescence_fraction": 0.50,
                    "negative_mid_fluorescence_range": [0.51, 0.80],
                    "top_positive_fraction": AUTOMATED_TOP_POSITIVE_FRACTION,
                    "refinement_max_call_probability": (
                        AUTOMATED_REFINEMENT_MAX_CALL_PROBABILITY
                    ),
                    "refinement_fluorescence_tail_fraction": (
                        AUTOMATED_REFINEMENT_FLUORESCENCE_FRACTION
                    ),
                    "refinement_positive_count": len(
                        first_refinement["positive"]
                    ),
                    "refinement_negative_count": len(
                        first_refinement["negative"]
                    ),
                    "second_refinement_max_call_probability": (
                        AUTOMATED_SECOND_REFINEMENT_MAX_CALL_PROBABILITY
                    ),
                    "second_refinement_positive_count": len(
                        second_refinement["positive"]
                    ),
                    "second_refinement_negative_count": len(
                        second_refinement["negative"]
                    ),
                    "random_seed": self.random_seed,
                },
            }

            model_predictions = []
            for state in self.image_states:
                measurements = state["cell_data"][usable_features].apply(
                    pd.to_numeric, errors="coerce"
                )
                prediction, positive_probability = (
                    model_calls_and_positive_probabilities(
                        pipeline, measurements
                    )
                )
                centroids = state["centroid_cache"]
                model_predictions.append({
                    "x": centroids["x"],
                    "y": centroids["y"],
                    "positive": prediction,
                    "positive_probability": positive_probability,
                })

            self.signals.finished.emit({
                "threshold_results": threshold_results,
                "auto_annotations": auto_annotations,
                "model_bundle": model_bundle,
                "model_predictions": model_predictions,
                "automatic_positive_count": (
                    len(selected["positive"])
                    + len(first_refinement["positive"])
                    + len(second_refinement["positive"])
                ),
                "automatic_negative_count": (
                    len(selected["negative"])
                    + len(first_refinement["negative"])
                    + len(second_refinement["negative"])
                ),
                "manual_positive_count": manual_positive,
                "manual_negative_count": manual_negative,
            })
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
        self.active_tool = "automated"
        self.cellpose_worker = None
        self.cuda_detection_worker = None
        self.cuda_gpu_available = None
        self.segmenting_selected_markers = set()
        self.released_generated_segmentations = {}
        self.automated_worker = None
        self.automated_edit_mode = False
        self.threshold_intensity_value = None
        self.threshold_channel_name = None
        self.threshold_worker = None
        self.threshold_histogram_worker = None
        self.threshold_histogram_request_id = 0
        self.threshold_histogram_timer = QTimer(self)
        self.threshold_histogram_timer.setSingleShot(True)
        self.threshold_histogram_timer.timeout.connect(
            self._start_threshold_histogram_worker
        )
        self.cell_probability_hover_timer = QTimer(self)
        self.cell_probability_hover_timer.setSingleShot(True)
        self.cell_probability_hover_timer.setInterval(
            CELL_PROBABILITY_HOVER_DELAY_MS
        )
        self.cell_probability_hover_timer.timeout.connect(
            self.show_hovered_cell_probability
        )
        self.hovered_prediction_key = None
        self.hovered_prediction = None
        self.hover_global_position = None
        self.hover_probability_visible = False

        # Segmentation data remain in whole-slide pixel coordinates. Only the
        # current FOV is cropped and converted to a boundary overlay.
        self.cell_data = None
        self.segmentation_masks = None
        self.cell_data_path = None
        self.segmentation_mask_path = None

        self.image_label = NapariImageCanvas("Select a TIFF or OME-Zarr image")
        self.image_label.image_clicked.connect(self.label_clicked_cell)
        self.image_label.image_hovered.connect(self.track_cell_probability_hover)
        self.image_label.image_hover_left.connect(self.cancel_cell_probability_hover)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(700, 700)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("""
            QWidget { background-color: black; border: 1px solid #333; }
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
        self.channel_dropdown.currentIndexChanged.connect(self.on_channel_changed)
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
        self.positive_annotations_checkbox.setStyleSheet("color: #00afd1;")
        self.positive_annotations_checkbox.stateChanged.connect(self.update_display)
        self.positive_annotations_checkbox.stateChanged.connect(
            self.sync_automated_annotation_visibility
        )
        self.negative_annotations_checkbox = QCheckBox("Show Negative")
        self.negative_annotations_checkbox.setChecked(True)
        self.negative_annotations_checkbox.setStyleSheet("color: #e02020;")
        self.negative_annotations_checkbox.stateChanged.connect(self.update_display)
        self.negative_annotations_checkbox.stateChanged.connect(
            self.sync_automated_annotation_visibility
        )
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
        self.modelled_phenotypes_checkbox.stateChanged.connect(
            self.sync_automated_model_visibility
        )
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

        random_forest_page = QWidget()
        random_forest_layout = QVBoxLayout(random_forest_page)
        random_forest_layout.setContentsMargins(0, 0, 0, 0)
        random_forest_layout.addWidget(training_panel)
        random_forest_layout.addWidget(model_panel)

        self.threshold_phenotype_name = QLineEdit()
        self.threshold_phenotype_name.setPlaceholderText("e.g. CD8-positive")
        self.automated_phenotype_name = QLineEdit()
        self.automated_phenotype_name.setPlaceholderText("e.g. CD8-positive")
        self.phenotype_name.textChanged.connect(
            self.sync_phenotype_name_from_random_forest
        )
        self.threshold_phenotype_name.textChanged.connect(
            self.sync_phenotype_name_from_threshold
        )
        self.automated_phenotype_name.textChanged.connect(
            self.sync_phenotype_name_from_automated
        )

        threshold_description = QLabel(
            "Pixels in the displayed channel above the intensity threshold "
            "are highlighted yellow."
        )
        threshold_description.setWordWrap(True)
        self.threshold_intensity_label = QLabel("Intensity threshold: —")
        self.threshold_intensity_slider = QSlider(Qt.Horizontal)
        self.threshold_intensity_slider.setRange(0, 1000)
        self.threshold_intensity_slider.setValue(500)
        self.threshold_intensity_slider.valueChanged.connect(
            self.threshold_slider_changed
        )
        self.threshold_intensity_histogram_label = QLabel(
            "All-image mean fluorescence per cell"
        )
        self.threshold_intensity_histogram_label.setWordWrap(True)
        self.threshold_intensity_histogram = CellHistogramWidget()
        self.threshold_intensity_histogram.setToolTip(
            "Distribution of mean fluorescence intensity per segmented cell "
            "for the selected channel and compartment across the entire "
            "current image. The orange line is the intensity threshold."
        )
        self.threshold_percent_label = QLabel(
            "Positive pixels required: >25%"
        )
        self.threshold_percent_slider = QSlider(Qt.Horizontal)
        self.threshold_percent_slider.setRange(1, 100)
        self.threshold_percent_slider.setValue(25)
        self.threshold_percent_slider.valueChanged.connect(
            self.threshold_percent_changed
        )
        self.threshold_fraction_histogram_label = QLabel(
            "All-image positive-pixel percentages per cell"
        )
        self.threshold_fraction_histogram_label.setWordWrap(True)
        self.threshold_fraction_histogram = CellHistogramWidget()
        self.threshold_fraction_histogram.setToolTip(
            "Distribution of the percentage of above-threshold pixels per "
            "segmented cell across the entire current image. The orange line "
            "is the percentage required to call a cell positive."
        )
        self.threshold_compartment_label = QLabel(
            "Positive-pixel denominator:"
        )
        self.threshold_nucleus_checkbox = QCheckBox("Nucleus")
        self.threshold_nucleus_checkbox.setChecked(True)
        self.threshold_nucleus_checkbox.setToolTip(
            "Use pixels deeper than the inward boundary distance."
        )
        self.threshold_nucleus_checkbox.stateChanged.connect(
            self.threshold_compartment_changed
        )
        self.threshold_cytoplasm_checkbox = QCheckBox(
            "Cytoplasm/Membrane"
        )
        self.threshold_cytoplasm_checkbox.setChecked(True)
        self.threshold_cytoplasm_checkbox.setToolTip(
            "Use the inward band measured from each cell boundary."
        )
        self.threshold_cytoplasm_checkbox.stateChanged.connect(
            self.threshold_compartment_changed
        )
        self.threshold_mask_button = QPushButton("Threshold Mask: On")
        self.threshold_mask_button.setCheckable(True)
        self.threshold_mask_button.setChecked(True)
        self.threshold_mask_button.setToolTip(
            "Show or hide the yellow above-threshold pixel mask."
        )
        self.threshold_mask_button.toggled.connect(
            self.threshold_mask_toggled
        )
        self.threshold_buffer_label = QLabel(
            buffer_distance_label(DEFAULT_INWARD_BUFFER_SLIDER_VALUE)
        )
        self.threshold_buffer_slider = QSlider(Qt.Horizontal)
        self.threshold_buffer_slider.setRange(
            0, MAXIMUM_INWARD_BUFFER_SLIDER_VALUE
        )
        self.threshold_buffer_slider.setSingleStep(1)
        self.threshold_buffer_slider.setPageStep(5)
        self.threshold_buffer_slider.setValue(
            DEFAULT_INWARD_BUFFER_SLIDER_VALUE
        )
        self.threshold_buffer_slider.setToolTip(
            "Inward boundary distance from 0.0 to 5.0 µm in 0.1 µm steps."
        )
        self.threshold_buffer_slider.valueChanged.connect(
            self.threshold_buffer_changed
        )
        self.apply_threshold_button = QPushButton(
            "Apply Threshold to All Cells"
        )
        self.apply_threshold_button.clicked.connect(
            self.apply_threshold_to_all_cells
        )
        self.threshold_phenotypes_checkbox = QCheckBox(
            "Show Threshold Phenotypes"
        )
        self.threshold_phenotypes_checkbox.setChecked(True)
        self.threshold_phenotypes_checkbox.stateChanged.connect(
            self.update_display
        )
        self.threshold_positive_count_label = QLabel("Threshold positive: 0")
        self.threshold_negative_count_label = QLabel("Threshold negative: 0")
        self.export_threshold_phenotypes_button = QPushButton(
            "Export Cell Phenotypes"
        )
        self.export_threshold_phenotypes_button.setToolTip(
            "Export every original Cellpose TSV column and the threshold-derived "
            "Positive/Negative phenotype labels for every loaded image"
        )
        self.export_threshold_phenotypes_button.clicked.connect(
            self.export_cell_phenotypes
        )

        threshold_panel = QGroupBox("Threshold Phenotyping")
        threshold_panel.setMinimumWidth(220)
        threshold_panel.setMaximumWidth(300)
        threshold_layout = QVBoxLayout()
        threshold_name_layout = QFormLayout()
        threshold_name_layout.addRow(
            "Phenotype:", self.threshold_phenotype_name
        )
        threshold_layout.addLayout(threshold_name_layout)
        threshold_layout.addWidget(threshold_description)
        threshold_layout.addWidget(self.threshold_mask_button)
        threshold_layout.addSpacing(8)
        threshold_layout.addWidget(self.threshold_intensity_histogram_label)
        threshold_layout.addWidget(self.threshold_intensity_histogram)
        threshold_layout.addWidget(self.threshold_intensity_label)
        threshold_layout.addWidget(self.threshold_intensity_slider)
        threshold_layout.addSpacing(8)
        threshold_layout.addWidget(self.threshold_fraction_histogram_label)
        threshold_layout.addWidget(self.threshold_fraction_histogram)
        threshold_layout.addWidget(self.threshold_percent_label)
        threshold_layout.addWidget(self.threshold_percent_slider)
        threshold_layout.addSpacing(8)
        threshold_layout.addWidget(self.threshold_compartment_label)
        threshold_layout.addWidget(self.threshold_nucleus_checkbox)
        threshold_layout.addWidget(self.threshold_cytoplasm_checkbox)
        threshold_layout.addWidget(self.threshold_buffer_label)
        threshold_layout.addWidget(self.threshold_buffer_slider)
        threshold_layout.addWidget(self.apply_threshold_button)
        threshold_layout.addWidget(self.threshold_phenotypes_checkbox)
        threshold_layout.addWidget(self.threshold_positive_count_label)
        threshold_layout.addWidget(self.threshold_negative_count_label)
        threshold_layout.addStretch()
        threshold_layout.addWidget(self.export_threshold_phenotypes_button)
        threshold_panel.setLayout(threshold_layout)

        threshold_page = QWidget()
        threshold_page_layout = QVBoxLayout(threshold_page)
        threshold_page_layout.setContentsMargins(0, 0, 0, 0)
        threshold_page_layout.addWidget(threshold_panel)
        threshold_page_layout.addStretch()

        automated_description = QLabel(
            "Automatically thresholds pixels by the displayed fluorescence " \
            "channel, then builds a random forest classifier using 25 " \
            "positive and 25 negative training cells based on positive " \
            "pixel proportions followed by two low-confidence refinement " \
            "rounds, adding five fluorescence-discordant cells to each " \
            "class per round. The final 35-positive/35-negative random " \
            "forest is applied to every loaded image."
        )
        automated_description.setWordWrap(True)
        self.auto_phenotype_button = QPushButton("Auto Phenotype")
        automated_description.setAlignment(Qt.AlignmentFlag.AlignJustify)
        self.auto_phenotype_button.clicked.connect(
            lambda: self.start_automated_phenotyping(reset_annotations=True)
        )
        self.automated_status_label = QLabel("Ready for automated phenotyping.")
        self.automated_status_label.setWordWrap(True)
        self.automated_edit_button = QPushButton("Edit")
        self.automated_edit_button.clicked.connect(self.open_automated_edit)
        self.automated_modelled_checkbox = QCheckBox(
            "Show Modelled Phenotypes"
        )
        self.automated_modelled_checkbox.setChecked(True)
        self.automated_modelled_checkbox.stateChanged.connect(
            self.automated_model_visibility_changed
        )
        self.automated_model_positive_count_label = QLabel("Model positive: 0")
        self.automated_model_negative_count_label = QLabel("Model negative: 0")
        self.automated_export_button = QPushButton("Export Cell Phenotypes")
        self.automated_export_button.clicked.connect(self.export_cell_phenotypes)

        automated_default_panel = QGroupBox("Automated Phenotyping")
        automated_default_layout = QVBoxLayout()
        automated_default_layout.addWidget(automated_description)
        automated_default_layout.addSpacing(8)
        automated_default_layout.addWidget(self.auto_phenotype_button)
        automated_default_layout.addWidget(self.automated_status_label)
        automated_default_layout.addWidget(self.automated_edit_button)
        automated_default_layout.addWidget(self.automated_modelled_checkbox)
        automated_default_layout.addWidget(
            self.automated_model_positive_count_label
        )
        automated_default_layout.addWidget(
            self.automated_model_negative_count_label
        )
        automated_default_layout.addStretch()
        automated_default_layout.addWidget(self.automated_export_button)
        automated_default_panel.setLayout(automated_default_layout)
        automated_default_page = QWidget()
        automated_default_page_layout = QVBoxLayout(automated_default_page)
        automated_default_page_layout.setContentsMargins(0, 0, 0, 0)
        automated_default_page_layout.addWidget(automated_default_panel)

        self.automated_intensity_label = QLabel(
            "Intensity threshold: 66% of display range"
        )
        self.automated_intensity_slider = QSlider(Qt.Horizontal)
        self.automated_intensity_slider.setRange(0, 1000)
        self.automated_intensity_slider.setValue(
            AUTOMATED_INTENSITY_SLIDER_VALUE
        )
        self.automated_intensity_slider.valueChanged.connect(
            self.automated_intensity_changed
        )
        self.automated_percent_label = QLabel(
            "Positive pixels required: >15%"
        )
        self.automated_percent_slider = QSlider(Qt.Horizontal)
        self.automated_percent_slider.setRange(1, 100)
        self.automated_percent_slider.setValue(
            AUTOMATED_POSITIVE_PIXEL_PERCENT
        )
        self.automated_percent_slider.valueChanged.connect(
            self.automated_percent_changed
        )
        self.automated_nucleus_checkbox = QCheckBox("Nucleus")
        self.automated_nucleus_checkbox.setChecked(True)
        self.automated_nucleus_checkbox.stateChanged.connect(
            self.automated_compartment_changed
        )
        self.automated_cytoplasm_checkbox = QCheckBox("Cytoplasm/Membrane")
        self.automated_cytoplasm_checkbox.setChecked(True)
        self.automated_cytoplasm_checkbox.stateChanged.connect(
            self.automated_compartment_changed
        )
        self.automated_threshold_mask_button = QPushButton(
            "Threshold Mask: On"
        )
        self.automated_threshold_mask_button.setCheckable(True)
        self.automated_threshold_mask_button.setChecked(True)
        self.automated_threshold_mask_button.setToolTip(
            "Show or hide the yellow above-threshold pixel mask."
        )
        self.automated_threshold_mask_button.toggled.connect(
            self.automated_threshold_mask_toggled
        )
        self.automated_buffer_label = QLabel(
            buffer_distance_label(DEFAULT_INWARD_BUFFER_SLIDER_VALUE)
        )
        self.automated_buffer_slider = QSlider(Qt.Horizontal)
        self.automated_buffer_slider.setRange(
            0, MAXIMUM_INWARD_BUFFER_SLIDER_VALUE
        )
        self.automated_buffer_slider.setSingleStep(1)
        self.automated_buffer_slider.setPageStep(5)
        self.automated_buffer_slider.setValue(
            DEFAULT_INWARD_BUFFER_SLIDER_VALUE
        )
        self.automated_buffer_slider.setToolTip(
            "Inward boundary distance from 0.0 to 5.0 µm in 0.1 µm steps."
        )
        self.automated_buffer_slider.valueChanged.connect(
            self.automated_buffer_changed
        )

        automated_threshold_panel = QGroupBox("Threshold Settings")
        automated_threshold_layout = QVBoxLayout()
        automated_threshold_layout.addWidget(
            self.automated_threshold_mask_button
        )
        automated_threshold_layout.addWidget(self.automated_intensity_label)
        automated_threshold_layout.addWidget(self.automated_intensity_slider)
        automated_threshold_layout.addWidget(self.automated_percent_label)
        automated_threshold_layout.addWidget(self.automated_percent_slider)
        automated_threshold_layout.addWidget(QLabel("Positive-pixel denominator:"))
        automated_threshold_layout.addWidget(self.automated_nucleus_checkbox)
        automated_threshold_layout.addWidget(self.automated_cytoplasm_checkbox)
        automated_threshold_layout.addWidget(self.automated_buffer_label)
        automated_threshold_layout.addWidget(self.automated_buffer_slider)
        automated_threshold_panel.setLayout(automated_threshold_layout)

        self.automated_positive_checkbox = QCheckBox("Show Positive")
        self.automated_positive_checkbox.setChecked(True)
        self.automated_positive_checkbox.setStyleSheet("color: #00afd1;")
        self.automated_positive_checkbox.stateChanged.connect(
            self.automated_annotation_visibility_changed
        )
        self.automated_negative_checkbox = QCheckBox("Show Negative")
        self.automated_negative_checkbox.setChecked(True)
        self.automated_negative_checkbox.setStyleSheet("color: #e02020;")
        self.automated_negative_checkbox.stateChanged.connect(
            self.automated_annotation_visibility_changed
        )
        self.automated_positive_count_label = QLabel("Positive: 0")
        self.automated_negative_count_label = QLabel("Negative: 0")
        self.automated_previous_positive_button = QPushButton("←")
        self.automated_previous_positive_button.setFixedWidth(36)
        self.automated_previous_positive_button.clicked.connect(
            lambda: self.navigate_training("positive", -1)
        )
        self.automated_next_positive_button = QPushButton("→")
        self.automated_next_positive_button.setFixedWidth(36)
        self.automated_next_positive_button.clicked.connect(
            lambda: self.navigate_training("positive", 1)
        )
        self.automated_positive_position_label = QLabel("0 / 0")
        self.automated_positive_position_label.setAlignment(Qt.AlignCenter)
        self.automated_previous_negative_button = QPushButton("←")
        self.automated_previous_negative_button.setFixedWidth(36)
        self.automated_previous_negative_button.clicked.connect(
            lambda: self.navigate_training("negative", -1)
        )
        self.automated_next_negative_button = QPushButton("→")
        self.automated_next_negative_button.setFixedWidth(36)
        self.automated_next_negative_button.clicked.connect(
            lambda: self.navigate_training("negative", 1)
        )
        self.automated_negative_position_label = QLabel("0 / 0")
        self.automated_negative_position_label.setAlignment(Qt.AlignCenter)
        automated_positive_navigation = QHBoxLayout()
        automated_positive_navigation.addWidget(
            self.automated_previous_positive_button
        )
        automated_positive_navigation.addWidget(
            self.automated_positive_position_label, stretch=1
        )
        automated_positive_navigation.addWidget(
            self.automated_next_positive_button
        )
        automated_negative_navigation = QHBoxLayout()
        automated_negative_navigation.addWidget(
            self.automated_previous_negative_button
        )
        automated_negative_navigation.addWidget(
            self.automated_negative_position_label, stretch=1
        )
        automated_negative_navigation.addWidget(
            self.automated_next_negative_button
        )

        automated_training_panel = QGroupBox("Random-Forest Training")
        automated_training_layout = QVBoxLayout()
        automated_training_name_layout = QFormLayout()
        automated_training_name_layout.addRow(
            "Phenotype:", self.automated_phenotype_name
        )
        automated_training_layout.addLayout(automated_training_name_layout)
        automated_training_layout.addWidget(self.automated_positive_checkbox)
        automated_training_layout.addWidget(self.automated_negative_checkbox)
        automated_training_layout.addWidget(self.automated_positive_count_label)
        automated_training_layout.addLayout(automated_positive_navigation)
        automated_training_layout.addWidget(self.automated_negative_count_label)
        automated_training_layout.addLayout(automated_negative_navigation)
        automated_training_panel.setLayout(automated_training_layout)

        self.automated_edit_status_label = QLabel("Edit labels or thresholds.")
        self.automated_edit_status_label.setWordWrap(True)
        self.automated_edit_modelled_checkbox = QCheckBox(
            "Show Modelled Phenotypes"
        )
        self.automated_edit_modelled_checkbox.setChecked(True)
        self.automated_edit_modelled_checkbox.stateChanged.connect(
            self.automated_model_visibility_changed
        )
        self.automated_edit_model_positive_count_label = QLabel(
            "Model positive: 0"
        )
        self.automated_edit_model_negative_count_label = QLabel(
            "Model negative: 0"
        )
        self.rephenotype_button = QPushButton("Re-Phenotype")
        self.rephenotype_button.clicked.connect(
            lambda: self.start_automated_phenotyping(reset_annotations=False)
        )
        automated_edit_model_panel = QGroupBox("Random-Forest Model")
        automated_edit_model_layout = QVBoxLayout()
        automated_edit_model_layout.addWidget(self.automated_edit_status_label)
        automated_edit_model_layout.addWidget(
            self.automated_edit_modelled_checkbox
        )
        automated_edit_model_layout.addWidget(
            self.automated_edit_model_positive_count_label
        )
        automated_edit_model_layout.addWidget(
            self.automated_edit_model_negative_count_label
        )
        automated_edit_model_layout.addWidget(self.rephenotype_button)
        automated_edit_model_panel.setLayout(automated_edit_model_layout)

        automated_edit_content = QWidget()
        automated_edit_layout = QVBoxLayout(automated_edit_content)
        automated_edit_layout.setContentsMargins(0, 0, 0, 0)
        automated_edit_layout.addWidget(automated_threshold_panel)
        automated_edit_layout.addWidget(automated_training_panel)
        automated_edit_layout.addWidget(automated_edit_model_panel)
        automated_edit_layout.addStretch()
        automated_edit_scroll = QScrollArea()
        automated_edit_scroll.setWidgetResizable(True)
        automated_edit_scroll.setWidget(automated_edit_content)

        self.automated_panel_stack = QStackedWidget()
        self.automated_panel_stack.addWidget(automated_default_page)
        self.automated_panel_stack.addWidget(automated_edit_scroll)
        automated_page = QWidget()
        automated_page_layout = QVBoxLayout(automated_page)
        automated_page_layout.setContentsMargins(0, 0, 0, 0)
        automated_page_layout.addWidget(self.automated_panel_stack)

        self.segmenting_gpu_status_label = QLabel(
            "● Checking for a CUDA-compatible GPU..."
        )
        self.segmenting_gpu_status_label.setWordWrap(True)
        self.segmenting_gpu_status_label.setStyleSheet(
            "color: #b26a00; font-weight: bold;"
        )
        segmenting_description = QLabel(
            "Select one or more membrane-guiding markers shared by the loaded "
            "images. Selected channels are normalized and merged before "
            f"segmentation with Cellpose-SAM ({CELLPOSE_SAM_MODEL})."
        )
        segmenting_description.setWordWrap(True)
        self.segmenting_dapi_label = QLabel(
            "DAPI will be supplied as the nuclear channel when available."
        )
        self.segmenting_dapi_label.setWordWrap(True)
        self.segmenting_marker_content = QWidget()
        self.segmenting_marker_layout = QVBoxLayout(
            self.segmenting_marker_content
        )
        self.segmenting_marker_layout.setContentsMargins(4, 4, 4, 4)
        self.segmenting_marker_layout.setSpacing(3)
        self.segmenting_marker_checkboxes = []
        self.segmenting_marker_scroll = QScrollArea()
        self.segmenting_marker_scroll.setWidgetResizable(True)
        self.segmenting_marker_scroll.setMinimumHeight(220)
        self.segmenting_marker_scroll.setWidget(
            self.segmenting_marker_content
        )
        self.segment_button = QPushButton("Segment")
        self.segment_button.setToolTip(
            "Replace segmentation and cell-level measurements for every "
            "loaded image using the selected markers."
        )
        self.segment_button.clicked.connect(self.start_cellpose_segmentation)
        self.segmenting_status_label = QLabel(
            "Load images, select membrane markers, then click Segment."
        )
        self.segmenting_status_label.setWordWrap(True)

        segmenting_panel = QGroupBox("CellPoseSAM Segmentation")
        segmenting_layout = QVBoxLayout()
        segmenting_layout.addWidget(self.segmenting_gpu_status_label)
        segmenting_layout.addSpacing(6)
        segmenting_layout.addWidget(segmenting_description)
        segmenting_layout.addWidget(self.segmenting_dapi_label)
        segmenting_layout.addSpacing(6)
        segmenting_layout.addWidget(QLabel("Membrane-guiding markers:"))
        segmenting_layout.addWidget(self.segmenting_marker_scroll, stretch=1)
        segmenting_layout.addWidget(self.segment_button)
        segmenting_layout.addWidget(self.segmenting_status_label)
        segmenting_panel.setLayout(segmenting_layout)
        segmenting_page = QWidget()
        segmenting_page_layout = QVBoxLayout(segmenting_page)
        segmenting_page_layout.setContentsMargins(0, 0, 0, 0)
        segmenting_page_layout.addWidget(segmenting_panel)

        self.right_panel_stack = QStackedWidget()
        self.right_panel_stack.setMinimumWidth(220)
        self.right_panel_stack.setMaximumWidth(330)
        self.right_panel_stack.addWidget(segmenting_page)
        self.right_panel_stack.addWidget(random_forest_page)
        self.right_panel_stack.addWidget(threshold_page)
        self.right_panel_stack.addWidget(automated_page)

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

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        segmenting_menu = self.menu_bar.addMenu("&Segmenting")
        self.cellpose_sam_tool_action = QAction("CellPoseSAM", self)
        self.cellpose_sam_tool_action.setCheckable(True)
        self.cellpose_sam_tool_action.triggered.connect(
            lambda: self.set_tool_mode("cellpose_sam")
        )
        self.tool_action_group.addAction(self.cellpose_sam_tool_action)
        segmenting_menu.addAction(self.cellpose_sam_tool_action)

        phenotyping_menu = self.menu_bar.addMenu("&Phenotyping")
        self.random_forest_tool_action = QAction("Random Forest", self)
        self.random_forest_tool_action.setCheckable(True)
        self.random_forest_tool_action.triggered.connect(
            lambda: self.set_tool_mode("random_forest")
        )
        self.threshold_tool_action = QAction("Threshold Slider", self)
        self.threshold_tool_action.setCheckable(True)
        self.threshold_tool_action.triggered.connect(
            lambda: self.set_tool_mode("threshold")
        )
        self.automated_tool_action = QAction("Automated", self)
        self.automated_tool_action.setCheckable(True)
        self.automated_tool_action.setChecked(True)
        self.automated_tool_action.triggered.connect(
            lambda: self.set_tool_mode("automated")
        )
        self.tool_action_group.addAction(self.random_forest_tool_action)
        self.tool_action_group.addAction(self.threshold_tool_action)
        self.tool_action_group.addAction(self.automated_tool_action)
        phenotyping_menu.addAction(self.random_forest_tool_action)
        phenotyping_menu.addAction(self.threshold_tool_action)
        phenotyping_menu.addAction(self.automated_tool_action)
        layout.setMenuBar(self.menu_bar)

        viewer_layout = QHBoxLayout()
        viewer_layout.addWidget(self.image_label, stretch=1)
        viewer_layout.addWidget(self.right_panel_stack)
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
        self.update_threshold_controls()
        self.update_automated_controls()
        self.refresh_cellpose_marker_list()
        self.set_tool_mode("automated")
        self.start_cuda_detection()
        self.update_segmentation_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            self.display_pixmap()

    def display_pixmap(self):
        # Napari renders directly into the embedded canvas.
        return

    def sync_phenotype_name_from_random_forest(self, text):
        for widget in (
            self.threshold_phenotype_name,
            self.automated_phenotype_name,
        ):
            if widget.text() == text:
                continue
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)

    def sync_phenotype_name_from_threshold(self, text):
        for widget in (self.phenotype_name, self.automated_phenotype_name):
            if widget.text() == text:
                continue
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)

    def sync_phenotype_name_from_automated(self, text):
        for widget in (self.phenotype_name, self.threshold_phenotype_name):
            if widget.text() == text:
                continue
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)

    def sync_automated_annotation_visibility(self):
        if not hasattr(self, "automated_positive_checkbox"):
            return
        for source, target in (
            (self.positive_annotations_checkbox, self.automated_positive_checkbox),
            (self.negative_annotations_checkbox, self.automated_negative_checkbox),
        ):
            target.blockSignals(True)
            target.setChecked(source.isChecked())
            target.blockSignals(False)

    def automated_annotation_visibility_changed(self):
        for source, target in (
            (self.automated_positive_checkbox, self.positive_annotations_checkbox),
            (self.automated_negative_checkbox, self.negative_annotations_checkbox),
        ):
            target.blockSignals(True)
            target.setChecked(source.isChecked())
            target.blockSignals(False)
        self.update_display()

    def sync_automated_model_visibility(self):
        if not hasattr(self, "automated_modelled_checkbox"):
            return
        checked = self.modelled_phenotypes_checkbox.isChecked()
        for widget in (
            self.automated_modelled_checkbox,
            self.automated_edit_modelled_checkbox,
        ):
            widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(False)

    def automated_model_visibility_changed(self):
        source = self.sender()
        checked = bool(source.isChecked())
        self.modelled_phenotypes_checkbox.blockSignals(True)
        self.modelled_phenotypes_checkbox.setChecked(checked)
        self.modelled_phenotypes_checkbox.blockSignals(False)
        for widget in (
            self.automated_modelled_checkbox,
            self.automated_edit_modelled_checkbox,
        ):
            if widget is source:
                continue
            widget.blockSignals(True)
            widget.setChecked(checked)
            widget.blockSignals(False)
        self.update_display()

    def sync_automated_threshold_controls_from_master(self):
        pairs = (
            (self.automated_intensity_slider, self.threshold_intensity_slider),
            (self.automated_percent_slider, self.threshold_percent_slider),
            (self.automated_buffer_slider, self.threshold_buffer_slider),
        )
        for automated_widget, master_widget in pairs:
            automated_widget.blockSignals(True)
            automated_widget.setValue(master_widget.value())
            automated_widget.blockSignals(False)
        for automated_widget, master_widget in (
            (self.automated_nucleus_checkbox, self.threshold_nucleus_checkbox),
            (self.automated_cytoplasm_checkbox, self.threshold_cytoplasm_checkbox),
        ):
            automated_widget.blockSignals(True)
            automated_widget.setChecked(master_widget.isChecked())
            automated_widget.blockSignals(False)
        self.automated_threshold_mask_button.blockSignals(True)
        self.automated_threshold_mask_button.setChecked(
            self.threshold_mask_button.isChecked()
        )
        self.automated_threshold_mask_button.setText(
            self.threshold_mask_button.text()
        )
        self.automated_threshold_mask_button.blockSignals(False)
        self.automated_percent_label.setText(
            f"Positive pixels required: >{self.automated_percent_slider.value()}%"
        )
        self.automated_buffer_label.setText(
            buffer_distance_label(self.automated_buffer_slider.value())
        )
        if self.current_fov is not None:
            self._update_threshold_value(invalidate=False)
            self.automated_intensity_label.setText(
                self.threshold_intensity_label.text()
            )

    def automated_intensity_changed(self, value):
        self.threshold_intensity_slider.blockSignals(True)
        self.threshold_intensity_slider.setValue(value)
        self.threshold_intensity_slider.blockSignals(False)
        if self.current_fov is not None:
            self._update_threshold_value(invalidate=True)
            self.automated_intensity_label.setText(
                self.threshold_intensity_label.text()
            )
        else:
            self.automated_intensity_label.setText(
                f"Intensity threshold: {value / 10:.1f}% of display range"
            )
        self.automated_edit_status_label.setText(
            "Threshold settings changed. Click Re-Phenotype to apply."
        )
        self.update_display()

    def automated_percent_changed(self, value):
        self.threshold_percent_slider.blockSignals(True)
        self.threshold_percent_slider.setValue(value)
        self.threshold_percent_slider.blockSignals(False)
        self.threshold_percent_label.setText(
            f"Positive pixels required: >{value}%"
        )
        self.automated_percent_label.setText(
            f"Positive pixels required: >{value}%"
        )
        self._invalidate_threshold_predictions()
        self.automated_edit_status_label.setText(
            "Threshold settings changed. Click Re-Phenotype to apply."
        )
        self.update_display()

    def automated_compartment_changed(self):
        for automated_widget, master_widget in (
            (self.automated_nucleus_checkbox, self.threshold_nucleus_checkbox),
            (self.automated_cytoplasm_checkbox, self.threshold_cytoplasm_checkbox),
        ):
            master_widget.blockSignals(True)
            master_widget.setChecked(automated_widget.isChecked())
            master_widget.blockSignals(False)
        self._invalidate_threshold_predictions()
        self.automated_edit_status_label.setText(
            "Threshold settings changed. Click Re-Phenotype to apply."
        )
        self.update_automated_controls()
        self.update_display()

    def threshold_mask_toggled(self, checked):
        text = f"Threshold Mask: {'On' if checked else 'Off'}"
        self.threshold_mask_button.setText(text)
        self.automated_threshold_mask_button.blockSignals(True)
        self.automated_threshold_mask_button.setChecked(bool(checked))
        self.automated_threshold_mask_button.setText(text)
        self.automated_threshold_mask_button.blockSignals(False)
        self.update_display()

    def automated_threshold_mask_toggled(self, checked):
        text = f"Threshold Mask: {'On' if checked else 'Off'}"
        self.automated_threshold_mask_button.setText(text)
        self.threshold_mask_button.blockSignals(True)
        self.threshold_mask_button.setChecked(bool(checked))
        self.threshold_mask_button.setText(text)
        self.threshold_mask_button.blockSignals(False)
        self.update_display()

    def automated_buffer_changed(self, value):
        self.threshold_buffer_slider.blockSignals(True)
        self.threshold_buffer_slider.setValue(value)
        self.threshold_buffer_slider.blockSignals(False)
        self.threshold_buffer_label.setText(
            buffer_distance_label(value)
        )
        self.automated_buffer_label.setText(
            buffer_distance_label(value)
        )
        self._invalidate_threshold_predictions()
        self.automated_edit_status_label.setText(
            "Threshold settings changed. Click Re-Phenotype to apply."
        )
        self.update_display()

    def open_automated_edit(self):
        self.automated_edit_mode = True
        self.automated_panel_stack.setCurrentIndex(1)
        self.sync_automated_threshold_controls_from_master()
        self.sync_automated_annotation_visibility()
        self.sync_automated_model_visibility()
        self.update_automated_controls()
        self.update_display()

    def start_cuda_detection(self):
        self.cuda_gpu_available = None
        self.segmenting_gpu_status_label.setText(
            "● Checking for a CUDA-compatible GPU..."
        )
        self.segmenting_gpu_status_label.setStyleSheet(
            "color: #b26a00; font-weight: bold;"
        )
        self.update_segmentation_controls()
        worker = CudaDetectionWorker()
        worker.signals.finished.connect(self.on_cuda_detection_finished)
        worker.signals.error.connect(self.on_cuda_detection_error)
        self.cuda_detection_worker = worker
        self.thread_pool.start(worker)

    def on_cuda_detection_finished(self, available):
        self.cuda_detection_worker = None
        self.cuda_gpu_available = bool(available)
        if self.cuda_gpu_available:
            self.segmenting_gpu_status_label.setText(
                "● CUDA-compatible GPU detected"
            )
            self.segmenting_gpu_status_label.setStyleSheet(
                "color: #238636; font-weight: bold;"
            )
            self.segmenting_status_label.setText(
                "Load images, select membrane markers, then click Segment."
            )
        else:
            self.segmenting_gpu_status_label.setText(
                "● CUDA-compatible GPU not detected"
            )
            self.segmenting_gpu_status_label.setStyleSheet(
                "color: #c62828; font-weight: bold;"
            )
            self.segmenting_status_label.setText(
                "Segmentation options are disabled because Cellpose-SAM "
                "requires a CUDA-compatible GPU."
            )
        self.update_segmentation_controls()

    def on_cuda_detection_error(self, _error_message):
        self.on_cuda_detection_finished(False)

    def _shared_membrane_markers(self):
        """Return membrane-marker names available in every loaded image."""
        if not self.loaded_images:
            return []
        first_names = membrane_marker_names(
            self.loaded_images[0]["img"].get_channel_names()
        )
        shared = set(first_names)
        for state in self.loaded_images[1:]:
            shared.intersection_update(
                membrane_marker_names(state["img"].get_channel_names())
            )
        return [name for name in first_names if name in shared]

    def selected_cellpose_markers(self):
        return [
            checkbox.text()
            for checkbox in self.segmenting_marker_checkboxes
            if checkbox.isChecked()
        ]

    def refresh_cellpose_marker_list(self):
        """Rebuild the scrollable list from channels shared by the project."""
        if not hasattr(self, "segmenting_marker_layout"):
            return
        selected = set(self.segmenting_selected_markers)
        selected.update(self.selected_cellpose_markers())
        while self.segmenting_marker_layout.count():
            item = self.segmenting_marker_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.segmenting_marker_checkboxes = []

        shared_markers = self._shared_membrane_markers()
        self.segmenting_selected_markers = selected.intersection(shared_markers)
        if shared_markers:
            for marker_name in shared_markers:
                checkbox = QCheckBox(marker_name)
                checkbox.setChecked(
                    marker_name in self.segmenting_selected_markers
                )
                checkbox.toggled.connect(
                    lambda checked, name=marker_name:
                    self.cellpose_marker_selection_changed(name, checked)
                )
                self.segmenting_marker_layout.addWidget(checkbox)
                self.segmenting_marker_checkboxes.append(checkbox)
            self.segmenting_marker_layout.addStretch()
        else:
            message = (
                "Load an image to list markers."
                if not self.loaded_images
                else "No non-DAPI marker is shared by every loaded image."
            )
            placeholder = QLabel(message)
            placeholder.setWordWrap(True)
            self.segmenting_marker_layout.addWidget(placeholder)
            self.segmenting_marker_layout.addStretch()

        nuclear_names = [
            dapi_channel_name(state["img"].get_channel_names())
            for state in self.loaded_images
        ]
        if not nuclear_names:
            dapi_message = (
                "DAPI will be supplied as the nuclear channel when available."
            )
        elif all(name is not None for name in nuclear_names):
            unique_names = list(dict.fromkeys(nuclear_names))
            if len(unique_names) == 1:
                dapi_message = f"Nuclear channel: {unique_names[0]}"
            else:
                dapi_message = "DAPI nuclear channels will be used by name."
        elif any(name is not None for name in nuclear_names):
            dapi_message = (
                "DAPI will be used where available; some images have no DAPI "
                "channel."
            )
        else:
            dapi_message = (
                "No DAPI channel was found; segmentation will use membrane "
                "guidance only."
            )
        self.segmenting_dapi_label.setText(dapi_message)
        self.update_segmentation_controls()

    def cellpose_marker_selection_changed(self, marker_name, checked):
        if checked:
            self.segmenting_selected_markers.add(marker_name)
        else:
            self.segmenting_selected_markers.discard(marker_name)
        self.update_segmentation_controls()

    def update_segmentation_controls(self):
        if not hasattr(self, "segment_button"):
            return
        gpu_ready = self.cuda_gpu_available is True
        ready = (
            gpu_ready
            and bool(self.loaded_images)
            and bool(self.selected_cellpose_markers())
            and not self.is_loading
            and self.cellpose_worker is None
        )
        self.segment_button.setEnabled(ready)
        self.segmenting_marker_scroll.setEnabled(
            gpu_ready and not self.is_loading
        )
        for checkbox in self.segmenting_marker_checkboxes:
            checkbox.setEnabled(gpu_ready and not self.is_loading)

    def _release_generated_mask_handles(self):
        """Release generated memmaps so Windows can atomically replace them."""
        self.released_generated_segmentations = {}
        for image_index, state in enumerate(self.loaded_images):
            _cell_output, mask_output = output_paths_for_image(
                state["image_path"]
            )
            current_path = state.get("segmentation_mask_path")
            if current_path is None:
                continue
            if Path(current_path).resolve() != mask_output.resolve():
                continue
            masks = state.get("segmentation_masks")
            if not isinstance(masks, np.memmap):
                continue
            self.released_generated_segmentations[image_index] = (
                state.get("cell_data_path"),
                current_path,
            )
            try:
                masks._mmap.close()
            except Exception:
                pass
            state["segmentation_masks"] = None
            if image_index == self.current_image_index:
                self.segmentation_masks = None

    def _restore_released_segmentations(self):
        for image_index, paths in self.released_generated_segmentations.items():
            cell_path, mask_path = paths
            if not cell_path or not mask_path:
                continue
            try:
                cell_data, masks, cell_path, mask_path = self._read_segmentation(
                    cell_path,
                    mask_path,
                    self.loaded_images[image_index]["img"],
                )
            except Exception:
                continue
            self.loaded_images[image_index].update({
                "cell_data": cell_data,
                "segmentation_masks": masks,
                "cell_data_path": cell_path,
                "segmentation_mask_path": mask_path,
            })
        self.released_generated_segmentations = {}
        if 0 <= self.current_image_index < len(self.loaded_images):
            state = self.loaded_images[self.current_image_index]
            self.cell_data = state["cell_data"]
            self.segmentation_masks = state["segmentation_masks"]

    def start_cellpose_segmentation(self):
        marker_names = self.selected_cellpose_markers()
        if self.cuda_gpu_available is not True:
            QMessageBox.warning(
                self,
                "CellPoseSAM segmentation",
                "A CUDA-compatible GPU is required for segmentation.",
            )
            return
        if not self.loaded_images:
            QMessageBox.warning(
                self,
                "CellPoseSAM segmentation",
                "Load at least one image before segmenting.",
            )
            return
        if not marker_names:
            QMessageBox.warning(
                self,
                "CellPoseSAM segmentation",
                "Select at least one membrane-guiding marker.",
            )
            return

        self._capture_current_image_state()
        self._release_generated_mask_handles()
        self.set_loading(
            True,
            f"Loading Cellpose-SAM model {CELLPOSE_SAM_MODEL}...",
        )
        self.segmenting_status_label.setText(
            "Segmentation is running. The model may be downloaded on first use."
        )
        worker = CellposeSegmentationWorker(
            images=[state["img"] for state in self.loaded_images],
            marker_names=marker_names,
            pixel_size_um=DEFAULT_PIXEL_SIZE_UM,
        )
        worker.signals.progress.connect(self.on_cellpose_segmentation_progress)
        worker.signals.finished.connect(self.on_cellpose_segmentation_finished)
        worker.signals.error.connect(self.on_cellpose_segmentation_error)
        self.cellpose_worker = worker
        self.thread_pool.start(worker)

    def on_cellpose_segmentation_progress(self, message):
        self.status_label.setText(message)
        self.segmenting_status_label.setText(message)

    def on_cellpose_segmentation_finished(self, results):
        try:
            result_by_image = {
                str(Path(result["image_path"]).resolve()): result
                for result in results
            }
            if len(result_by_image) != len(self.loaded_images):
                raise ValueError(
                    "Cellpose results did not match the loaded image count."
                )

            total_cells = 0
            for state in self.loaded_images:
                image_key = str(Path(state["image_path"]).resolve())
                if image_key not in result_by_image:
                    raise ValueError(
                        f"No Cellpose result was returned for {image_key}."
                    )
                result = result_by_image[image_key]
                cell_data, masks, cell_path, mask_path = self._read_segmentation(
                    result["cell_data_path"],
                    result["segmentation_mask_path"],
                    state["img"],
                )
                state.update({
                    "cell_data_path": cell_path,
                    "segmentation_mask_path": mask_path,
                    "cell_data": cell_data,
                    "segmentation_masks": masks,
                    "annotations": {},
                    "centroid_cache": None,
                    "mask_label_row_cache": None,
                    "model_predictions": None,
                    "threshold_predictions": None,
                    "automated_exclusions": set(),
                })
                total_cells += len(cell_data)

            self.released_generated_segmentations = {}
            self.model_bundle = None
            self.training_navigation_indices = {
                "positive": -1,
                "negative": -1,
            }
            self.threshold_intensity_value = None
            self.threshold_channel_name = None
            self.threshold_histogram_request_id += 1
            self.model_status_label.setText(
                "No model trained or loaded after re-segmentation."
            )
            current_index = self.current_image_index
            self._activate_image(current_index)
            self.segmentation_checkbox.setChecked(True)
            self.update_annotation_counts()
            self.update_model_prediction_counts()
            self.update_threshold_prediction_counts()
            message = (
                f"Cellpose-SAM replaced segmentation for "
                f"{len(results)} image(s), producing {total_cells:,} cells."
            )
            self.segmenting_status_label.setText(message)
            self.cellpose_worker = None
            self.set_loading(False, message)
            self.update_display()
        except Exception:
            self.on_cellpose_segmentation_error(traceback.format_exc())

    def on_cellpose_segmentation_error(self, error_message):
        self._restore_released_segmentations()
        self.cellpose_worker = None
        self.segmenting_status_label.setText("Cellpose-SAM segmentation failed.")
        self.set_loading(False, "Cellpose-SAM segmentation failed.")
        self.update_display()
        QMessageBox.warning(
            self,
            "Could not segment images",
            error_message,
        )

    def set_tool_mode(self, tool):
        if tool not in {
            "cellpose_sam", "random_forest", "threshold", "automated"
        }:
            tool = "random_forest"
        self.active_tool = tool
        segmenting_mode = tool == "cellpose_sam"
        threshold_mode = tool == "threshold"
        automated_mode = tool == "automated"
        page_index = {
            "cellpose_sam": 0,
            "random_forest": 1,
            "threshold": 2,
            "automated": 3,
        }[tool]
        self.right_panel_stack.setCurrentIndex(page_index)
        self.cellpose_sam_tool_action.setChecked(segmenting_mode)
        self.random_forest_tool_action.setChecked(tool == "random_forest")
        self.threshold_tool_action.setChecked(threshold_mode)
        self.automated_tool_action.setChecked(automated_mode)
        if automated_mode:
            self.automated_edit_mode = False
            self.automated_panel_stack.setCurrentIndex(0)

        if threshold_mode and self.current_fov is not None:
            displayed_channel = self.channel_dropdown.currentText()
            if (
                self.threshold_intensity_value is None
                or self.threshold_channel_name != displayed_channel
            ):
                self._update_threshold_value(invalidate=False)
        if threshold_mode:
            self.request_threshold_histogram_refresh()
        else:
            self.threshold_histogram_request_id += 1
            self.threshold_histogram_timer.stop()
        self.update_model_controls()
        self.update_threshold_controls()
        self.update_automated_controls()
        self.update_segmentation_controls()
        self.update_display()

    def _invalidate_threshold_predictions(self):
        for state in self.loaded_images:
            state["threshold_predictions"] = None
        self.update_threshold_prediction_counts()
        self.update_threshold_controls()

    def _update_threshold_histogram_markers(self):
        self.threshold_intensity_histogram.set_marker(
            self.threshold_intensity_value
        )
        self.threshold_fraction_histogram.set_marker(
            self.threshold_percent_slider.value()
        )

    def request_threshold_histogram_refresh(self, delay_ms=0):
        """Queue whole-image histogram statistics without blocking the UI."""
        if not hasattr(self, "threshold_intensity_histogram"):
            return
        self.threshold_histogram_request_id += 1
        self.threshold_histogram_timer.stop()
        self._update_threshold_histogram_markers()

        if self.active_tool != "threshold":
            return
        if not (0 <= self.current_image_index < len(self.loaded_images)):
            self.threshold_intensity_histogram.set_message("Load an image")
            self.threshold_fraction_histogram.set_message("Load an image")
            return
        if self.current_fov is None:
            self.threshold_intensity_histogram.set_message("Generate an FOV")
            self.threshold_fraction_histogram.set_message("Generate an FOV")
            return
        if self.cell_data is None or self.segmentation_masks is None:
            self.threshold_intensity_histogram.set_message(
                "Load cell data and a segmentation mask"
            )
            self.threshold_fraction_histogram.set_message(
                "Load cell data and a segmentation mask"
            )
            return
        if self.threshold_compartment() is None:
            self.threshold_intensity_histogram.set_message(
                "Select at least one compartment"
            )
            self.threshold_fraction_histogram.set_message(
                "Select at least one compartment"
            )
            return

        self.threshold_intensity_histogram.set_status(
            "Updating all image cells…"
        )
        self.threshold_fraction_histogram.set_status(
            "Updating all image cells…"
        )
        if self.threshold_histogram_worker is not None:
            return
        if delay_ms > 0:
            self.threshold_histogram_timer.start(delay_ms)
        else:
            self._start_threshold_histogram_worker()

    def _start_threshold_histogram_worker(self):
        if self.threshold_histogram_worker is not None:
            return
        if (
            self.active_tool != "threshold"
            or self.current_fov is None
            or self.cell_data is None
            or self.segmentation_masks is None
            or not (0 <= self.current_image_index < len(self.loaded_images))
        ):
            return
        compartment = self.threshold_compartment()
        if compartment is None:
            return
        if self.threshold_intensity_value is None:
            self._update_threshold_value(invalidate=False)

        state = self.loaded_images[self.current_image_index]
        try:
            self._cell_centroid_cache(state)
        except Exception as error:
            self.threshold_intensity_histogram.set_message(
                "Could not map cell centroids"
            )
            self.threshold_fraction_histogram.set_message(
                "Could not map cell centroids"
            )
            self.status_label.setText(str(error))
            return

        worker = ThresholdHistogramWorker(
            request_id=self.threshold_histogram_request_id,
            image_index=self.current_image_index,
            image_state=state,
            channel_name=self.channel_dropdown.currentText(),
            intensity_threshold=self.threshold_intensity_value,
            compartment=compartment,
            inward_buffer_pixels=buffer_pixels_from_slider(
                self.threshold_buffer_slider.value()
            ),
        )
        worker.signals.finished.connect(self.on_threshold_histograms_ready)
        worker.signals.error.connect(self.on_threshold_histograms_error)
        self.threshold_histogram_worker = worker
        self.thread_pool.start(worker)

    def on_threshold_histograms_ready(self, result):
        self.threshold_histogram_worker = None
        if result["request_id"] != self.threshold_histogram_request_id:
            self._start_threshold_histogram_worker()
            return
        if result["image_index"] != self.current_image_index:
            return

        denominator_pixels = np.asarray(result["denominator_pixels"])
        included = denominator_pixels > 0
        mean_intensities = np.asarray(result["mean_intensity"])[included]
        positive_percentages = (
            np.asarray(result["positive_fraction"])[included] * 100.0
        )
        self.threshold_intensity_histogram.set_data(
            mean_intensities,
            marker=self.threshold_intensity_value,
        )
        self.threshold_fraction_histogram.set_data(
            positive_percentages,
            marker=self.threshold_percent_slider.value(),
            value_range=(0.0, 100.0),
        )
        channel_name = result["channel_name"]
        self.threshold_intensity_histogram_label.setText(
            f"All-image mean {channel_name} fluorescence per cell"
        )
        self.threshold_fraction_histogram_label.setText(
            "All-image positive-pixel percentages per cell"
        )

    def on_threshold_histograms_error(self, payload):
        self.threshold_histogram_worker = None
        if payload["request_id"] != self.threshold_histogram_request_id:
            self._start_threshold_histogram_worker()
            return
        self.threshold_intensity_histogram.set_message(
            "Could not calculate histogram"
        )
        self.threshold_fraction_histogram.set_message(
            "Could not calculate histogram"
        )
        self.status_label.setText(payload["error"])

    def _update_threshold_value(self, invalidate=True):
        if self.current_fov is None:
            self.threshold_intensity_value = None
            self.threshold_channel_name = None
            self.threshold_intensity_label.setText("Intensity threshold: —")
            return
        self.threshold_intensity_value = intensity_threshold_from_slider(
            self.current_fov,
            self.threshold_intensity_slider.value(),
            self.threshold_intensity_slider.maximum(),
        )
        self.threshold_channel_name = self.channel_dropdown.currentText()
        normalized_percent = (
            100
            * self.threshold_intensity_slider.value()
            / self.threshold_intensity_slider.maximum()
        )
        self.threshold_intensity_label.setText(
            f"Intensity threshold: {self.threshold_intensity_value:.6g} "
            f"({normalized_percent:.1f}% of display range)"
        )
        if invalidate:
            self._invalidate_threshold_predictions()

    def threshold_slider_changed(self):
        if self.current_fov is not None:
            self._update_threshold_value(invalidate=True)
        self._update_threshold_histogram_markers()
        self.request_threshold_histogram_refresh(delay_ms=300)
        self.update_display()

    def threshold_percent_changed(self, value):
        self.threshold_percent_label.setText(
            f"Positive pixels required: >{value}%"
        )
        self._invalidate_threshold_predictions()
        self._update_threshold_histogram_markers()
        self.update_display()

    def threshold_compartment(self):
        use_nucleus = self.threshold_nucleus_checkbox.isChecked()
        use_cytoplasm = self.threshold_cytoplasm_checkbox.isChecked()
        if use_nucleus and use_cytoplasm:
            return "all"
        if use_nucleus:
            return "nucleus"
        if use_cytoplasm:
            return "cytoplasm_membrane"
        return None

    def threshold_compartment_changed(self):
        self._invalidate_threshold_predictions()
        self.update_threshold_controls()
        self.request_threshold_histogram_refresh()
        self.update_display()

    def threshold_buffer_changed(self, value):
        self.threshold_buffer_label.setText(
            buffer_distance_label(value)
        )
        self.automated_buffer_slider.blockSignals(True)
        self.automated_buffer_slider.setValue(value)
        self.automated_buffer_slider.blockSignals(False)
        self.automated_buffer_label.setText(buffer_distance_label(value))
        self._invalidate_threshold_predictions()
        self.request_threshold_histogram_refresh(delay_ms=300)
        self.update_display()

    def on_channel_changed(self):
        threshold_edit = (
            self.active_tool == "automated" and self.automated_edit_mode
        )
        if self.active_tool == "threshold" or threshold_edit:
            self.threshold_intensity_value = None
            self.threshold_channel_name = None
            self._invalidate_threshold_predictions()
            if self.active_tool == "threshold":
                self.threshold_histogram_request_id += 1
                self.threshold_histogram_timer.stop()
                self.threshold_intensity_histogram.set_message("Loading channel…")
                self.threshold_fraction_histogram.set_message("Loading channel…")
        self.reload_current_fov()

    def current_threshold_highlight(self):
        threshold_visible = (
            self.active_tool == "threshold"
            and self.threshold_mask_button.isChecked()
        ) or (
            self.active_tool == "automated"
            and self.automated_edit_mode
            and self.automated_threshold_mask_button.isChecked()
        )
        if not threshold_visible or self.current_fov is None:
            return None
        if (
            self.threshold_intensity_value is None
            or self.threshold_channel_name != self.channel_dropdown.currentText()
        ):
            self._update_threshold_value(
                invalidate=self.threshold_channel_name is not None
            )
        return np.asarray(self.current_fov) > self.threshold_intensity_value

    def set_loading(self, loading: bool, message: str = ""):
        if loading:
            self.cancel_cell_probability_hover()
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
        self.update_threshold_controls()
        self.update_automated_controls()
        self.update_segmentation_controls()

    def open_qptiff(self):
        source_dialog = QMessageBox(self)
        source_dialog.setWindowTitle("Add image")
        source_dialog.setText("Select the image format to add:")
        tiff_button = source_dialog.addButton(
            "TIFF / QPTIFF", QMessageBox.AcceptRole
        )
        zarr_button = source_dialog.addButton(
            "OME-Zarr directory", QMessageBox.ActionRole
        )
        source_dialog.addButton(QMessageBox.Cancel)
        source_dialog.exec()

        selected = source_dialog.clickedButton()
        if selected is tiff_button:
            image_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select TIFF image",
                "",
                "TIFF images (*.qptiff *.tif *.tiff);;All files (*)",
            )
        elif selected is zarr_button:
            image_path = QFileDialog.getExistingDirectory(
                self,
                "Select OME-Zarr directory",
                "",
                QFileDialog.ShowDirsOnly,
            )
        else:
            return
        if not image_path:
            return

        self.set_loading(True, "Loading image...")
        try:
            state = self._create_image_state(image_path)
            self._capture_current_image_state()
            self.loaded_images.append(state)
            self._refresh_image_carousel()
            self._activate_image(len(self.loaded_images) - 1)
            self.refresh_cellpose_marker_list()
            self.status_label.setText(
                f"Loaded {Path(state['image_path']).name} "
                f"as {state['img'].format_name} "
                f"({len(self.loaded_images)} image(s) in the carousel). Use "
                "Load Segmentation to import existing data, or choose "
                "Segmenting > CellPoseSAM to generate it."
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
            self.loaded_images[self.current_image_index][
                "threshold_predictions"
            ] = None
            self.loaded_images[self.current_image_index]["centroid_cache"] = None
            self.loaded_images[self.current_image_index][
                "mask_label_row_cache"
            ] = None
            self.loaded_images[self.current_image_index][
                "automated_exclusions"
            ] = set()
            self._capture_current_image_state()
            self.update_annotation_counts()
            self.segmentation_checkbox.setChecked(True)
            self.status_label.setText(
                f"Loaded {len(self.cell_data):,} cells and a "
                f"{self.segmentation_masks.shape[1]} x "
                f"{self.segmentation_masks.shape[0]} mask."
            )
            if self.active_tool == "threshold":
                self.request_threshold_histogram_refresh()
            self.update_display()
        except Exception:
            self.status_label.setText(traceback.format_exc())
        finally:
            self.set_loading(False, self.status_label.text())

    def _create_image_state(self, image_path, cell_path=None, mask_path=None):
        image_path = str(Path(image_path).expanduser().resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
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
            "mask_label_row_cache": None,
            "model_predictions": None,
            "threshold_predictions": None,
            "automated_exclusions": set(),
        }

    def _read_mask(self, mask_path, image=None, description="Segmentation mask"):
        mask_path = str(Path(mask_path).expanduser().resolve())
        if not Path(mask_path).is_file():
            raise FileNotFoundError(f"{description} not found: {mask_path}")

        try:
            # Keep large, uncompressed masks on disk when possible so adding
            # several images does not require loading every mask into RAM.
            masks = np.squeeze(tifffile.memmap(mask_path))
        except (ValueError, TypeError):
            masks = np.squeeze(tifffile.imread(mask_path))
        if masks.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional {description.lower()}; "
                f"got {masks.shape}."
            )
        if image is not None:
            _, image_height, image_width = image.get_shape()
            if masks.shape != (image_height, image_width):
                raise ValueError(
                    f"The {description.lower()} dimensions do not match the "
                    f"image ({masks.shape[1]} x {masks.shape[0]} mask; "
                    f"{image_width} x {image_height} image)."
                )
        return masks, mask_path

    def _read_segmentation(self, cell_path, mask_path, image=None):
        cell_path = str(Path(cell_path).expanduser().resolve())
        if not Path(cell_path).is_file():
            raise FileNotFoundError(f"Cell-data file not found: {cell_path}")

        separator = "," if cell_path.lower().endswith(".csv") else "\t"
        cell_data = pd.read_csv(cell_path, sep=separator)
        if cell_data.empty:
            raise ValueError("The selected cell-data file contains no rows.")

        masks, mask_path = self._read_mask(
            mask_path,
            image,
            "Segmentation mask",
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
        self.cancel_cell_probability_hover()
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
        if self.active_tool in {"threshold", "automated"}:
            self.threshold_intensity_value = None
            self.threshold_channel_name = None
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
        self.update_threshold_prediction_counts()

        if self.current_fov is None:
            self.image_label.clear()
            self.image_label.setText(
                f"{Path(self.image_path).name} loaded.\n\nClick 'Generate FOV'."
            )
        else:
            self.update_display()
        self.update_image_carousel_controls()
        self.update_model_controls()
        self.update_threshold_controls()
        self.update_automated_controls()
        self.update_segmentation_controls()
        if self.active_tool in {"threshold", "automated"}:
            if self.current_fov is not None:
                self._update_threshold_value(invalidate=False)
        if self.active_tool == "threshold":
            self.request_threshold_histogram_refresh()

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

    def _mask_label_row_cache(self, state):
        cached = state.get("mask_label_row_cache")
        if cached is not None:
            return cached
        cache = {}
        masks = state.get("segmentation_masks")
        if masks is not None and state.get("cell_data") is not None:
            centroids = self._cell_centroid_cache(state)
            finite = (
                np.isfinite(centroids["x"])
                & np.isfinite(centroids["y"])
            )
            x = np.zeros(len(centroids["x"]), dtype=np.int64)
            y = np.zeros(len(centroids["y"]), dtype=np.int64)
            x[finite] = np.rint(centroids["x"][finite]).astype(np.int64)
            y[finite] = np.rint(centroids["y"][finite]).astype(np.int64)
            valid = (
                finite
                & (x >= 0)
                & (x < masks.shape[1])
                & (y >= 0)
                & (y < masks.shape[0])
            )
            valid_rows = np.flatnonzero(valid)
            labels = masks[y[valid], x[valid]]
            for row_index, raw_label in zip(valid_rows, labels):
                label = int(raw_label)
                if label > 0 and label not in cache:
                    cache[label] = int(row_index)
        state["mask_label_row_cache"] = cache
        return cache

    def _model_prediction_at_fraction(self, x_fraction, y_fraction):
        if (
            self.current_fov is None
            or self.segmentation_masks is None
            or self.current_x0 is None
            or self.current_y0 is None
            or not (0 <= self.current_image_index < len(self.loaded_images))
        ):
            return None
        state = self.loaded_images[self.current_image_index]
        predictions = state.get("model_predictions")
        if predictions is None or "positive_probability" not in predictions:
            return None

        height, width = self.current_fov.shape[:2]
        local_x = min(max(int(x_fraction * width), 0), width - 1)
        local_y = min(max(int(y_fraction * height), 0), height - 1)
        global_x = int(self.current_x0) + local_x
        global_y = int(self.current_y0) + local_y
        if not (
            0 <= global_y < self.segmentation_masks.shape[0]
            and 0 <= global_x < self.segmentation_masks.shape[1]
        ):
            return None

        raw_cell_id = self.segmentation_masks[global_y, global_x]
        if raw_cell_id == 0:
            return None
        cell_id = str(
            raw_cell_id.item()
            if hasattr(raw_cell_id, "item")
            else raw_cell_id
        )
        label_row_cache = self._mask_label_row_cache(state)
        row_index = label_row_cache.get(int(raw_cell_id))
        if row_index is None:
            try:
                row_index = self._find_cell_row_index(
                    state,
                    {
                        "cell_id": cell_id,
                        "centroid_x": float(global_x),
                        "centroid_y": float(global_y),
                    },
                )
            except (TypeError, ValueError):
                return None
            if row_index is not None:
                label_row_cache[int(raw_cell_id)] = int(row_index)
        if row_index is None:
            return None

        positive = np.asarray(predictions["positive"], dtype=bool)
        positive_probability = np.asarray(
            predictions["positive_probability"], dtype=float
        )
        if not (
            0 <= row_index < positive.size
            and row_index < positive_probability.size
            and np.isfinite(positive_probability[row_index])
        ):
            return None
        return {
            "image_index": self.current_image_index,
            "row_index": int(row_index),
            "cell_id": cell_id,
            "positive": bool(positive[row_index]),
            "positive_probability": float(positive_probability[row_index]),
        }

    def track_cell_probability_hover(
        self, x_fraction, y_fraction, global_position
    ):
        hovered_prediction = self._model_prediction_at_fraction(
            x_fraction, y_fraction
        )
        if hovered_prediction is None:
            self.cancel_cell_probability_hover()
            return

        prediction_key = (
            hovered_prediction["image_index"],
            hovered_prediction["row_index"],
        )
        if prediction_key != self.hovered_prediction_key:
            self.cancel_cell_probability_hover()
            self.hovered_prediction_key = prediction_key
            self.hovered_prediction = hovered_prediction
            self.hover_global_position = QPoint(global_position)
            self.cell_probability_hover_timer.start()
            return

        self.hover_global_position = QPoint(global_position)
        if self.hover_probability_visible:
            self.show_hovered_cell_probability()

    def show_hovered_cell_probability(self):
        if self.hovered_prediction is None or self.hover_global_position is None:
            return
        positive_probability = np.clip(
            self.hovered_prediction["positive_probability"], 0.0, 1.0
        )
        if self.hovered_prediction["positive"]:
            call = "Positive"
            call_probability = positive_probability
        else:
            call = "Negative"
            call_probability = 1.0 - positive_probability
        phenotype_name = (
            self.phenotype_name.text().strip()
            or (
                self.model_bundle.get("phenotype_name", "").strip()
                if self.model_bundle is not None
                else ""
            )
            or "Phenotype"
        )
        QToolTip.showText(
            self.hover_global_position + QPoint(0, -42),
            f"{phenotype_name}: {call} ({call_probability:.1%})",
            self.image_label,
        )
        self.hover_probability_visible = True

    def cancel_cell_probability_hover(self):
        if hasattr(self, "cell_probability_hover_timer"):
            self.cell_probability_hover_timer.stop()
        self.hovered_prediction_key = None
        self.hovered_prediction = None
        self.hover_global_position = None
        self.hover_probability_visible = False
        QToolTip.hideText()

    def label_clicked_cell(self, x_fraction, y_fraction):
        can_label = self.active_tool == "random_forest" or (
            self.active_tool == "automated" and self.automated_edit_mode
        )
        if (
            not can_label
            or self.current_fov is None
            or self.segmentation_masks is None
        ):
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
            if self.active_tool == "automated" and row_index is not None:
                state.setdefault("automated_exclusions", set()).add(
                    int(row_index)
                )
                self.automated_edit_status_label.setText(
                    "Training labels changed. Click Re-Phenotype to apply."
                )
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
            "source": "manual",
        }
        if row_index is not None:
            state.setdefault("automated_exclusions", set()).discard(
                int(row_index)
            )
        if self.active_tool == "automated":
            self.automated_edit_status_label.setText(
                "Training labels changed. Click Re-Phenotype to apply."
            )
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
        if hasattr(self, "automated_positive_count_label"):
            self.automated_positive_count_label.setText(
                f"Positive: {positive}"
            )
            self.automated_negative_count_label.setText(
                f"Negative: {negative}"
            )
        for label, count in (("positive", positive), ("negative", negative)):
            if self.training_navigation_indices[label] >= count:
                self.training_navigation_indices[label] = -1
        self.update_training_navigation_controls()
        self.update_model_controls()
        self.update_automated_controls()

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
        controls = [
            (
                "positive", self.previous_positive_button,
                self.next_positive_button, self.positive_position_label,
            ),
            (
                "negative", self.previous_negative_button,
                self.next_negative_button, self.negative_position_label,
            ),
        ]
        if hasattr(self, "automated_previous_positive_button"):
            controls.extend([
                (
                    "positive", self.automated_previous_positive_button,
                    self.automated_next_positive_button,
                    self.automated_positive_position_label,
                ),
                (
                    "negative", self.automated_previous_negative_button,
                    self.automated_next_negative_button,
                    self.automated_negative_position_label,
                ),
            ])
        for label, previous_button, next_button, position_label in controls:
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
        self.cancel_cell_probability_hover()
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
            training_data = pd.DataFrame(rows, columns=features)
            pipeline, features = fit_random_forest(
                training_data=training_data,
                targets=targets,
                feature_columns=features,
            )
            self.model_bundle = {
                "format": MODEL_FORMAT,
                "version": MODEL_VERSION,
                "phenotype_name": self.phenotype_name.text().strip(),
                "feature_columns": features,
                "algorithm": RANDOM_FOREST_ALGORITHM,
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
        self.cancel_cell_probability_hover()
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
                prediction, positive_probability = (
                    model_calls_and_positive_probabilities(
                        self.model_bundle["pipeline"], measurements
                    )
                )
                state["model_predictions"] = {
                    "x": centroids["x"],
                    "y": centroids["y"],
                    "positive": prediction,
                    "positive_probability": positive_probability,
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
        if hasattr(self, "automated_model_positive_count_label"):
            for positive_label, negative_label in (
                (
                    self.automated_model_positive_count_label,
                    self.automated_model_negative_count_label,
                ),
                (
                    self.automated_edit_model_positive_count_label,
                    self.automated_edit_model_negative_count_label,
                ),
            ):
                positive_label.setText(f"Model positive: {positive:,}")
                negative_label.setText(f"Model negative: {negative:,}")

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

    def _set_automated_threshold_defaults(self):
        for slider, value in (
            (self.threshold_intensity_slider, AUTOMATED_INTENSITY_SLIDER_VALUE),
            (self.threshold_percent_slider, AUTOMATED_POSITIVE_PIXEL_PERCENT),
            (
                self.threshold_buffer_slider,
                DEFAULT_INWARD_BUFFER_SLIDER_VALUE,
            ),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        for checkbox in (
            self.threshold_nucleus_checkbox,
            self.threshold_cytoplasm_checkbox,
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self.threshold_percent_label.setText(
            f"Positive pixels required: >{AUTOMATED_POSITIVE_PIXEL_PERCENT}%"
        )
        self.threshold_buffer_label.setText(
            buffer_distance_label(DEFAULT_INWARD_BUFFER_SLIDER_VALUE)
        )
        self.threshold_mask_button.blockSignals(True)
        self.threshold_mask_button.setChecked(True)
        self.threshold_mask_button.setText("Threshold Mask: On")
        self.threshold_mask_button.blockSignals(False)
        self.threshold_intensity_value = None
        self.threshold_channel_name = None
        if self.current_fov is not None:
            self._update_threshold_value(invalidate=True)
        self.sync_automated_threshold_controls_from_master()

    def _automated_manual_training(self):
        manual_training = []
        for image_index, state in enumerate(self.loaded_images):
            for annotation in state["annotations"].values():
                if annotation.get("source") == "automated":
                    continue
                row_index = self._find_cell_row_index(state, annotation)
                if row_index is None:
                    continue
                annotation["row_index"] = int(row_index)
                annotation["source"] = "manual"
                manual_training.append({
                    "image_index": image_index,
                    "row_index": int(row_index),
                    "label": annotation["label"],
                })
        return manual_training

    def start_automated_phenotyping(self, reset_annotations):
        if self.automated_worker is not None:
            return
        if self.current_fov is None:
            QMessageBox.warning(
                self,
                "Automated phenotyping",
                "Generate a field of view before running automated phenotyping.",
            )
            return
        if not self.loaded_images or any(
            state["cell_data"] is None or state["segmentation_masks"] is None
            for state in self.loaded_images
        ):
            QMessageBox.warning(
                self,
                "Automated phenotyping",
                "Every loaded image must have cell data and a segmentation mask.",
            )
            return
        if reset_annotations and any(
            state["annotations"] for state in self.loaded_images
        ):
            choice = QMessageBox.question(
                self,
                "Replace training labels?",
                "Auto Phenotype will replace the existing training labels with "
                "35 automatically selected positive and 35 negative cells. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        if reset_annotations:
            self._set_automated_threshold_defaults()
        else:
            self.sync_automated_threshold_controls_from_master()
        compartment = self.threshold_compartment()
        if compartment is None:
            QMessageBox.warning(
                self,
                "Automated phenotyping",
                "Select Nucleus, Cytoplasm/Membrane, or both compartments.",
            )
            return

        channel_name = self.channel_dropdown.currentText()
        if (
            self.threshold_intensity_value is None
            or self.threshold_channel_name != channel_name
        ):
            self._update_threshold_value(invalidate=True)
        try:
            features = self._shared_numeric_features()
            if not features:
                raise ValueError(
                    "No shared numeric measurement columns were found across "
                    "the loaded cell-data files."
                )
            for state in self.loaded_images:
                if channel_name not in state["img"].get_channel_names():
                    raise ValueError(
                        f"{Path(state['image_path']).name} does not contain the "
                        f"channel '{channel_name}'."
                    )
                self._cell_centroid_cache(state)
            manual_training = (
                [] if reset_annotations else self._automated_manual_training()
            )
            excluded_rows = (
                []
                if reset_annotations
                else [
                    (image_index, int(row_index))
                    for image_index, state in enumerate(self.loaded_images)
                    for row_index in state.get("automated_exclusions", set())
                ]
            )
        except Exception as error:
            QMessageBox.warning(
                self, "Could not start automated phenotyping", str(error)
            )
            return

        # A fixed seed makes automated cell selection reproducible whenever
        # the images, segmentation, measurements, and settings are unchanged.
        random_seed = AUTOMATED_RANDOM_SEED
        self.automated_reset_annotations = bool(reset_annotations)
        self.automated_status_label.setText(
            "Thresholding all cells and training the random forest…"
        )
        self.automated_edit_status_label.setText(
            "Thresholding all cells and training the random forest…"
        )
        self.set_loading(True, "Running automated phenotyping...")
        worker = AutomatedPhenotypeWorker(
            image_states=list(self.loaded_images),
            channel_name=channel_name,
            intensity_threshold=self.threshold_intensity_value,
            positive_pixel_fraction=self.threshold_percent_slider.value() / 100,
            compartment=compartment,
            inward_buffer_pixels=buffer_pixels_from_slider(
                self.threshold_buffer_slider.value()
            ),
            feature_columns=features,
            phenotype_name=self.phenotype_name.text().strip(),
            manual_training=manual_training,
            excluded_rows=excluded_rows,
            random_seed=random_seed,
        )
        worker.signals.finished.connect(self.on_automated_phenotyping_finished)
        worker.signals.error.connect(self.on_automated_phenotyping_error)
        self.automated_worker = worker
        self.thread_pool.start(worker)

    def on_automated_phenotyping_finished(self, result):
        try:
            if len(result["model_predictions"]) != len(self.loaded_images):
                raise ValueError(
                    "Automated results did not match the loaded image count."
                )
            for image_index, state in enumerate(self.loaded_images):
                if self.automated_reset_annotations:
                    state["annotations"] = {}
                    state["automated_exclusions"] = set()
                else:
                    state["annotations"] = {
                        cell_id: annotation
                        for cell_id, annotation in state["annotations"].items()
                        if annotation.get("source") != "automated"
                    }
                state["annotations"].update(
                    result["auto_annotations"][image_index]
                )
                state["threshold_predictions"] = result[
                    "threshold_results"
                ][image_index]
                state["model_predictions"] = result[
                    "model_predictions"
                ][image_index]

            self.model_bundle = result["model_bundle"]
            if 0 <= self.current_image_index < len(self.loaded_images):
                self.annotations = self.loaded_images[
                    self.current_image_index
                ]["annotations"]
            training_total = self.model_bundle["training_samples"]
            feature_total = len(self.model_bundle["feature_columns"])
            message = (
                f"Automated random forest trained on {training_total} cells "
                f"and {feature_total} features, then applied to all loaded images."
            )
            self.model_status_label.setText(message)
            self.automated_status_label.setText(message)
            self.automated_edit_status_label.setText(message)
            self.modelled_phenotypes_checkbox.setChecked(True)
            self.training_navigation_indices = {
                "positive": -1,
                "negative": -1,
            }
            self.update_annotation_counts()
            self.update_model_prediction_counts()
            self.update_threshold_prediction_counts()
            self.update_model_controls()
            self.update_threshold_controls()
            self.update_automated_controls()
            self.update_display()
            self.set_loading(False, message)
        except Exception as error:
            self.on_automated_phenotyping_error(str(error))
        finally:
            self.automated_worker = None

    def on_automated_phenotyping_error(self, error_message):
        self.automated_worker = None
        self.set_loading(False)
        self.automated_status_label.setText("Automated phenotyping failed.")
        self.automated_edit_status_label.setText(
            "Automated phenotyping failed."
        )
        QMessageBox.warning(
            self,
            "Could not auto phenotype",
            error_message,
        )

    def update_automated_controls(self):
        if not hasattr(self, "auto_phenotype_button"):
            return
        all_have_segmentation = bool(self.loaded_images) and all(
            state["cell_data"] is not None
            and state["segmentation_masks"] is not None
            for state in self.loaded_images
        )
        has_fov = self.current_fov is not None
        ready = all_have_segmentation and has_fov and not self.is_loading
        has_predictions = any(
            state.get("model_predictions") is not None
            for state in self.loaded_images
        )
        all_have_predictions = bool(self.loaded_images) and all(
            state.get("model_predictions") is not None
            for state in self.loaded_images
        )
        self.auto_phenotype_button.setEnabled(ready)
        self.automated_edit_button.setEnabled(
            not self.is_loading
            and any(state["annotations"] for state in self.loaded_images)
        )
        editable = (
            self.active_tool == "automated"
            and self.automated_edit_mode
            and not self.is_loading
        )
        compartment_selected = (
            self.automated_nucleus_checkbox.isChecked()
            or self.automated_cytoplasm_checkbox.isChecked()
        )
        for widget in (
            self.automated_intensity_slider,
            self.automated_percent_slider,
            self.automated_nucleus_checkbox,
            self.automated_cytoplasm_checkbox,
            self.automated_threshold_mask_button,
            self.automated_positive_checkbox,
            self.automated_negative_checkbox,
            self.automated_phenotype_name,
        ):
            widget.setEnabled(editable)
        self.automated_buffer_slider.setEnabled(
            editable
            and (
                self.automated_nucleus_checkbox.isChecked()
                != self.automated_cytoplasm_checkbox.isChecked()
            )
        )
        self.rephenotype_button.setEnabled(
            editable and ready and compartment_selected
        )
        for widget in (
            self.automated_modelled_checkbox,
            self.automated_edit_modelled_checkbox,
        ):
            widget.setEnabled(has_predictions and not self.is_loading)
        self.automated_export_button.setEnabled(
            self.model_bundle is not None
            and all_have_predictions
            and not self.is_loading
        )

    def apply_threshold_to_all_cells(self):
        if self.current_fov is None:
            QMessageBox.warning(
                self,
                "Apply threshold",
                "Generate a field of view before choosing an intensity threshold.",
            )
            return
        if not self.loaded_images or any(
            state["cell_data"] is None or state["segmentation_masks"] is None
            for state in self.loaded_images
        ):
            QMessageBox.warning(
                self,
                "Apply threshold",
                "Every loaded image must have cell data and a segmentation mask.",
            )
            return

        compartment = self.threshold_compartment()
        if compartment is None:
            QMessageBox.warning(
                self,
                "Apply threshold",
                "Select Nucleus, Cytoplasm/Membrane, or both compartments.",
            )
            return

        channel_name = self.channel_dropdown.currentText()
        if (
            self.threshold_intensity_value is None
            or self.threshold_channel_name != channel_name
        ):
            self._update_threshold_value(invalidate=True)

        try:
            for state in self.loaded_images:
                if channel_name not in state["img"].get_channel_names():
                    raise ValueError(
                        f"{Path(state['image_path']).name} does not contain the "
                        f"channel '{channel_name}'."
                    )
                self._cell_centroid_cache(state)
        except Exception as error:
            QMessageBox.warning(self, "Could not apply threshold", str(error))
            return

        positive_pixel_fraction = self.threshold_percent_slider.value() / 100
        inward_buffer_pixels = buffer_pixels_from_slider(
            self.threshold_buffer_slider.value()
        )
        self._invalidate_threshold_predictions()
        self.set_loading(
            True,
            f"Applying {channel_name} threshold to all loaded cells...",
        )
        worker = ThresholdApplyWorker(
            image_states=list(self.loaded_images),
            channel_name=channel_name,
            intensity_threshold=self.threshold_intensity_value,
            positive_pixel_fraction=positive_pixel_fraction,
            compartment=compartment,
            inward_buffer_pixels=inward_buffer_pixels,
        )
        worker.signals.finished.connect(self.on_threshold_applied)
        worker.signals.error.connect(self.on_threshold_apply_error)
        self.threshold_worker = worker
        self.thread_pool.start(worker)

    def on_threshold_applied(self, results):
        try:
            if len(results) != len(self.loaded_images):
                raise ValueError(
                    "Threshold results did not match the loaded image count."
                )
            for state, predictions in zip(self.loaded_images, results):
                if len(predictions["positive"]) != len(state["cell_data"]):
                    raise ValueError(
                        f"Threshold result count does not match "
                        f"{Path(state['image_path']).name}."
                    )
                state["threshold_predictions"] = predictions
            self.threshold_phenotypes_checkbox.setChecked(True)
            self.update_threshold_prediction_counts()
            self.update_threshold_controls()
            self.update_display()
            total_cells = sum(
                len(state["cell_data"]) for state in self.loaded_images
            )
            self.set_loading(
                False,
                f"Applied threshold to {total_cells:,} cells across "
                f"{len(self.loaded_images)} image(s).",
            )
        except Exception as error:
            self.on_threshold_apply_error(str(error))
        finally:
            self.threshold_worker = None

    def on_threshold_apply_error(self, error_message):
        self.threshold_worker = None
        self.set_loading(False)
        QMessageBox.warning(
            self,
            "Could not apply threshold",
            error_message,
        )

    def current_threshold_markers(self):
        if (
            self.active_tool != "threshold"
            or not self.threshold_phenotypes_checkbox.isChecked()
            or not (0 <= self.current_image_index < len(self.loaded_images))
            or self.current_fov is None
        ):
            return []
        predictions = self.loaded_images[self.current_image_index].get(
            "threshold_predictions"
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
                "source": "threshold",
            }
            for index in indices
        ]

    def update_threshold_prediction_counts(self):
        if not hasattr(self, "threshold_positive_count_label"):
            return
        positive = negative = 0
        for state in self.loaded_images:
            predictions = state.get("threshold_predictions")
            if predictions is None:
                continue
            positive_count = int(np.count_nonzero(predictions["positive"]))
            positive += positive_count
            negative += int(len(predictions["positive"]) - positive_count)
        self.threshold_positive_count_label.setText(
            f"Threshold positive: {positive:,}"
        )
        self.threshold_negative_count_label.setText(
            f"Threshold negative: {negative:,}"
        )

    def update_threshold_controls(self):
        if not hasattr(self, "threshold_intensity_slider"):
            return
        threshold_mode = self.active_tool == "threshold"
        has_fov = self.current_fov is not None
        all_have_segmentation = bool(self.loaded_images) and all(
            state["cell_data"] is not None
            and state["segmentation_masks"] is not None
            for state in self.loaded_images
        )
        has_predictions = any(
            state.get("threshold_predictions") is not None
            for state in self.loaded_images
        )
        all_have_predictions = bool(self.loaded_images) and all(
            state.get("threshold_predictions") is not None
            for state in self.loaded_images
        )
        editable = threshold_mode and not self.is_loading
        compartment = self.threshold_compartment()
        self.threshold_phenotype_name.setEnabled(editable)
        self.threshold_mask_button.setEnabled(editable and has_fov)
        self.threshold_intensity_slider.setEnabled(editable and has_fov)
        self.threshold_percent_slider.setEnabled(editable and has_fov)
        self.threshold_nucleus_checkbox.setEnabled(editable)
        self.threshold_cytoplasm_checkbox.setEnabled(editable)
        self.threshold_buffer_slider.setEnabled(
            editable and has_fov and compartment in {
                "nucleus", "cytoplasm_membrane"
            }
        )
        self.apply_threshold_button.setEnabled(
            editable
            and has_fov
            and all_have_segmentation
            and compartment is not None
        )
        self.threshold_phenotypes_checkbox.setEnabled(
            editable and has_predictions
        )
        self.export_threshold_phenotypes_button.setEnabled(
            editable and all_have_predictions
        )

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
        threshold_export = self.active_tool == "threshold"
        if threshold_export:
            prediction_key = "threshold_predictions"
            prediction_source = "Threshold"
            features = []
            phenotype_name = (
                self.threshold_phenotype_name.text().strip() or "Phenotype"
            )
            missing_message = (
                "Apply the threshold to all loaded images before exporting."
            )
        else:
            if self.model_bundle is None:
                QMessageBox.warning(
                    self,
                    "Export cell phenotypes",
                    "Train or import and apply a model before exporting.",
                )
                return
            prediction_key = "model_predictions"
            prediction_source = "Model"
            features = list(self.model_bundle["feature_columns"])
            phenotype_name = (
                self.phenotype_name.text().strip()
                or self.model_bundle.get("phenotype_name", "").strip()
                or "Phenotype"
            )
            missing_message = (
                "Apply the model to all loaded images before exporting."
            )

        if not self.loaded_images or any(
            state.get(prediction_key) is None
            for state in self.loaded_images
        ):
            QMessageBox.warning(
                self,
                "Export cell phenotypes",
                missing_message,
            )
            return

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
                predictions = state[prediction_key]
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
                label_sources = np.full(
                    len(cell_data), prediction_source, dtype=object
                )
                if not threshold_export:
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
        self.cancel_cell_probability_hover()
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
            ) or not hasattr(
                bundle.get("pipeline"), "predict_proba"
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
        self.cancel_cell_probability_hover()
        for state in self.loaded_images:
            try:
                state["img"].close()
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
        self.cellpose_worker = None
        self.segmenting_selected_markers = set()
        self.released_generated_segmentations = {}
        self.automated_edit_mode = False
        self.threshold_intensity_value = None
        self.threshold_channel_name = None
        self.threshold_histogram_request_id += 1
        self.threshold_histogram_timer.stop()
        self.project_path = None
        self.training_navigation_indices = {"positive": -1, "negative": -1}
        self.phenotype_name.clear()
        self.threshold_intensity_slider.blockSignals(True)
        self.threshold_intensity_slider.setValue(500)
        self.threshold_intensity_slider.blockSignals(False)
        self.threshold_percent_slider.blockSignals(True)
        self.threshold_percent_slider.setValue(25)
        self.threshold_percent_slider.blockSignals(False)
        self.threshold_nucleus_checkbox.blockSignals(True)
        self.threshold_nucleus_checkbox.setChecked(True)
        self.threshold_nucleus_checkbox.blockSignals(False)
        self.threshold_cytoplasm_checkbox.blockSignals(True)
        self.threshold_cytoplasm_checkbox.setChecked(True)
        self.threshold_cytoplasm_checkbox.blockSignals(False)
        self.threshold_buffer_slider.blockSignals(True)
        self.threshold_buffer_slider.setValue(
            DEFAULT_INWARD_BUFFER_SLIDER_VALUE
        )
        self.threshold_buffer_slider.blockSignals(False)
        self.threshold_percent_label.setText("Positive pixels required: >25%")
        self.threshold_buffer_label.setText(
            buffer_distance_label(DEFAULT_INWARD_BUFFER_SLIDER_VALUE)
        )
        self.threshold_mask_button.blockSignals(True)
        self.threshold_mask_button.setChecked(True)
        self.threshold_mask_button.setText("Threshold Mask: On")
        self.threshold_mask_button.blockSignals(False)
        self.threshold_intensity_label.setText("Intensity threshold: —")
        self.threshold_intensity_histogram_label.setText(
            "All-image mean fluorescence per cell"
        )
        self.threshold_fraction_histogram_label.setText(
            "All-image positive-pixel percentages per cell"
        )
        self.threshold_intensity_histogram.set_message("Load an image")
        self.threshold_fraction_histogram.set_message("Load an image")
        self.channel_dropdown.clear()
        self.image_label.clear()
        self.image_label.setText("Select a TIFF or OME-Zarr image")
        self.model_status_label.setText("No model trained or loaded.")
        self.automated_status_label.setText(
            "Ready for automated phenotyping."
        )
        self.automated_edit_status_label.setText(
            "Edit labels or thresholds."
        )
        self.automated_panel_stack.setCurrentIndex(0)
        for slider, value in (
            (self.automated_intensity_slider, AUTOMATED_INTENSITY_SLIDER_VALUE),
            (self.automated_percent_slider, AUTOMATED_POSITIVE_PIXEL_PERCENT),
            (
                self.automated_buffer_slider,
                DEFAULT_INWARD_BUFFER_SLIDER_VALUE,
            ),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        for checkbox in (
            self.automated_nucleus_checkbox,
            self.automated_cytoplasm_checkbox,
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self.automated_threshold_mask_button.blockSignals(True)
        self.automated_threshold_mask_button.setChecked(True)
        self.automated_threshold_mask_button.setText("Threshold Mask: On")
        self.automated_threshold_mask_button.blockSignals(False)
        self._refresh_image_carousel()
        self.refresh_cellpose_marker_list()
        if self.cuda_gpu_available is False:
            self.segmenting_status_label.setText(
                "Segmentation options are disabled because Cellpose-SAM "
                "requires a CUDA-compatible GPU."
            )
        else:
            self.segmenting_status_label.setText(
                "Load images, select membrane markers, then click Segment."
            )
        self.update_annotation_counts()
        self.update_model_prediction_counts()
        self.update_threshold_prediction_counts()
        self.set_tool_mode("automated")
        self.update_model_controls()
        self.update_threshold_controls()
        self.update_automated_controls()
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
                "automated_exclusions": sorted(
                    int(row_index)
                    for row_index in state.get("automated_exclusions", set())
                ),
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
            "version": 4,
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
                "tool": self.active_tool,
                "segmenting": {
                    "tool": "cellpose_sam",
                    "markers": self.selected_cellpose_markers(),
                    "model": CELLPOSE_SAM_MODEL,
                },
                "threshold": {
                    "intensity_slider": self.threshold_intensity_slider.value(),
                    "positive_pixel_percent": (
                        self.threshold_percent_slider.value()
                    ),
                    "use_nucleus": (
                        self.threshold_nucleus_checkbox.isChecked()
                    ),
                    "use_cytoplasm_membrane": (
                        self.threshold_cytoplasm_checkbox.isChecked()
                    ),
                    "show_mask": self.threshold_mask_button.isChecked(),
                    "inward_buffer_microns": buffer_microns_from_slider(
                        self.threshold_buffer_slider.value()
                    ),
                    "inward_buffer_pixels": (
                        buffer_pixels_from_slider(
                            self.threshold_buffer_slider.value()
                        )
                    ),
                },
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
            if version not in {1, 2, 3, 4}:
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
                cell_count = (
                    0 if state["cell_data"] is None
                    else len(state["cell_data"])
                )
                state["automated_exclusions"] = {
                    int(row_index)
                    for row_index in entry.get("automated_exclusions", [])
                    if 0 <= int(row_index) < cell_count
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
                    old_state["img"].close()
                except Exception:
                    pass
            self.loaded_images = loaded_states
            self.current_image_index = -1
            segmenting_settings = viewer.get("segmenting", {})
            self.segmenting_selected_markers = {
                str(marker)
                for marker in segmenting_settings.get("markers", [])
            }
            self.refresh_cellpose_marker_list()
            self.model_bundle = None
            self.threshold_intensity_value = None
            self.threshold_channel_name = None
            self.model_status_label.setText("No model trained or loaded.")
            self.fov_size = int(viewer.get("fov_size", 512))
            self.phenotype_name.setText(phenotype.get("name", ""))
            threshold_settings = viewer.get("threshold", {})
            self.threshold_intensity_slider.blockSignals(True)
            self.threshold_intensity_slider.setValue(
                int(threshold_settings.get("intensity_slider", 500))
            )
            self.threshold_intensity_slider.blockSignals(False)
            self.threshold_percent_slider.blockSignals(True)
            self.threshold_percent_slider.setValue(
                int(threshold_settings.get("positive_pixel_percent", 25))
            )
            self.threshold_percent_slider.blockSignals(False)
            self.threshold_nucleus_checkbox.blockSignals(True)
            self.threshold_nucleus_checkbox.setChecked(
                bool(threshold_settings.get("use_nucleus", True))
            )
            self.threshold_nucleus_checkbox.blockSignals(False)
            self.threshold_cytoplasm_checkbox.blockSignals(True)
            self.threshold_cytoplasm_checkbox.setChecked(
                bool(
                    threshold_settings.get(
                        "use_cytoplasm_membrane", True
                    )
                )
            )
            self.threshold_cytoplasm_checkbox.blockSignals(False)
            show_threshold_mask = bool(
                threshold_settings.get("show_mask", True)
            )
            threshold_mask_text = (
                f"Threshold Mask: {'On' if show_threshold_mask else 'Off'}"
            )
            for threshold_mask_button in (
                self.threshold_mask_button,
                self.automated_threshold_mask_button,
            ):
                threshold_mask_button.blockSignals(True)
                threshold_mask_button.setChecked(show_threshold_mask)
                threshold_mask_button.setText(threshold_mask_text)
                threshold_mask_button.blockSignals(False)
            if "inward_buffer_microns" in threshold_settings:
                buffer_slider_value = int(round(
                    float(threshold_settings["inward_buffer_microns"])
                    * THRESHOLD_BUFFER_SLIDER_STEPS_PER_UM
                ))
            elif "inward_buffer_pixels" in threshold_settings:
                buffer_slider_value = buffer_slider_from_pixels(
                    threshold_settings["inward_buffer_pixels"]
                )
            else:
                buffer_slider_value = DEFAULT_INWARD_BUFFER_SLIDER_VALUE
            self.threshold_buffer_slider.blockSignals(True)
            self.threshold_buffer_slider.setValue(
                buffer_slider_value
            )
            self.threshold_buffer_slider.blockSignals(False)
            self.automated_buffer_slider.blockSignals(True)
            self.automated_buffer_slider.setValue(
                self.threshold_buffer_slider.value()
            )
            self.automated_buffer_slider.blockSignals(False)
            self.threshold_percent_label.setText(
                "Positive pixels required: "
                f">{self.threshold_percent_slider.value()}%"
            )
            self.threshold_buffer_label.setText(
                buffer_distance_label(self.threshold_buffer_slider.value())
            )
            self.automated_buffer_label.setText(
                buffer_distance_label(self.automated_buffer_slider.value())
            )
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
            self.set_tool_mode(viewer.get("tool", "automated"))
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
            self.fov_size,
            self.channel_dropdown.currentIndex(),
            self.img.get_dapi_channel_index(default=0),
        )
        worker.signals.finished.connect(self.on_fov_loaded)
        worker.signals.error.connect(self.on_fov_error)
        self.thread_pool.start(worker)

    def on_fov_loaded(self, marker_fov, dapi_fov):
        self.current_fov, self.current_dapi_fov = marker_fov, dapi_fov
        self.set_loading(False)
        self.regenerate_button.setEnabled(True)
        threshold_edit = (
            self.active_tool == "automated" and self.automated_edit_mode
        )
        if self.active_tool == "threshold" or threshold_edit:
            if self.threshold_intensity_value is None:
                self._update_threshold_value(invalidate=False)
            elif (
                self.threshold_channel_name
                != self.channel_dropdown.currentText()
            ):
                self._update_threshold_value(invalidate=True)
            if self.active_tool == "threshold":
                self.request_threshold_histogram_refresh()
            else:
                self.automated_intensity_label.setText(
                    self.threshold_intensity_label.text()
                )
        self.update_automated_controls()
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
            automated_threshold_edit = (
                self.active_tool == "automated" and self.automated_edit_mode
            )
            if self.active_tool == "threshold":
                threshold_highlight = self.current_threshold_highlight()
                annotation_markers = self.current_threshold_markers()
            elif automated_threshold_edit:
                threshold_highlight = self.current_threshold_highlight()
                annotation_markers = (
                    self.current_model_markers()
                    + self.current_annotation_markers()
                )
            else:
                threshold_highlight = None
                annotation_markers = (
                    self.current_model_markers()
                    + self.current_annotation_markers()
                )
            self.current_pixmap = None
            self.image_label.set_scene(
                marker_arr=self.current_fov,
                marker_color=self.color_dropdown.currentText(),
                dapi_arr=self.current_dapi_fov,
                show_dapi=self.dapi_checkbox.isChecked(),
                segmentation_boundary=boundary,
                threshold_highlight=threshold_highlight,
                annotation_markers=annotation_markers,
            )
        except Exception:
            self.status_label.setText(traceback.format_exc())
