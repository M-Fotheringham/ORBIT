from __future__ import annotations

import unittest
from types import MethodType
from unittest.mock import Mock

import numpy as np

from orbit.gui.fov_viewer import (
    AutomatedThresholdRangeWorker,
    OrbitFOVViewer,
)
from orbit.threshold import (
    intensity_threshold_from_range,
    intensity_threshold_from_slider,
    pooled_image_intensity_range,
)


class _Image:
    def __init__(self, channels):
        self.channels = channels
        self.get_channel = Mock(side_effect=lambda index: list(channels.values())[index])

    def get_channel_names(self):
        return list(self.channels)

    def get_shape(self):
        return (len(self.channels), *next(iter(self.channels.values())).shape)


class _LazyChannel:
    def __init__(self, shape):
        self.shape = shape
        self.sample_sizes = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("The complete source image must not be materialized.")

    def __getitem__(self, selection):
        rows, columns = selection
        height = len(range(*rows.indices(self.shape[0])))
        width = len(range(*columns.indices(self.shape[1])))
        self.sample_sizes.append(height * width)
        return np.arange(height * width, dtype=float).reshape(height, width)


class _Viewer:
    _automated_threshold_key = OrbitFOVViewer._automated_threshold_key
    _request_automated_threshold_range = OrbitFOVViewer._request_automated_threshold_range
    _update_threshold_value = OrbitFOVViewer._update_threshold_value
    on_automated_threshold_range_ready = OrbitFOVViewer.on_automated_threshold_range_ready
    on_automated_threshold_range_error = OrbitFOVViewer.on_automated_threshold_range_error

    def __init__(self, images):
        self.loaded_images = [{"img": image} for image in images]
        self.active_tool = "automated"
        self.is_loading = False
        self.current_fov = np.full((4, 4), 99_999.0)
        self.channel_dropdown = Mock()
        self.channel_dropdown.currentText.return_value = "CD8"
        self.threshold_intensity_slider = Mock()
        self.threshold_intensity_slider.value.return_value = 660
        self.threshold_intensity_slider.maximum.return_value = 1000
        self.threshold_intensity_label = Mock()
        self.automated_intensity_label = Mock()
        self.automated_status_label = Mock()
        self.threshold_intensity_value = None
        self.threshold_channel_name = None
        self.automated_threshold_ranges = {}
        self.automated_threshold_errors = {}
        self.automated_threshold_range_worker = None
        self.thread_pool = Mock()
        self.status_label = Mock()
        self._invalidate_threshold_predictions = Mock()
        self.update_automated_controls = Mock()
        self.update_display = Mock()


class AutomatedThresholdTests(unittest.TestCase):
    def setUp(self):
        self.images = [
            _Image({"DAPI": np.zeros((10, 10)), "CD8": np.arange(100).reshape(10, 10)}),
            _Image({"CD8": np.arange(100, 200).reshape(10, 10)}),
        ]

    def test_small_project_matches_exact_pooled_pixel_percentiles(self):
        expected = np.percentile(np.arange(200), [1, 99])
        np.testing.assert_allclose(pooled_image_intensity_range(self.images, "CD8"), expected)
        self.assertAlmostEqual(
            intensity_threshold_from_range(tuple(expected), 660),
            expected[0] + 0.66 * (expected[1] - expected[0]),
        )

    def test_image_order_does_not_change_calibration(self):
        np.testing.assert_array_equal(
            pooled_image_intensity_range(self.images, "CD8"),
            pooled_image_intensity_range(self.images[::-1], "CD8"),
        )

    def test_large_lazy_images_are_sampled_before_materialization(self):
        channels = [_LazyChannel((10_000, 10_000)), _LazyChannel((5000, 5000))]
        images = [_Image({"CD8": channel}) for channel in channels]
        first = pooled_image_intensity_range(images, "CD8", max_samples=1000)
        second = pooled_image_intensity_range(images, "CD8", max_samples=1000)
        self.assertEqual(first, second)
        self.assertLessEqual(sum(channel.sample_sizes[0] for channel in channels), 1000)
        self.assertTrue(all(channel.sample_sizes[0] > 0 for channel in channels))
        self.assertGreater(channels[0].sample_sizes[0], channels[1].sample_sizes[0])

    def test_nonfinite_pixels_are_excluded_from_the_pooled_range(self):
        images = [
            _Image({"CD8": np.array([[np.nan, np.inf]])}),
            _Image({"CD8": np.array([[1.0, 5.0, 9.0, -np.inf]])}),
        ]
        np.testing.assert_allclose(
            pooled_image_intensity_range(images, "CD8"), np.percentile([1, 5, 9], [1, 99])
        )
        with self.assertRaisesRegex(ValueError, "no finite"):
            pooled_image_intensity_range(images[:1], "CD8")

    def test_narrow_images_respect_the_sample_budget(self):
        for shape in ((10_000, 1), (1, 10_000)):
            with self.subTest(shape=shape):
                channel = _LazyChannel(shape)
                pooled_image_intensity_range([_Image({"CD8": channel})], "CD8", 3)
                self.assertGreater(channel.sample_sizes[0], 0)
                self.assertLessEqual(channel.sample_sizes[0], 3)

    def test_missing_marker_empty_project_and_invalid_budget_fail(self):
        for images, channel, budget in (
            (self.images, "missing", 1000), ([], "CD8", 1000), (self.images, "CD8", 0)
        ):
            with self.subTest(channel=channel, budget=budget), self.assertRaises(ValueError):
                pooled_image_intensity_range(images, channel, budget)

    def test_background_calibration_and_fov_navigation_use_the_same_threshold(self):
        viewer = _Viewer(self.images)
        viewer._update_threshold_value(invalidate=False)
        self.assertIsNone(viewer.threshold_intensity_value)
        worker = viewer.thread_pool.start.call_args.args[0]
        self.assertIsInstance(worker, AutomatedThresholdRangeWorker)
        worker.run()
        expected = intensity_threshold_from_range(tuple(np.percentile(np.arange(200), [1, 99])), 660)
        self.assertAlmostEqual(viewer.threshold_intensity_value, expected)
        for fov in (np.zeros((2, 2)), np.full((8, 8), 1e6)):
            viewer.current_fov = fov
            viewer._update_threshold_value(invalidate=False)
            self.assertAlmostEqual(viewer.threshold_intensity_value, expected)
        for image in self.images:
            image.get_channel.assert_called_once()
        self.assertIn("all loaded images", viewer.automated_intensity_label.setText.call_args.args[0])

    def test_adding_an_image_discards_the_previous_threshold_until_recalibrated(self):
        viewer = _Viewer(self.images[:1])
        viewer._update_threshold_value(invalidate=False)
        viewer.automated_threshold_range_worker.run()
        old_threshold = viewer.threshold_intensity_value
        viewer.loaded_images.append({"img": self.images[1]})
        viewer._update_threshold_value(invalidate=False)
        self.assertIsNone(viewer.threshold_intensity_value)
        viewer.automated_threshold_range_worker.run()
        self.assertGreater(viewer.threshold_intensity_value, old_threshold)

    def test_training_controls_wait_for_project_calibration(self):
        viewer = _Viewer(self.images)
        viewer.automated_edit_mode = True
        for state in viewer.loaded_images:
            state.update(cell_data=object(), segmentation_masks=object(), annotations={})
        for name in (
            "auto_phenotype_button", "automated_edit_button", "automated_intensity_slider",
            "automated_percent_slider", "automated_nucleus_checkbox", "automated_cytoplasm_checkbox",
            "automated_threshold_mask_button", "automated_positive_checkbox", "automated_negative_checkbox",
            "automated_phenotype_name", "automated_feature_selection_button", "automated_edit_probability_slider",
            "automated_probability_slider", "automated_default_feature_selection_button", "automated_buffer_slider",
            "rephenotype_button", "automated_modelled_checkbox", "automated_edit_modelled_checkbox",
            "automated_export_button",
        ):
            setattr(viewer, name, Mock())
        viewer.automated_nucleus_checkbox.isChecked.return_value = True
        viewer.automated_cytoplasm_checkbox.isChecked.return_value = True
        viewer.model_bundle = None
        viewer.update_automated_controls = MethodType(OrbitFOVViewer.update_automated_controls, viewer)
        viewer.update_automated_controls()
        viewer.auto_phenotype_button.setEnabled.assert_called_with(False)
        viewer.rephenotype_button.setEnabled.assert_called_with(False)
        viewer.automated_threshold_range_worker.run()
        viewer.auto_phenotype_button.setEnabled.assert_called_with(True)
        viewer.rephenotype_button.setEnabled.assert_called_with(True)

    def test_threshold_overlay_uses_the_calibrated_project_value(self):
        viewer = _Viewer(self.images)
        viewer.automated_edit_mode = True
        viewer.automated_threshold_mask_button = Mock()
        viewer.automated_threshold_mask_button.isChecked.return_value = True
        viewer.current_fov = np.array([[0.0, 75.0], [150.0, 200.0]])
        self.assertIsNone(OrbitFOVViewer.current_threshold_highlight(viewer))
        viewer.automated_threshold_range_worker.run()
        np.testing.assert_array_equal(
            OrbitFOVViewer.current_threshold_highlight(viewer),
            viewer.current_fov > viewer.threshold_intensity_value,
        )

    def test_stale_background_results_do_not_replace_current_project_calibration(self):
        viewer = _Viewer(self.images[:1])
        viewer._update_threshold_value(invalidate=False)
        worker = viewer.automated_threshold_range_worker
        old_key = worker.key
        viewer.loaded_images.append({"img": self.images[1]})
        worker.run()
        self.assertNotIn(old_key, viewer.automated_threshold_ranges)
        self.assertIsNone(viewer.threshold_intensity_value)
        self.assertEqual(len(viewer.automated_threshold_range_worker.images), 2)

    def test_changing_marker_uses_its_own_project_range(self):
        image = _Image({"CD8": np.arange(100).reshape(10, 10), "DAPI": np.full((10, 10), 200)})
        viewer = _Viewer([image])
        viewer._update_threshold_value(invalidate=False)
        viewer.automated_threshold_range_worker.run()
        viewer.channel_dropdown.currentText.return_value = "DAPI"
        viewer._update_threshold_value(invalidate=False)
        self.assertIsNone(viewer.threshold_intensity_value)
        viewer.automated_threshold_range_worker.run()
        self.assertEqual(viewer.threshold_intensity_value, 200.0)

    def test_worker_errors_leave_no_fov_threshold_fallback(self):
        viewer = _Viewer(self.images)
        viewer.channel_dropdown.currentText.return_value = "missing"
        viewer._update_threshold_value(invalidate=False)
        viewer.automated_threshold_range_worker.run()
        self.assertIsNone(viewer.threshold_intensity_value)
        self.assertIn(viewer._automated_threshold_key(), viewer.automated_threshold_errors)
        self.assertIsNone(viewer.automated_threshold_range_worker)

    def test_manual_threshold_slider_keeps_its_existing_fov_behavior(self):
        viewer = _Viewer(self.images)
        viewer.active_tool = "threshold"
        viewer.current_fov = np.arange(25).reshape(5, 5)
        viewer._update_threshold_value(invalidate=False)
        self.assertEqual(
            viewer.threshold_intensity_value, intensity_threshold_from_slider(viewer.current_fov, 660)
        )
        viewer.thread_pool.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
