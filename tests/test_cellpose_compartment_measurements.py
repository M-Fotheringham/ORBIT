from __future__ import annotations

import unittest

import numpy as np

from orbit.models.cellpose_segmentation import measure_segmented_cells


class _MeasurementImage:
    def __init__(self, channel_names, channels):
        self._channel_names = list(channel_names)
        self._channels = list(channels)

    def get_channel_names(self):
        return list(self._channel_names)

    def get_channel(self, index):
        return self._channels[index]


class CellposeCompartmentMeasurementTests(unittest.TestCase):
    def test_exports_cell_nucleus_and_membrane_intensities(self):
        masks = np.zeros((9, 9), dtype=np.uint32)
        masks[1:8, 1:8] = 1

        # A one-pixel inward buffer leaves a 3x3 nucleus under ORBIT's
        # boundary-distance convention. Give each compartment a distinct
        # intensity so accidental whole-cell reuse is detected.
        channel = np.zeros(masks.shape, dtype=np.float32)
        channel[masks == 1] = 2.0
        channel[3:6, 3:6] = 10.0
        image = _MeasurementImage(["CD8"], [channel])

        cells = measure_segmented_cells(
            masks,
            image,
            pixel_size_um=1.0,
            membrane_width_um=1.0,
        )

        cell = cells.iloc[0]
        self.assertEqual(cell["Nucleus Area px"], 9)
        self.assertEqual(cell["Membrane Area px"], 40)
        self.assertEqual(
            cell["Nucleus Area px"] + cell["Membrane Area px"],
            cell["Area px"],
        )
        self.assertEqual(cell["CD8: Nucleus Mean"], 10.0)
        self.assertEqual(cell["CD8: Membrane Mean"], 2.0)
        self.assertAlmostEqual(
            cell["CD8: Cell Mean"],
            (9 * 10.0 + 40 * 2.0) / 49,
        )
        for compartment in ("Cell", "Nucleus", "Membrane"):
            for statistic in ("Mean", "Std Dev", "Min", "Max"):
                self.assertIn(f"CD8: {compartment} {statistic}", cells.columns)

    def test_small_cell_has_nan_nucleus_statistics_and_valid_membrane(self):
        masks = np.zeros((5, 5), dtype=np.uint32)
        masks[2, 2] = 1
        image = _MeasurementImage(
            ["DAPI"],
            [np.arange(25, dtype=np.float32).reshape(5, 5)],
        )

        cells = measure_segmented_cells(
            masks,
            image,
            pixel_size_um=1.0,
            membrane_width_um=2.0,
        )

        cell = cells.iloc[0]
        self.assertEqual(cell["Nucleus Area px"], 0)
        self.assertEqual(cell["Membrane Area px"], 1)
        self.assertTrue(np.isnan(cell["DAPI: Nucleus Mean"]))
        self.assertEqual(cell["DAPI: Membrane Mean"], 12.0)
        self.assertEqual(cell["DAPI: Cell Mean"], 12.0)

    def test_rejects_invalid_physical_scale(self):
        masks = np.ones((3, 3), dtype=np.uint32)
        image = _MeasurementImage(["CD8"], [np.ones((3, 3))])

        with self.assertRaisesRegex(ValueError, "pixel size"):
            measure_segmented_cells(masks, image, pixel_size_um=0)
        with self.assertRaisesRegex(ValueError, "membrane-compartment width"):
            measure_segmented_cells(masks, image, membrane_width_um=0)


if __name__ == "__main__":
    unittest.main()
