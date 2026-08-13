import unittest

import numpy as np
import pandas as pd

from orbit.coordinates import centroid_columns, centroid_pixel_coordinates


class CentroidCoordinateTests(unittest.TestCase):
    def test_pixel_columns_are_preferred_over_micrometre_columns(self):
        cells = pd.DataFrame({
            "Centroid X µm": [100.0, 200.0],
            "Centroid Y µm": [50.0, 75.0],
            "Centroid Y px": [500.0, 750.0],
            "Centroid X px": [1000.0, 2000.0],
        })

        self.assertEqual(
            centroid_columns(cells),
            ("Centroid X px", "Centroid Y px"),
        )
        coordinates = centroid_pixel_coordinates(cells, pixel_size_um=2.0)
        np.testing.assert_array_equal(coordinates["x"], [1000.0, 2000.0])
        np.testing.assert_array_equal(coordinates["y"], [500.0, 750.0])

    def test_micrometre_only_coordinates_use_image_pixel_size(self):
        cells = pd.DataFrame({
            "Centroid X µm": [100.0, 200.0],
            "Centroid Y µm": [50.0, 75.0],
        })

        coordinates = centroid_pixel_coordinates(cells, pixel_size_um=2.0)

        np.testing.assert_array_equal(coordinates["x"], [50.0, 100.0])
        np.testing.assert_array_equal(coordinates["y"], [25.0, 37.5])

    def test_micrometre_coordinates_require_a_valid_pixel_size(self):
        cells = pd.DataFrame({
            "Centroid X µm": [100.0],
            "Centroid Y µm": [50.0],
        })

        with self.assertRaisesRegex(ValueError, "physical pixel size"):
            centroid_pixel_coordinates(cells)

    def test_fuzzy_columns_still_prefer_pixels(self):
        cells = pd.DataFrame({
            "cell_center_x_microns": [100.0],
            "cell_center_y_microns": [50.0],
            "cell_center_x_pixels": [1000.0],
            "cell_center_y_pixels": [500.0],
        })

        self.assertEqual(
            centroid_columns(cells),
            ("cell_center_x_pixels", "cell_center_y_pixels"),
        )


if __name__ == "__main__":
    unittest.main()
