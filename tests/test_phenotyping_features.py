import unittest

from orbit.models.phenotyping_features import is_pixel_unit_measurement


class PhenotypingFeatureTests(unittest.TestCase):
    def test_pixel_unit_morphology_columns_are_excluded(self):
        for column in (
            "Perimeter px",
            "Area_px",
            "Equivalent diameter (pixels)",
            "Nucleus Area px²",
            "Membrane Area pixel^2",
        ):
            with self.subTest(column=column):
                self.assertTrue(is_pixel_unit_measurement(column))

    def test_micron_and_fluorescence_columns_are_retained(self):
        for column in (
            "Perimeter µm",
            "Area µm²",
            "Equivalent diameter microns",
            "CD8: Membrane Mean",
            "Positive pixel %",
        ):
            with self.subTest(column=column):
                self.assertFalse(is_pixel_unit_measurement(column))


if __name__ == "__main__":
    unittest.main()
