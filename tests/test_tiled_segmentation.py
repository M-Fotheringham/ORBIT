import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from orbit.models.tiled_segmentation import (
    AstroPathMaskStitcher,
    DEFAULT_SEGMENTATION_FOV_SIZE,
    tiled_segmentation_fovs,
)


class TiledSegmentationTests(unittest.TestCase):
    def test_default_segmentation_fovs_are_1024_pixels(self):
        self.assertEqual(DEFAULT_SEGMENTATION_FOV_SIZE, 1024)
        fovs = tiled_segmentation_fovs((2200, 2400))
        self.assertEqual(fovs[0].height, 1024)
        self.assertEqual(fovs[0].width, 1024)
        self.assertEqual(
            sorted({fov.x0 for fov in fovs})[1],
            round(1024 * 0.80),
        )

    def test_fovs_cover_image_and_primary_regions_have_one_owner(self):
        fovs = tiled_segmentation_fovs((1000, 1100), fov_size=512, overlap=0.20)
        self.assertEqual(max(fov.y1 for fov in fovs), 1000)
        self.assertEqual(max(fov.x1 for fov in fovs), 1100)
        x_origins = sorted({fov.x0 for fov in fovs})
        self.assertEqual(x_origins[1] - x_origins[0], round(512 * 0.80))

        for y in (0, 100, 499, 999):
            for x in (0, 100, 500, 1099):
                owners = [
                    fov for fov in fovs
                    if fov.primary_y0 <= y < fov.primary_y1
                    and fov.primary_x0 <= x < fov.primary_x1
                ]
                self.assertEqual(len(owners), 1)

    def test_stitcher_keeps_only_primary_region_centroids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            left, right = tiled_segmentation_fovs(
                (100, 180), fov_size=100, overlap=0.20
            )
            stitcher = AstroPathMaskStitcher(
                image_shape=(100, 180),
                cell_path=directory / "cells.tsv",
                mask_path=directory / "mask.tif",
                model_name="test",
                marker_names=["Membrane"],
                nuclear_channel_name="DAPI",
            )

            left_mask = np.zeros((100, 100), dtype=np.uint32)
            left_mask[10:20, 10:20] = 1
            left_mask[40:50, 90:100] = 2
            left_cells = pd.DataFrame({
                "Cell ID": [1, 2],
                "Centroid X px": [15.0, 95.0],
                "Centroid Y px": [15.0, 45.0],
            })
            self.assertEqual(
                stitcher.add_fov(left, left_mask, left_cells, 0.5), 1
            )

            right_mask = np.zeros((100, 100), dtype=np.uint32)
            right_mask[10:20, 10:20] = 1
            right_mask[40:50, 0:10] = 2
            right_cells = pd.DataFrame({
                "Cell ID": [1, 2],
                "Centroid X px": [15.0, 5.0],
                "Centroid Y px": [15.0, 45.0],
            })
            self.assertEqual(
                stitcher.add_fov(right, right_mask, right_cells, 0.5), 1
            )

            cell_path, mask_path, cell_count = stitcher.finalize()
            self.assertEqual(cell_count, 2)
            cells = pd.read_csv(cell_path, sep="\t")
            self.assertEqual(cells["Cell ID"].tolist(), [1, 2])
            self.assertEqual(cells["Centroid X px"].tolist(), [15.0, 95.0])
            mask = tifffile.memmap(mask_path)
            self.assertEqual(set(np.unique(mask)), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
