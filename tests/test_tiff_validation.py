import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from orbit.image import QPTiffImage


class TiffDataBoundsTests(unittest.TestCase):
    def test_complete_tiff_opens_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.tif"
            tifffile.imwrite(
                path,
                np.arange(3 * 32 * 24, dtype=np.uint16).reshape(3, 32, 24),
                photometric="rgb",
                planarconfig="separate",
                metadata={"axes": "CYX"},
            )

            image = QPTiffImage(path)
            try:
                self.assertEqual(image.get_shape(), (3, 32, 24))
            finally:
                image.close()

    def test_truncated_tiff_is_rejected_before_pixel_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.tif"
            tifffile.imwrite(
                path,
                np.arange(3 * 32 * 24, dtype=np.uint16).reshape(3, 32, 24),
                photometric="rgb",
                planarconfig="separate",
                metadata={"axes": "CYX"},
            )
            with path.open("r+b") as stream:
                stream.truncate(path.stat().st_size - 128)

            with self.assertRaisesRegex(
                ValueError,
                r"TIFF pixel data are truncated.*Re-copy or re-export",
            ):
                QPTiffImage(path)


if __name__ == "__main__":
    unittest.main()
