from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orbit.project import relocated_reference, remember_relocation


class ProjectRelocationTests(unittest.TestCase):
    def test_relocated_sibling_in_same_directory_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_image = root / "old" / "slide.tif"
            selected_image = root / "new" / "slide.tif"
            selected_image.parent.mkdir(parents=True)
            selected_image.touch()
            selected_cells = selected_image.parent / "cells.tsv"
            selected_cells.touch()

            roots = {}
            remember_relocation(missing_image, selected_image, roots)

            self.assertEqual(
                relocated_reference(root / "old" / "cells.tsv", roots),
                str(selected_cells.resolve()),
            )

    def test_matching_ancestor_relocates_assets_in_sibling_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_image = root / "old" / "study" / "images" / "slide.tif"
            selected_image = root / "new" / "study" / "images" / "slide.tif"
            selected_image.parent.mkdir(parents=True)
            selected_image.touch()
            selected_mask = (
                root / "new" / "study" / "segmentations" / "mask.tif"
            )
            selected_mask.parent.mkdir(parents=True)
            selected_mask.touch()

            roots = {}
            remember_relocation(missing_image, selected_image, roots)

            self.assertEqual(
                relocated_reference(
                    root / "old" / "study" / "segmentations" / "mask.tif",
                    roots,
                ),
                str(selected_mask.resolve()),
            )

    def test_nonexistent_relocated_candidate_is_not_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_image = root / "old" / "slide.tif"
            selected_image = root / "new" / "slide.tif"
            selected_image.parent.mkdir(parents=True)
            selected_image.touch()

            roots = {}
            remember_relocation(missing_image, selected_image, roots)

            self.assertIsNone(
                relocated_reference(root / "old" / "missing.tsv", roots)
            )

    def test_longest_matching_relocation_root_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old"
            broad_new = root / "broad"
            specific_new = root / "specific"
            broad_target = broad_new / "study" / "slide.tif"
            specific_target = specific_new / "slide.tif"
            broad_target.parent.mkdir(parents=True)
            specific_target.parent.mkdir(parents=True)
            broad_target.touch()
            specific_target.touch()
            roots = {
                old: broad_new,
                old / "study": specific_new,
            }

            self.assertEqual(
                relocated_reference(old / "study" / "slide.tif", roots),
                str(specific_target.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
