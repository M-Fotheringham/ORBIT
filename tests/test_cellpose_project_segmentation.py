import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from orbit.models import cellpose_segmentation


class _ProjectImage:
    def __init__(self, channel_names, channels=None):
        self.path = Path("project_image.tif")
        self._channel_names = list(channel_names)
        self._channels = channels

    def get_shape(self):
        return (len(self._channel_names), 100, 120)

    def get_channel_names(self):
        return list(self._channel_names)

    def get_channel(self, index):
        if self._channels is None:
            raise AssertionError("No channel data were supplied for this test.")
        return self._channels[index]

    @staticmethod
    def get_pixel_size_um(default=None):
        return default


class ProjectSegmentationSelectionTests(unittest.TestCase):
    def test_marker_names_survive_dapi_fov_selection(self):
        image = _ProjectImage(["DAPI", "CD8"])
        candidate_fov = object()
        model = object()

        with (
            patch.object(
                cellpose_segmentation,
                "tiled_segmentation_fovs",
                return_value=[candidate_fov],
            ),
            patch.object(
                cellpose_segmentation,
                "select_dapi_positive_fovs",
                return_value=[candidate_fov],
            ),
            patch.object(
                cellpose_segmentation,
                "create_cellpose_sam_model",
                return_value=model,
            ),
            patch.object(
                cellpose_segmentation,
                "segment_image",
                return_value={"segmented": True},
            ) as segment_image,
        ):
            result = cellpose_segmentation.segment_project_images(
                [image],
                ["CD8"],
                dapi_positive_only=True,
            )

        self.assertEqual(result, [{"segmented": True}])
        self.assertEqual(segment_image.call_args.args[1], ["CD8"])
        self.assertIs(segment_image.call_args.args[2], model)
        self.assertEqual(segment_image.call_args.kwargs["fovs"], [candidate_fov])

    def test_generic_marker_names_survive_all_fov_selection(self):
        image = _ProjectImage(["Channel 0", "Channel 1"])
        candidate_fov = object()

        with (
            patch.object(
                cellpose_segmentation,
                "tiled_segmentation_fovs",
                return_value=[candidate_fov],
            ),
            patch.object(
                cellpose_segmentation,
                "create_cellpose_sam_model",
                return_value=object(),
            ),
            patch.object(
                cellpose_segmentation,
                "segment_image",
                return_value={"segmented": True},
            ) as segment_image,
        ):
            cellpose_segmentation.segment_project_images(
                [image],
                ["Channel 1"],
                dapi_positive_only=True,
            )

        self.assertEqual(segment_image.call_args.args[1], ["Channel 1"])
        self.assertEqual(segment_image.call_args.kwargs["fovs"], [candidate_fov])

    def test_explicit_generic_nuclear_channel_controls_fov_selection(self):
        image = _ProjectImage(["Channel 0", "Channel 8"])
        candidate_fov = object()

        with (
            patch.object(
                cellpose_segmentation,
                "tiled_segmentation_fovs",
                return_value=[candidate_fov],
            ),
            patch.object(
                cellpose_segmentation,
                "select_dapi_positive_fovs",
                return_value=[candidate_fov],
            ) as select_positive_fovs,
            patch.object(
                cellpose_segmentation,
                "create_cellpose_sam_model",
                return_value=object(),
            ),
            patch.object(
                cellpose_segmentation,
                "segment_image",
                return_value={"segmented": True},
            ) as segment_image,
        ):
            cellpose_segmentation.segment_project_images(
                [image],
                ["Channel 8"],
                nuclear_channel_name="Channel 0",
                dapi_positive_only=True,
            )

        self.assertEqual(select_positive_fovs.call_args.args[2], 0)
        self.assertEqual(
            segment_image.call_args.kwargs["nuclear_channel_name"],
            "Channel 0",
        )

    def test_generic_nuclear_channel_is_added_to_cellpose_input(self):
        base = np.arange(36, dtype=np.float32).reshape(6, 6)
        image = _ProjectImage(
            ["Channel 0", "Channel 8"],
            channels=[base, base[::-1]],
        )

        model_input, nuclear_name = cellpose_segmentation.build_cellpose_input(
            image,
            ["Channel 8"],
            nuclear_channel_name="Channel 0",
        )

        self.assertEqual(nuclear_name, "Channel 0")
        self.assertEqual(model_input.shape, (6, 6, 3))
        self.assertTrue(np.any(model_input[..., 0]))
        self.assertTrue(np.any(model_input[..., 1]))
        self.assertFalse(np.any(model_input[..., 2]))

    def test_dapi_remains_the_default_nuclear_channel(self):
        base = np.arange(36, dtype=np.float32).reshape(6, 6)
        image = _ProjectImage(
            ["DAPI", "CD8"],
            channels=[base, base[::-1]],
        )

        _model_input, nuclear_name = cellpose_segmentation.build_cellpose_input(
            image,
            ["CD8"],
        )

        self.assertEqual(nuclear_name, "DAPI")

    def test_nuclear_channel_cannot_also_be_a_membrane_marker(self):
        base = np.arange(36, dtype=np.float32).reshape(6, 6)
        image = _ProjectImage(
            ["Channel 0", "Channel 8"],
            channels=[base, base[::-1]],
        )

        with self.assertRaisesRegex(ValueError, "cannot also be used"):
            cellpose_segmentation.build_cellpose_input(
                image,
                ["Channel 0"],
                nuclear_channel_name="Channel 0",
            )


if __name__ == "__main__":
    unittest.main()
