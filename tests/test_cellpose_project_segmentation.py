import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.models import cellpose_segmentation


class _ProjectImage:
    def __init__(self, channel_names):
        self.path = Path("project_image.tif")
        self._channel_names = list(channel_names)

    def get_shape(self):
        return (len(self._channel_names), 100, 120)

    def get_channel_names(self):
        return list(self._channel_names)

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


if __name__ == "__main__":
    unittest.main()
