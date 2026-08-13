import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "stage_cellpose_model.py"
)
SPEC = importlib.util.spec_from_file_location("stage_cellpose_model", SCRIPT_PATH)
staging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staging)


class CellposeModelStagingTests(unittest.TestCase):
    def test_valid_existing_model_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / staging.MODEL_NAME
            destination.write_bytes(b"model")
            with (
                patch.object(staging, "MODEL_SIZE_BYTES", len(b"model")),
                patch.object(
                    staging,
                    "MODEL_SHA256",
                    staging.hashlib.sha256(b"model").hexdigest(),
                ),
                patch.object(staging, "download_model") as download,
            ):
                result = staging.stage_model(destination)

            self.assertEqual(result, destination.resolve())
            download.assert_not_called()

    def test_source_is_verified_and_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "existing-cpsam-v2"
            destination = root / "build" / staging.MODEL_NAME
            source.write_bytes(b"verified model")
            with (
                patch.object(staging, "MODEL_SIZE_BYTES", len(b"verified model")),
                patch.object(
                    staging,
                    "MODEL_SHA256",
                    staging.hashlib.sha256(b"verified model").hexdigest(),
                ),
            ):
                staging.stage_model(destination, source=source)

            self.assertEqual(destination.read_bytes(), b"verified model")
            self.assertFalse(destination.with_name(".cpsam_v2.part").exists())


if __name__ == "__main__":
    unittest.main()
