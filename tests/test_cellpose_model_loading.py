import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.models.cellpose_segmentation import (
    CELLPOSE_SAM_MODEL,
    CELLPOSE_SAM_MODEL_SIZE_BYTES,
    ORBIT_CELLPOSE_MODEL_ENV,
    bundled_cellpose_sam_model_path,
    create_cellpose_sam_model,
)


class _StreamWritingCellposeModel:
    """Stand in for Cellpose's first-run tqdm model downloader."""

    def __init__(self, **kwargs):
        sys.stdout.write("model download")
        sys.stdout.flush()
        sys.stderr.write("model progress")
        sys.stderr.flush()
        self.kwargs = kwargs


def _fake_cellpose_modules(model_class):
    fake_models = types.SimpleNamespace(CellposeModel=model_class)
    fake_cellpose = types.SimpleNamespace(models=fake_models)
    return {
        "cellpose": fake_cellpose,
        "cellpose.models": fake_models,
    }


class CellposeModelLoadingTests(unittest.TestCase):
    def test_model_download_works_without_console_streams(self):
        modules = _fake_cellpose_modules(_StreamWritingCellposeModel)
        with (
            patch.dict(sys.modules, modules),
            patch.object(sys, "stdout", None),
            patch.object(sys, "stderr", None),
        ):
            model = create_cellpose_sam_model(gpu=False)

            self.assertEqual(model.kwargs, {
                "gpu": False,
                "pretrained_model": CELLPOSE_SAM_MODEL,
                "use_bfloat16": False,
            })
            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)

    def test_installed_model_path_bypasses_cellpose_download(self):
        modules = _fake_cellpose_modules(_StreamWritingCellposeModel)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / CELLPOSE_SAM_MODEL
            with model_path.open("wb") as stream:
                stream.truncate(CELLPOSE_SAM_MODEL_SIZE_BYTES)

            with (
                patch.dict(
                    os.environ,
                    {ORBIT_CELLPOSE_MODEL_ENV: str(model_path)},
                    clear=False,
                ),
                patch.dict(sys.modules, modules),
            ):
                model = create_cellpose_sam_model(gpu=False)

        self.assertEqual(model.kwargs, {
            "gpu": False,
            "pretrained_model": str(model_path.resolve()),
            "use_bfloat16": False,
        })

    def test_invalid_bundled_model_is_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / CELLPOSE_SAM_MODEL
            model_path.write_bytes(b"incomplete")
            with patch.dict(
                os.environ,
                {ORBIT_CELLPOSE_MODEL_ENV: str(model_path)},
                clear=False,
            ):
                self.assertIsNone(bundled_cellpose_sam_model_path())

    def test_incomplete_standalone_install_does_not_download(self):
        modules = _fake_cellpose_modules(_StreamWritingCellposeModel)
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                sys,
                "frozen",
                True,
                create=True,
            ),
            patch(
                "orbit.models.cellpose_segmentation."
                "bundled_cellpose_sam_model_path",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "connected to the internet"),
        ):
            create_cellpose_sam_model(gpu=False)

    def test_model_loading_preserves_real_console_streams(self):
        modules = _fake_cellpose_modules(_StreamWritingCellposeModel)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(sys.modules, modules),
            patch.object(sys, "stdout", stdout),
            patch.object(sys, "stderr", stderr),
        ):
            create_cellpose_sam_model(gpu=False)

            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)
            self.assertEqual(stdout.getvalue(), "model download")
            self.assertEqual(stderr.getvalue(), "model progress")

    def test_missing_console_streams_are_restored_after_model_error(self):
        class FailingCellposeModel(_StreamWritingCellposeModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                raise RuntimeError("synthetic model failure")

        modules = _fake_cellpose_modules(FailingCellposeModel)
        with (
            patch.dict(sys.modules, modules),
            patch.object(sys, "stdout", None),
            patch.object(sys, "stderr", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic model failure"):
                create_cellpose_sam_model(gpu=False)

            self.assertIsNone(sys.stdout)
            self.assertIsNone(sys.stderr)


if __name__ == "__main__":
    unittest.main()
