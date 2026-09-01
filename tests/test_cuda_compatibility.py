import sys
import types
import unittest
from unittest.mock import patch

from orbit.models.cellpose_segmentation import (
    CUDA_MINIMUM_WINDOWS_DRIVER,
    cuda_compatibility_details,
)


def _fake_torch(*, available, name="TITAN RTX", capability=(7, 5), archs=None):
    cuda = types.SimpleNamespace(
        is_available=lambda: bool(available),
        device_count=lambda: 1 if available else 0,
        get_device_name=lambda _index: name,
        get_device_capability=lambda _index: capability,
        get_arch_list=lambda: list(archs or []),
    )
    return types.SimpleNamespace(
        __version__="2.10.0+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=cuda,
    )


class CudaCompatibilityTests(unittest.TestCase):
    def test_reports_minimum_driver_when_cuda_cannot_initialize(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(available=False)}):
            details = cuda_compatibility_details()

        self.assertFalse(details["available"])
        self.assertIn(CUDA_MINIMUM_WINDOWS_DRIVER, details["reason"])
        self.assertIn("CUDA 12.8", details["reason"])

    def test_accepts_titan_rtx_in_shared_cuda_build(self):
        with patch.dict(
            sys.modules,
            {"torch": _fake_torch(available=True, archs=["sm_75", "sm_120"])},
        ):
            details = cuda_compatibility_details()

        self.assertTrue(details["available"])
        self.assertEqual(details["device_name"], "TITAN RTX")
        self.assertEqual(details["architecture"], "sm_75")

    def test_rejects_gpu_architecture_missing_from_wheel(self):
        with patch.dict(
            sys.modules,
            {
                "torch": _fake_torch(
                    available=True,
                    name="TITAN Xp",
                    capability=(6, 1),
                    archs=["sm_75", "sm_120"],
                )
            },
        ):
            details = cuda_compatibility_details()

        self.assertFalse(details["available"])
        self.assertIn("compute capability 6.1", details["reason"])
        self.assertIn("TITAN V", details["reason"])


if __name__ == "__main__":
    unittest.main()
