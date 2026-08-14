import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]


class CellposeBundleConfigurationTests(unittest.TestCase):
    def test_pyinstaller_does_not_bundle_the_large_model(self):
        specification = (REPOSITORY_ROOT / "ORBIT.spec").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("cellpose_models", specification)
        self.assertNotIn("cpsam_v2", specification)
        self.assertIn("CELLPOSE_SAM_MODEL_NOTICE.txt", specification)

    def test_windows_build_does_not_stage_the_model(self):
        script = (REPOSITORY_ROOT / "build_windows_app.bat").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("scripts\\stage_cellpose_model.py", script)
        self.assertIn("pyinstaller --clean", script)
        self.assertIn(
            'if exist "%ROOT%\\dist\\ORBIT\\cellpose_models\\cpsam_v2"',
            script,
        )

    def test_installer_downloads_and_verifies_the_model(self):
        installer = (
            REPOSITORY_ROOT / "installer" / "orbit_installer_setup.iss"
        ).read_text(encoding="utf-8")

        self.assertIn('Source: "{#MyDistDir}\\*"', installer)
        self.assertIn("recursesubdirs", installer)
        self.assertIn("createallsubdirs", installer)
        self.assertIn("external download ignoreversion", installer)
        self.assertIn("DestDir: \"{app}\\cellpose_models\"", installer)
        self.assertIn(
            "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
            installer,
        )
        self.assertIn("Check: ShouldInstallCellposeModel", installer)

    def test_installer_and_application_share_the_taskbar_identity(self):
        installer = (
            REPOSITORY_ROOT / "installer" / "orbit_installer_setup.iss"
        ).read_text(encoding="utf-8")
        application = (
            REPOSITORY_ROOT / "src" / "orbit" / "app.py"
        ).read_text(encoding="utf-8")

        identity = "MFotheringham.ORBIT"
        self.assertIn(identity, installer)
        self.assertIn(identity, application)


if __name__ == "__main__":
    unittest.main()
