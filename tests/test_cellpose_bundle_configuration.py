import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]


class CellposeBundleConfigurationTests(unittest.TestCase):
    def test_pyinstaller_copies_the_staged_model(self):
        specification = (REPOSITORY_ROOT / "ORBIT.spec").read_text(
            encoding="utf-8"
        )

        self.assertIn("build' / 'cellpose_models' / 'cpsam_v2'", specification)
        self.assertIn("(str(cellpose_model), 'cellpose_models')", specification)
        self.assertIn("stage_cellpose_model.py", specification)

    def test_windows_build_stages_before_packaging(self):
        script = (REPOSITORY_ROOT / "build_windows_app.bat").read_text(
            encoding="utf-8"
        )

        stage_index = script.index("scripts\\stage_cellpose_model.py")
        package_index = script.index("pyinstaller --clean")
        self.assertLess(stage_index, package_index)

    def test_installer_recursively_includes_the_distribution(self):
        installer = (
            REPOSITORY_ROOT / "installer" / "orbit_installer_setup.iss"
        ).read_text(encoding="utf-8")

        self.assertIn('Source: "{#MyAppDistDir}\\*"', installer)
        self.assertIn("recursesubdirs", installer)
        self.assertIn("createallsubdirs", installer)


if __name__ == "__main__":
    unittest.main()
