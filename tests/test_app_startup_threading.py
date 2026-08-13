import ast
import unittest
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "src" / "orbit" / "app.py"


class AppStartupThreadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

    def test_viewer_is_not_imported_by_a_qthread_worker(self):
        imported_qtcore_names = {
            alias.name
            for node in self.tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "PySide6.QtCore"
            for alias in node.names
        }
        class_names = {
            node.name
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
        }

        self.assertNotIn("QThread", imported_qtcore_names)
        self.assertNotIn("ViewerImportWorker", class_names)

    def test_main_schedules_gui_thread_viewer_loading(self):
        controller = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "StartupController"
        )
        method_names = {
            node.name
            for node in controller.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("load_main_window", method_names)

        timer_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "QTimer"
            and node.func.attr == "singleShot"
        ]
        self.assertEqual(len(timer_calls), 1)
        callback = timer_calls[0].args[1]
        self.assertIsInstance(callback, ast.Attribute)
        self.assertEqual(callback.attr, "load_main_window")


if __name__ == "__main__":
    unittest.main()
