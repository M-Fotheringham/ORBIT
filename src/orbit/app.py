import ctypes
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from orbit.gui.fov_viewer import OrbitFOVViewer


def find_icon_path():
    candidates = [
        # Compiled standalone distribution
        Path(sys.argv[0]).resolve().parent
        / "docs"
        / "figs"
        / "icon_logo.ico",

        # Running from the source repository
        Path(__file__).resolve().parents[2]
        / "docs"
        / "figs"
        / "icon_logo.ico",
    ]

    return next((path for path in candidates if path.is_file()), None)


def main():
    if sys.platform == "win32":
        # Gives Windows a stable taskbar identity for ORBIT.
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "MFotheringham.ORBIT"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("ORBIT")
    app.setApplicationDisplayName("ORBIT")

    icon_path = find_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    window = OrbitFOVViewer()

    if icon_path is not None:
        window.setWindowIcon(QIcon(str(icon_path)))

    window.show()
    app.exec()


if __name__ == "__main__":
    main()
