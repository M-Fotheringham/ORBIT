# Keep ORBIT's bundled font, artwork, and loading images in Nuitka builds.
# nuitka-project: --include-package-data=orbit

import ctypes
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from orbit.gui.startup_splash import StartupSplash
from orbit.resources import install_cosmic_alien_font


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


class StartupController(QObject):
    """Import and construct the viewer on Qt's GUI thread.

    Importing the viewer also imports Napari and VisPy. Those packages may
    initialize Qt and OpenGL state at import time, so doing that work in a
    background ``QThread`` can violate Qt's thread-affinity rules and terminate
    the Windows process with a native segmentation fault. A zero-delay timer
    lets the still-image splash paint first, then performs all GUI-related
    initialization on the application thread.
    """

    def __init__(self, application, splash, icon_path, icon):
        super().__init__(application)
        self.application = application
        self.splash = splash
        self.icon_path = icon_path
        self.icon = icon
        self.window = None
        self.failure = None

    @Slot()
    def load_main_window(self):
        try:
            from orbit.gui.fov_viewer import OrbitFOVViewer
        except Exception:
            self.handle_startup_failure(traceback.format_exc())
            return
        self.show_main_window(OrbitFOVViewer)

    @Slot(object)
    def show_main_window(self, viewer_class):
        try:
            self.window = viewer_class()
        except Exception:
            self.handle_startup_failure(traceback.format_exc())
            return

        if self.icon_path is not None:
            self.window.setWindowIcon(self.icon)
        self.window.show()
        self.application.processEvents()
        self.splash.finish()

    @Slot(str)
    def handle_startup_failure(self, details):
        if self.failure is not None:
            return
        self.failure = details
        self.splash.finish()
        QMessageBox.critical(
            None,
            "Could not start ORBIT",
            "ORBIT could not finish starting.\n\n" + details,
        )
        self.application.quit()


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
    install_cosmic_alien_font(app)

    icon_path = find_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    icon = QIcon(str(icon_path)) if icon_path is not None else QIcon()
    splash = StartupSplash(icon=icon)
    splash.start()

    startup_controller = StartupController(app, splash, icon_path, icon)

    # The popup has already painted its selected image. Defer one event-loop
    # turn, but keep Napari/VisPy imports on the Qt application thread.
    QTimer.singleShot(0, startup_controller.load_main_window)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
