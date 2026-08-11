# Keep ORBIT's bundled font, artwork, and loading animations in Nuitka builds.
# nuitka-project: --include-package-data=orbit

import ctypes
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from orbit.gui.startup_splash import StartupVideoSplash
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


class ViewerImportWorker(QObject):
    """Import the heavyweight viewer while the startup video remains active."""

    loaded = Signal(object)
    failed = Signal(str)

    @Slot()
    def run(self):
        try:
            from orbit.gui.fov_viewer import OrbitFOVViewer
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.loaded.emit(OrbitFOVViewer)


class StartupController(QObject):
    """Finish startup on Qt's GUI thread after the viewer import completes."""

    def __init__(self, application, splash, icon_path, icon):
        super().__init__(application)
        self.application = application
        self.splash = splash
        self.icon_path = icon_path
        self.icon = icon
        self.window = None
        self.failure = None

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
    splash = StartupVideoSplash(icon=icon)
    splash.start()

    import_thread = QThread()
    import_worker = ViewerImportWorker()
    import_worker.moveToThread(import_thread)
    import_thread.started.connect(import_worker.run)
    import_worker.loaded.connect(import_thread.quit)
    import_worker.failed.connect(import_thread.quit)
    import_thread.finished.connect(import_worker.deleteLater)

    startup_controller = StartupController(app, splash, icon_path, icon)
    import_worker.loaded.connect(startup_controller.show_main_window)
    import_worker.failed.connect(startup_controller.handle_startup_failure)

    # The popup has already painted its first frame; begin imports immediately.
    QTimer.singleShot(0, import_thread.start)

    exit_code = app.exec()
    if import_thread.isRunning():
        import_thread.quit()
        import_thread.wait()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
