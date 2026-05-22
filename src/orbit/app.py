from PySide6.QtWidgets import QApplication

from orbit.gui.fov_viewer import OrbitFOVViewer


def main():
    app = QApplication([])

    window = OrbitFOVViewer()
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
