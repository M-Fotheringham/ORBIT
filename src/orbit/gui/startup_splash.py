"""Instant still-image loading popup shown while ORBIT imports its viewer."""

import secrets

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from orbit.resources import (
    GOALIE_ASSISTANT_IMAGE,
    LOADING_IMAGES,
    asset_path,
)


class StartupSplash(QDialog):
    """Compact startup window using a randomized hockey collision still."""

    PLAYER_WIDTH = 480
    PLAYER_HEIGHT = 270
    SETTINGS_KEY = "startup/last_loading_image"

    def __init__(self, icon=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Starting ORBIT")
        self.setWindowFlags(
            Qt.SplashScreen
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(False)
        if icon is not None and not icon.isNull():
            self.setWindowIcon(icon)

        title = QLabel("ORBIT is preparing your workspace…")
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title.setStyleSheet(
            "color: #173b3f; font-size: 15px; padding-left: 4px;"
        )

        goalie_label = QLabel(self)
        goalie_label.setFixedSize(68, 68)
        goalie_label.setAlignment(Qt.AlignCenter)
        goalie_label.setStyleSheet(
            "background: #071c2c; border: 1px solid #70bfc2;"
        )
        goalie = QPixmap(str(asset_path(GOALIE_ASSISTANT_IMAGE)))
        if goalie.isNull():
            raise RuntimeError("Could not load the bundled goalie assistant image.")
        goalie_label.setPixmap(
            goalie.scaled(
                66,
                66,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "background: #071013; border: 1px solid #70bfc2;"
        )

        image_name = self._choose_image()
        still = QPixmap(str(asset_path(image_name)))
        if still.isNull():
            raise RuntimeError(f"Could not load the startup image: {image_name}")
        self.image_label.setPixmap(
            still.scaled(
                self.PLAYER_WIDTH,
                self.PLAYER_HEIGHT,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        self.progress.setStyleSheet(
            "QProgressBar {"
            " border: none; background: #dff3f3; border-radius: 3px;"
            "}"
            "QProgressBar::chunk {"
            " background: #42afb2; border-radius: 3px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        header.addWidget(title, 1)
        header.addWidget(goalie_label)

        layout.addLayout(header)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)

        self.adjustSize()

    @classmethod
    def _choose_image(cls):
        """Choose a hit still other than the one used on the previous launch."""
        settings = QSettings("M-Fotheringham", "ORBIT")
        last_index = settings.value(cls.SETTINGS_KEY, -1, type=int)
        available = [
            index
            for index in range(len(LOADING_IMAGES))
            if index != last_index
        ]
        index = secrets.choice(available or [0])
        settings.setValue(cls.SETTINGS_KEY, index)
        return LOADING_IMAGES[index]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f7ffff"))
        painter.drawRoundedRect(self.rect(), 14, 14)

    def start(self):
        available = self.screen().availableGeometry()
        self.move(available.center() - self.rect().center())
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def finish(self):
        self.close()


# Preserve compatibility for any external launcher that imported the old name.
StartupVideoSplash = StartupSplash


__all__ = ["StartupSplash", "StartupVideoSplash"]
