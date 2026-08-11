"""Instant animated loading popup shown while ORBIT imports its viewer."""

import secrets

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QColor, QMovie, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from orbit.resources import LOADING_ANIMATIONS, asset_path


class StartupVideoSplash(QDialog):
    """Compact startup window using a seek-free randomized animation."""

    PLAYER_WIDTH = 480
    PLAYER_HEIGHT = 270
    SETTINGS_KEY = "startup/last_loading_animation"

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
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #173b3f; font-size: 15px; padding: 2px 0 1px 0;"
        )

        self.animation_label = QLabel(self)
        self.animation_label.setFixedSize(
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )
        self.animation_label.setAlignment(Qt.AlignCenter)
        self.animation_label.setStyleSheet(
            "background: #071013; border: 1px solid #70bfc2;"
        )

        animation_name = self._choose_animation()
        self.movie = QMovie(str(asset_path(animation_name)), parent=self)
        self.movie.setCacheMode(QMovie.CacheMode.CacheNone)
        self.movie.setScaledSize(
            QSize(self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        )
        self.animation_label.setMovie(self.movie)

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
        layout.addWidget(title)
        layout.addWidget(self.animation_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)

        self.adjustSize()

    @classmethod
    def _choose_animation(cls):
        """Choose a clip other than the one used on the previous launch."""
        settings = QSettings("M-Fotheringham", "ORBIT")
        last_index = settings.value(cls.SETTINGS_KEY, -1, type=int)
        available = [
            index
            for index in range(len(LOADING_ANIMATIONS))
            if index != last_index
        ]
        index = secrets.choice(available or [0])
        settings.setValue(cls.SETTINGS_KEY, index)
        return LOADING_ANIMATIONS[index]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f7ffff"))
        painter.drawRoundedRect(self.rect(), 14, 14)

    def start(self):
        available = self.screen().availableGeometry()
        self.move(available.center() - self.rect().center())
        # Decode and display the first frame before any heavier imports start.
        self.movie.jumpToFrame(0)
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()
        self.movie.start()
        QApplication.processEvents()

    def finish(self):
        self.movie.stop()
        self.close()


__all__ = ["StartupVideoSplash"]
