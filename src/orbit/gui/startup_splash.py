"""Video loading popup displayed while ORBIT imports its main viewer."""

import secrets

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from orbit.resources import LOADING_VIDEO, asset_path


class StartupVideoSplash(QDialog):
    """Compact, frameless startup window with randomized video playback."""

    PLAYER_WIDTH = 600
    PLAYER_HEIGHT = 338
    END_MARGIN_MS = 5_000

    def __init__(self, icon=None, parent=None):
        super().__init__(parent)
        self._random_seek_complete = False

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

        self.video_widget = QVideoWidget(self)
        self.video_widget.setFixedSize(
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )
        self.video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
        self.video_widget.setStyleSheet(
            "background: #071013; border: 1px solid #70bfc2;"
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
        layout.addWidget(title)
        layout.addWidget(self.video_widget, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)

        self.audio_output = QAudioOutput(self)
        self.audio_output.setMuted(True)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.player.setLoops(QMediaPlayer.Loops.Infinite)
        self.player.durationChanged.connect(self._seek_to_random_start)
        self.player.setSource(QUrl.fromLocalFile(str(asset_path(LOADING_VIDEO))))

        self.adjustSize()

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
        self.player.play()

    def finish(self):
        self.player.stop()
        self.close()

    def _seek_to_random_start(self, duration_ms):
        if self._random_seek_complete or duration_ms <= 0:
            return

        latest_start = max(duration_ms - self.END_MARGIN_MS, 0)
        position_ms = (
            secrets.randbelow(latest_start + 1) if latest_start else 0
        )
        self._random_seek_complete = True
        self.player.setPosition(position_ms)
        self.player.play()


__all__ = ["StartupVideoSplash"]
