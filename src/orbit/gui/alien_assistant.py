"""A compact Clippy-style assistant for ORBIT's existing status messages."""

from PySide6.QtCore import Qt, QLineF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from orbit.resources import ALIEN_ASSISTANT_IMAGE, asset_path


class AssistantMessageLabel(QLabel):
    """Word-wrapped message text that retains long details in a tooltip."""

    TOOLTIP_LIMIT = 3000

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("background: transparent; color: #173b3f;")
        self.setText(text)

    def setText(self, text):
        text = "" if text is None else str(text)
        super().setText(text)
        tooltip = text
        if len(tooltip) > self.TOOLTIP_LIMIT:
            tooltip = tooltip[: self.TOOLTIP_LIMIT].rstrip() + "\n…"
        self.setToolTip(tooltip)


class SpeechBubble(QWidget):
    """Rounded speech bubble whose left-hand tail points at the assistant."""

    def __init__(self, message="", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumHeight(58)
        self.setMaximumHeight(82)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.message_label = AssistantMessageLabel(message, self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(25, 9, 14, 9)
        layout.addWidget(self.message_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        outline = QColor("#70bfc2")
        fill = QColor("#f5ffff")
        bubble = QRectF(12.0, 1.0, max(self.width() - 13.0, 1.0),
                        max(self.height() - 2.0, 1.0))

        painter.setBrush(fill)
        painter.setPen(QPen(outline, 1.4))
        painter.drawRoundedRect(bubble, 13.0, 13.0)

        centre_y = bubble.center().y()
        tail = QPainterPath()
        tail.moveTo(bubble.left() + 1.0, centre_y - 9.0)
        tail.lineTo(1.5, centre_y)
        tail.lineTo(bubble.left() + 1.0, centre_y + 9.0)
        tail.closeSubpath()
        painter.fillPath(tail, fill)
        painter.setPen(QPen(outline, 1.4))
        painter.drawLine(QLineF(
            bubble.left() + 1.0, centre_y - 9.0, 1.5, centre_y
        ))
        painter.drawLine(QLineF(
            1.5, centre_y, bubble.left() + 1.0, centre_y + 9.0
        ))


class ClickableAlienLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class AlienAssistantWidget(QWidget):
    """Small alien plus one reusable, optionally hideable speech bubble."""

    def __init__(self, message="", parent=None):
        super().__init__(parent)
        self.setMaximumHeight(92)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.alien_label = ClickableAlienLabel(self)
        self.alien_label.setFixedSize(70, 86)
        self.alien_label.setAlignment(Qt.AlignCenter)
        self.alien_label.setCursor(Qt.PointingHandCursor)
        self.alien_label.setToolTip(
            "Click the ORBIT assistant to hide or show guidance."
        )
        self.alien_label.setAccessibleName("ORBIT alien assistant")
        pixmap = QPixmap(str(asset_path(ALIEN_ASSISTANT_IMAGE)))
        self.alien_label.setPixmap(
            pixmap.scaled(
                self.alien_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        self.speech_bubble = SpeechBubble(message, self)
        self.message_label = self.speech_bubble.message_label
        self.alien_label.clicked.connect(self.toggle_bubble)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(3)
        layout.addWidget(self.alien_label, alignment=Qt.AlignBottom)
        layout.addWidget(self.speech_bubble, stretch=1)

    def set_message(self, message, reveal=True):
        self.message_label.setText(message)
        if reveal:
            self.speech_bubble.show()

    def toggle_bubble(self):
        self.speech_bubble.setVisible(not self.speech_bubble.isVisible())


__all__ = ["AlienAssistantWidget"]
