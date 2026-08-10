"""Paths and application-level setup for ORBIT's bundled visual assets."""

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"
COSMIC_ALIEN_FONT = "ca.ttf"
ALIEN_ASSISTANT_IMAGE = "burt_transparent.png"


def asset_path(filename: str) -> Path:
    """Return a validated path to an asset shipped inside the orbit package."""
    path = (ASSET_DIRECTORY / filename).resolve()
    if path.parent != ASSET_DIRECTORY.resolve() or not path.is_file():
        raise FileNotFoundError(f"Bundled ORBIT asset not found: {filename}")
    return path


def install_cosmic_alien_font(application) -> str | None:
    """Install Cosmic Alien privately and make it ORBIT's application font."""
    try:
        font_id = QFontDatabase.addApplicationFont(
            str(asset_path(COSMIC_ALIEN_FONT))
        )
    except (OSError, RuntimeError):
        return None

    if font_id < 0:
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        return None

    family = families[0]
    point_size = application.font().pointSize()
    application.setFont(QFont(family, max(point_size, 10)))
    return family
