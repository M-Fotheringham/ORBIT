"""Public ORBIT API with lazy imports for fast application startup."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from orbit.fov import RandomFOVGenerator
    from orbit.image import OrbitImage, QPTiffImage


__all__ = [
    "OrbitImage",
    "QPTiffImage",
    "RandomFOVGenerator",
]


def __getattr__(name):
    if name in {"OrbitImage", "QPTiffImage"}:
        from orbit.image import OrbitImage, QPTiffImage

        globals().update(
            OrbitImage=OrbitImage,
            QPTiffImage=QPTiffImage,
        )
        return globals()[name]

    if name == "RandomFOVGenerator":
        from orbit.fov import RandomFOVGenerator

        globals()[name] = RandomFOVGenerator
        return RandomFOVGenerator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
