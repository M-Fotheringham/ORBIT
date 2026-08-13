"""Coordinate-column selection and conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coordinates_are_microns(column) -> bool:
    """Return whether a coordinate column is expressed in micrometres."""
    name = str(column).lower()
    return any(unit in name for unit in ("µm", "μm", " um", "micron"))


def _coordinate_unit_rank(column) -> int:
    """Prefer source pixels, then unitless values, then physical units."""
    name = str(column).lower().replace("_", " ").replace("-", " ")
    if any(unit in name for unit in (" px", "pixel")):
        return 0
    if coordinates_are_microns(column):
        return 2
    return 1


def centroid_columns(cell_data: pd.DataFrame) -> tuple[object, object]:
    """Return X/Y centroid columns, preferring pixel coordinates.

    ORBIT-generated tables contain both pixel and micrometre centroids. Pixel
    values are the mask's native coordinate system and therefore avoid a lossy
    or incorrect physical-scale round trip for OME-Zarr images.
    """

    def find_axis(axis):
        exact = [
            f"Centroid {axis.upper()} px",
            f"Centroid {axis.upper()}",
            f"Centroid {axis.upper()} µm",
            f"Centroid {axis.upper()} μm",
            f"Centroid {axis.upper()} um",
        ]
        for candidate in exact:
            if candidate in cell_data.columns:
                return candidate

        matches = []
        for column_index, column in enumerate(cell_data.columns):
            normalized = str(column).lower().replace("_", " ").replace("-", " ")
            has_centroid = "centroid" in normalized or "center" in normalized
            has_axis = (
                f" {axis} " in f" {normalized} " or normalized.endswith(axis)
            )
            if has_centroid and has_axis:
                matches.append((_coordinate_unit_rank(column), column_index, column))
        return min(matches)[2] if matches else None

    x_column, y_column = find_axis("x"), find_axis("y")
    if x_column is None or y_column is None:
        raise ValueError(
            "Cell data must contain X and Y centroid columns to display "
            "model predictions."
        )
    return x_column, y_column


def centroid_pixel_coordinates(
    cell_data: pd.DataFrame,
    pixel_size_um: float | None = None,
) -> dict:
    """Return centroid arrays in the segmentation mask's pixel coordinates."""
    x_column, y_column = centroid_columns(cell_data)
    x = pd.to_numeric(cell_data[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(cell_data[y_column], errors="coerce").to_numpy(dtype=float)

    micron_axes = (
        coordinates_are_microns(x_column),
        coordinates_are_microns(y_column),
    )
    if any(micron_axes):
        try:
            scale = float(pixel_size_um)
        except (TypeError, ValueError):
            scale = np.nan
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Cell centroids are stored in micrometres, but the image does "
                "not provide a valid physical pixel size."
            )
        if micron_axes[0]:
            x = x / scale
        if micron_axes[1]:
            y = y / scale

    return {
        "x": x,
        "y": y,
        "x_column": x_column,
        "y_column": y_column,
    }


__all__ = [
    "centroid_columns",
    "centroid_pixel_coordinates",
    "coordinates_are_microns",
]
