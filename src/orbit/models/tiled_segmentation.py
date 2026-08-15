"""Tiled whole-image segmentation planning and AstroPath-style stitching.

AstroPath acquires neighbouring fields with overlap, but assigns every field a
non-overlapping *primary region* for whole-slide assembly.  ORBIT follows the
same ownership rule here: Cellpose-SAM sees the complete overlapping FOV while
only cells whose centroid falls in that FOV's primary region are retained.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from orbit.fov import DEFAULT_MINIMUM_DAPI_FRACTION, RandomFOVGenerator


DEFAULT_SEGMENTATION_FOV_SIZE = 1024
DEFAULT_SEGMENTATION_FOV_OVERLAP = 0.20


@dataclass(frozen=True, slots=True)
class SegmentationFOV:
    """One overlapping segmentation FOV and its global ownership bounds."""

    index: int
    y0: int
    x0: int
    height: int
    width: int
    primary_y0: int
    primary_x0: int
    primary_y1: int
    primary_x1: int

    @property
    def y1(self) -> int:
        return self.y0 + self.height

    @property
    def x1(self) -> int:
        return self.x0 + self.width

    @property
    def local_primary_bounds(self) -> tuple[int, int, int, int]:
        return (
            self.primary_y0 - self.y0,
            self.primary_x0 - self.x0,
            self.primary_y1 - self.y0,
            self.primary_x1 - self.x0,
        )


def _axis_origins(length: int, fov_size: int, overlap: float) -> list[int]:
    length, fov_size = int(length), int(fov_size)
    if length <= 0 or fov_size <= 0:
        raise ValueError("Image dimensions and segmentation FOV size must be positive.")
    if not 0.0 <= float(overlap) < 1.0:
        raise ValueError("Segmentation FOV overlap must be between 0 and 1.")
    if length <= fov_size:
        return [0]

    stride = max(int(round(fov_size * (1.0 - float(overlap)))), 1)
    final_origin = length - fov_size
    origins = list(range(0, final_origin + 1, stride))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def _primary_intervals(
    origins: list[int],
    fov_size: int,
    length: int,
) -> list[tuple[int, int]]:
    """Partition an axis at the midpoint of each neighbouring overlap."""
    if len(origins) == 1:
        return [(0, int(length))]
    boundaries = [0]
    for left, right in zip(origins, origins[1:]):
        # Midpoint of [right, left + fov_size), the shared overlap.  This is
        # AstroPath's primary-region idea expressed for a regular FOV grid.
        boundaries.append((int(left) + int(fov_size) + int(right)) // 2)
    boundaries.append(int(length))
    return list(zip(boundaries, boundaries[1:]))


def tiled_segmentation_fovs(
    image_shape: tuple[int, int],
    fov_size: int = DEFAULT_SEGMENTATION_FOV_SIZE,
    overlap: float = DEFAULT_SEGMENTATION_FOV_OVERLAP,
) -> list[SegmentationFOV]:
    """Cover an image with FOVs and 20%-overlap primary ownership regions."""
    image_height, image_width = map(int, image_shape)
    y_origins = _axis_origins(image_height, int(fov_size), float(overlap))
    x_origins = _axis_origins(image_width, int(fov_size), float(overlap))
    y_primary = _primary_intervals(y_origins, int(fov_size), image_height)
    x_primary = _primary_intervals(x_origins, int(fov_size), image_width)

    fovs: list[SegmentationFOV] = []
    for y_index, y0 in enumerate(y_origins):
        for x_index, x0 in enumerate(x_origins):
            fovs.append(SegmentationFOV(
                index=len(fovs),
                y0=y0,
                x0=x0,
                height=min(int(fov_size), image_height - y0),
                width=min(int(fov_size), image_width - x0),
                primary_y0=y_primary[y_index][0],
                primary_x0=x_primary[x_index][0],
                primary_y1=y_primary[y_index][1],
                primary_x1=x_primary[x_index][1],
            ))
    return fovs


def select_dapi_positive_fovs(
    image,
    fovs: list[SegmentationFOV],
    dapi_channel: int | None,
    minimum_dapi_fraction: float = DEFAULT_MINIMUM_DAPI_FRACTION,
    progress_callback=None,
) -> list[SegmentationFOV]:
    """Select tiled FOVs with the random-field generator's DAPI criterion."""
    if dapi_channel is None:
        return list(fovs)
    generator = RandomFOVGenerator(image)
    threshold = generator.dapi_threshold(int(dapi_channel))
    selected: list[SegmentationFOV] = []
    for current, fov in enumerate(fovs, start=1):
        dapi = image.get_region(
            channel=int(dapi_channel),
            y0=fov.y0,
            x0=fov.x0,
            height=fov.height,
            width=fov.width,
        )
        positive_fraction = generator.dapi_positive_fraction(dapi, threshold)
        if positive_fraction >= float(minimum_dapi_fraction):
            selected.append(fov)
        if progress_callback is not None:
            progress_callback(current, len(fovs), fov, positive_fraction)
    return selected


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=target.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _close_memmap(array) -> None:
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


class AstroPathMaskStitcher:
    """Stream primary-region-owned FOV labels into a disk-backed slide mask."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int],
        cell_path: str | Path,
        mask_path: str | Path,
        model_name: str,
        marker_names: list[str],
        nuclear_channel_name: str | None,
    ):
        self.image_shape = tuple(map(int, image_shape))
        self.cell_path = Path(cell_path)
        self.mask_path = Path(mask_path)
        self.cell_path.parent.mkdir(parents=True, exist_ok=True)
        self.mask_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_cell_path = _temporary_path(self.cell_path)
        self.temporary_mask_path = _temporary_path(self.mask_path)
        # tifffile.memmap creates the TIFF itself, so remove mkstemp's empty file.
        self.temporary_mask_path.unlink(missing_ok=True)
        self.mask = tifffile.memmap(
            self.temporary_mask_path,
            shape=self.image_shape,
            dtype=np.uint32,
            bigtiff=True,
            metadata={
                "axes": "YX",
                "ORBIT segmentation model": str(model_name),
                "ORBIT membrane markers": list(marker_names),
                "ORBIT nuclear marker": nuclear_channel_name or "",
                "ORBIT stitching": "AstroPath-style primary regions",
            },
        )
        self.mask[:] = 0
        self.next_cell_id = 1
        self.cell_count = 0
        self._wrote_header = False
        self._closed = False

    def add_fov(
        self,
        fov: SegmentationFOV,
        local_masks: np.ndarray,
        cells: pd.DataFrame,
        pixel_size_um: float,
    ) -> int:
        """Retain centroid-owned cells and write their labels into the slide."""
        local_masks = np.asarray(local_masks, dtype=np.uint32)
        if local_masks.shape != (fov.height, fov.width):
            raise ValueError(
                f"FOV mask shape {local_masks.shape} does not match "
                f"{(fov.height, fov.width)}."
            )
        if cells.empty:
            return 0

        global_x = pd.to_numeric(cells["Centroid X px"], errors="raise") + fov.x0
        global_y = pd.to_numeric(cells["Centroid Y px"], errors="raise") + fov.y0
        owned = (
            (global_x >= fov.primary_x0)
            & (global_x < fov.primary_x1)
            & (global_y >= fov.primary_y0)
            & (global_y < fov.primary_y1)
        )
        retained = cells.loc[owned].copy()
        if retained.empty:
            return 0

        old_ids = retained["Cell ID"].to_numpy(dtype=np.int64)
        new_ids = np.arange(
            self.next_cell_id,
            self.next_cell_id + len(retained),
            dtype=np.uint32,
        )
        maximum_local_label = int(local_masks.max(initial=0))
        lookup = np.zeros(maximum_local_label + 1, dtype=np.uint32)
        lookup[old_ids] = new_ids
        relabelled = lookup[local_masks]

        target = self.mask[fov.y0:fov.y1, fov.x0:fov.x1]
        insert = (relabelled > 0) & (target == 0)
        target[insert] = relabelled[insert]

        retained["Cell ID"] = new_ids.astype(np.int64)
        inserted_ids = np.unique(relabelled[insert])
        inserted_ids = inserted_ids[inserted_ids > 0]
        retained = retained.loc[
            retained["Cell ID"].isin(inserted_ids.astype(np.int64))
        ].copy()
        if retained.empty:
            self.next_cell_id += len(new_ids)
            return 0
        for column in (
            "Centroid X px",
            "Bounding box X min px",
            "Bounding box X max px",
        ):
            if column in retained:
                retained[column] = retained[column] + fov.x0
        for column in (
            "Centroid Y px",
            "Bounding box Y min px",
            "Bounding box Y max px",
        ):
            if column in retained:
                retained[column] = retained[column] + fov.y0
        retained["Centroid X µm"] = (
            retained["Centroid X px"] * float(pixel_size_um)
        )
        retained["Centroid Y µm"] = (
            retained["Centroid Y px"] * float(pixel_size_um)
        )
        retained.to_csv(
            self.temporary_cell_path,
            sep="\t",
            index=False,
            mode="a",
            header=not self._wrote_header,
        )
        self._wrote_header = True
        # Advance across all allocated IDs, including a rare label completely
        # hidden by an already-stitched neighbour, to keep IDs collision-free.
        self.next_cell_id += len(new_ids)
        self.cell_count += len(retained)
        return len(retained)

    def finalize(self) -> tuple[Path, Path, int]:
        if self.cell_count == 0:
            raise ValueError(
                "Cellpose-SAM did not identify any cells in the selected FOVs."
            )
        _close_memmap(self.mask)
        self._closed = True
        os.replace(self.temporary_cell_path, self.cell_path)
        os.replace(self.temporary_mask_path, self.mask_path)
        return self.cell_path, self.mask_path, self.cell_count

    def abort(self) -> None:
        if not self._closed:
            try:
                _close_memmap(self.mask)
            except Exception:
                pass
            self._closed = True
        self.temporary_cell_path.unlink(missing_ok=True)
        self.temporary_mask_path.unlink(missing_ok=True)


__all__ = [
    "AstroPathMaskStitcher",
    "DEFAULT_SEGMENTATION_FOV_OVERLAP",
    "DEFAULT_SEGMENTATION_FOV_SIZE",
    "SegmentationFOV",
    "select_dapi_positive_fovs",
    "tiled_segmentation_fovs",
]
