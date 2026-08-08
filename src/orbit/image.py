"""Image backends used by ORBIT.

TIFF images retain the original eager behaviour. OME-Zarr images are exposed as
chunked Dask arrays so a field of view reads only the chunks that intersect the
requested region.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tiff


def _axis_name(axis: Any) -> str:
    """Return a normalized NGFF axis name."""
    if isinstance(axis, str):
        return axis.strip().lower()
    if isinstance(axis, dict):
        return str(axis.get("name", "")).strip().lower()
    return str(getattr(axis, "name", axis)).strip().lower()


def _compute(array):
    """Materialize a NumPy-like or Dask array without importing Dask directly."""
    compute = getattr(array, "compute", None)
    if callable(compute):
        array = compute()
    return np.asarray(array)


class QPTiffImage:
    """Unified TIFF and OME-Zarr image reader used by existing ORBIT code.

    The historical class name is retained to avoid breaking saved projects and
    downstream imports. ``get_shape`` always reports ``(C, Y, X)``. For OME-
    Zarr data, non-spatial axes such as T and Z are currently fixed at index 0.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Could not find image: {self.path}")

        self.tif = None
        self.series = None
        self._tiff_levels = []
        self._levels = []
        self._axes: tuple[str, ...] = ()
        self._metadata: dict[str, Any] = {}
        self.pixel_size_um: float | None = None

        if self._looks_like_ome_zarr(self.path):
            self.extension = ".ome.zarr"
            self.format_name = "OME-Zarr"
            self.is_ome_zarr = True
            self._open_ome_zarr()
        elif self.path.is_file():
            self.extension = self.path.suffix.lower()
            self.format_name = "TIFF"
            self.is_ome_zarr = False
            self._open_tiff()
        else:
            raise ValueError(
                f"Unsupported image path: {self.path}. Select a TIFF file or "
                "an OME-Zarr directory."
            )

    @staticmethod
    def _looks_like_ome_zarr(path: Path) -> bool:
        if not path.is_dir():
            return False
        lowered = path.name.lower()
        return (
            lowered.endswith(".zarr")
            or (path / ".zattrs").is_file()
            or (path / "zarr.json").is_file()
        )

    def _open_tiff(self):
        self.tif = tiff.TiffFile(self.path)
        self.series = self._highest_resolution_tiff_series()
        self.shape = tuple(int(value) for value in self.series.shape)
        if len(self.shape) != 3:
            raise ValueError(
                "ORBIT currently expects TIFF images in C-Y-X order; "
                f"got shape {self.shape}."
            )
        self.dtype = self.series.dtype
        self._axes = ("c", "y", "x")
        self.channel_names = self._get_tiff_channel_names()

    def _highest_resolution_tiff_series(self):
        """Use the full-resolution level of TIFF series 0.

        The original ORBIT reader uses ``tif.series[0]``. Keeping that series
        is important for QPTIFF because other top-level series may be overview,
        macro, or auxiliary images. Within the primary series, choose the
        largest pyramid level explicitly so level 0/source pixels are always
        used even if a TIFF reader exposes the levels in an unusual order.
        """
        if not self.tif.series:
            raise ValueError("The selected TIFF does not contain an image series.")

        primary = self.tif.series[0]
        levels = list(getattr(primary, "levels", ()) or ()) or [primary]
        compatible = [level for level in levels if len(level.shape) == 3]
        if not compatible:
            raise ValueError(
                "ORBIT expected TIFF series 0 to be a three-dimensional C-Y-X "
                f"image; got shape {primary.shape}."
            )
        self._tiff_levels = sorted(
            compatible,
            key=lambda level: int(level.shape[-2]) * int(level.shape[-1]),
            reverse=True,
        )
        return self._tiff_levels[0]

    def _open_ome_zarr(self):
        try:
            from ome_zarr.io import parse_url
            from ome_zarr.reader import Reader
        except ImportError as error:
            raise RuntimeError(
                "OME-Zarr support is not installed. Run 'uv sync' from the "
                "ORBIT repository and try again."
            ) from error

        location = parse_url(str(self.path))
        if location is None:
            raise ValueError(f"Could not open OME-Zarr store: {self.path}")

        nodes = list(Reader(location)())
        image_node = next(
            (
                node
                for node in nodes
                if getattr(node, "data", None)
                and isinstance(getattr(node, "metadata", None), dict)
                and node.metadata.get("axes")
            ),
            None,
        )
        if image_node is None:
            image_node = next(
                (node for node in nodes if getattr(node, "data", None)),
                None,
            )
        if image_node is None:
            raise ValueError(
                f"No multiscale image was found in OME-Zarr store: {self.path}"
            )

        self._metadata = dict(getattr(image_node, "metadata", {}) or {})
        raw_levels = list(image_node.data)
        if not raw_levels:
            raise ValueError(f"OME-Zarr image contains no resolution levels: {self.path}")

        axes_metadata = self._metadata.get("axes")
        if axes_metadata is None:
            axes_metadata = self._infer_axes(raw_levels[0].ndim)
        self._axes = tuple(_axis_name(axis) for axis in axes_metadata)
        if len(self._axes) != raw_levels[0].ndim:
            raise ValueError(
                "OME-Zarr axes do not match the image dimensions "
                f"({self._axes} versus {raw_levels[0].shape})."
            )
        if "y" not in self._axes or "x" not in self._axes:
            raise ValueError(
                f"OME-Zarr image must contain Y and X axes; got {self._axes}."
            )

        self._levels = [self._as_cyx(level) for level in raw_levels]
        self.shape = tuple(int(value) for value in self._levels[0].shape)
        self.dtype = self._levels[0].dtype
        self.channel_names = self._get_ome_zarr_channel_names()
        self.pixel_size_um = self._get_ome_zarr_pixel_size_um()

    @staticmethod
    def _infer_axes(ndim: int) -> tuple[str, ...]:
        inferred = {
            2: ("y", "x"),
            3: ("c", "y", "x"),
            4: ("z", "c", "y", "x"),
            5: ("t", "c", "z", "y", "x"),
        }.get(ndim)
        if inferred is None:
            raise ValueError(
                "OME-Zarr axes metadata are required for arrays with "
                f"{ndim} dimensions."
            )
        return inferred

    def _as_cyx(self, array):
        """Select T/Z/etc. at zero and reorder a lazy array to C-Y-X."""
        selectors = []
        remaining_axes = []
        for axis in self._axes:
            if axis in {"c", "y", "x"}:
                selectors.append(slice(None))
                remaining_axes.append(axis)
            else:
                selectors.append(0)
        selected = array[tuple(selectors)]
        if "c" not in remaining_axes:
            selected = selected[None, ...]
            remaining_axes.insert(0, "c")
        order = tuple(remaining_axes.index(axis) for axis in ("c", "y", "x"))
        if order != tuple(range(3)):
            selected = selected.transpose(order)
        return selected

    def get_shape(self):
        return self.shape

    def get_channel_names(self):
        return list(self.channel_names)

    def get_channel(self, channel: int = 0, level: int = 0):
        """Return one 2D channel.

        OME-Zarr returns a lazy Dask slice. Existing whole-image algorithms can
        call ``np.asarray`` when they intentionally need all pixels.
        """
        self._validate_channel(channel)
        if self.is_ome_zarr:
            return self._levels[level][channel]
        return self.series.asarray(key=channel)

    def get_region(
        self,
        channel: int,
        y0: int,
        x0: int,
        height: int,
        width: int,
        level: int = 0,
    ) -> np.ndarray:
        """Read one channel region, computing only intersecting Zarr chunks."""
        self._validate_channel(channel)
        y0, x0 = int(y0), int(x0)
        height, width = int(height), int(width)
        if min(y0, x0, height, width) < 0 or height == 0 or width == 0:
            raise ValueError("Image-region coordinates and dimensions must be positive.")
        channel_data = self.get_channel(channel, level=level)
        if y0 + height > channel_data.shape[0] or x0 + width > channel_data.shape[1]:
            raise ValueError(
                f"Requested region x={x0}:{x0 + width}, y={y0}:{y0 + height} "
                f"exceeds image dimensions {channel_data.shape[::-1]}."
            )
        return _compute(channel_data[y0 : y0 + height, x0 : x0 + width])

    def get_multiscale_channel(self, channel: int = 0) -> list:
        """Return a lazy OME-Zarr pyramid for direct use by Napari."""
        self._validate_channel(channel)
        if self.is_ome_zarr:
            return [level[channel] for level in self._levels]
        return [self.get_channel(channel)]

    def get_overview(self, channel: int = 0, max_size: int = 512) -> np.ndarray:
        """Return a low-power channel view without changing full-resolution data.

        The smallest pyramid level that still meets ``max_size`` is preferred,
        minimizing I/O while retaining enough pixels for a crisp navigator.
        Images without a pyramid are sampled after reading their source level.
        """
        self._validate_channel(channel)
        max_size = int(max_size)
        if max_size <= 0:
            raise ValueError("max_size must be positive.")

        if self.is_ome_zarr:
            level_index = self._overview_level_index(self._levels, max_size)
            overview = _compute(self._levels[level_index][channel])
        else:
            levels = self._tiff_levels or [self.series]
            level_index = self._overview_level_index(levels, max_size)
            overview = _compute(levels[level_index].asarray(key=channel))

        overview = np.squeeze(overview)
        if overview.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional overview; got {overview.shape}."
            )
        height, width = overview.shape
        scale = max(height / max_size, width / max_size, 1.0)
        if scale <= 1.0:
            return overview
        output_height = max(int(round(height / scale)), 1)
        output_width = max(int(round(width / scale)), 1)
        rows = np.linspace(0, height - 1, output_height).astype(np.int64)
        columns = np.linspace(0, width - 1, output_width).astype(np.int64)
        return overview[np.ix_(rows, columns)]

    @staticmethod
    def _overview_level_index(levels, max_size: int) -> int:
        dimensions = [
            max(int(level.shape[-2]), int(level.shape[-1]))
            for level in levels
        ]
        large_enough = [
            index
            for index, dimension in enumerate(dimensions)
            if dimension >= max_size
        ]
        if large_enough:
            return min(large_enough, key=lambda index: dimensions[index])
        return max(range(len(levels)), key=lambda index: dimensions[index])

    def get_dapi_channel_index(self, default: int = 0) -> int:
        for index, name in enumerate(self.channel_names):
            if "dapi" in str(name).strip().lower():
                return index
        return min(max(int(default), 0), self.shape[0] - 1)

    def get_pixel_size_um(self, default: float | None = None) -> float | None:
        return self.pixel_size_um if self.pixel_size_um is not None else default

    def close(self):
        if self.tif is not None:
            self.tif.close()

    def _validate_channel(self, channel: int):
        if not 0 <= int(channel) < self.shape[0]:
            raise IndexError(
                f"Channel {channel} is outside the valid range 0..{self.shape[0] - 1}."
            )

    def _get_ome_zarr_channel_names(self) -> list[str]:
        channel_count = self.shape[0]
        reader_names = self._metadata.get("channel_names")
        if isinstance(reader_names, Sequence) and not isinstance(
            reader_names, (str, bytes)
        ):
            names = [
                str(name or f"Channel {index}")
                for index, name in enumerate(reader_names)
            ]
            if len(names) >= channel_count:
                return names[:channel_count]
        metadata_candidates = [self._metadata]
        nested = self._metadata.get("metadata")
        if isinstance(nested, dict):
            metadata_candidates.append(nested)
        for metadata in metadata_candidates:
            omero = metadata.get("omero")
            if not isinstance(omero, dict):
                continue
            channels = omero.get("channels")
            if not isinstance(channels, Sequence):
                continue
            names = []
            for index, channel in enumerate(channels):
                if isinstance(channel, dict):
                    name = channel.get("label") or channel.get("name")
                else:
                    name = getattr(channel, "label", None)
                names.append(str(name or f"Channel {index}"))
            if len(names) >= channel_count:
                return names[:channel_count]
        return [f"Channel {index}" for index in range(channel_count)]

    def _get_ome_zarr_pixel_size_um(self) -> float | None:
        transformations = self._metadata.get("coordinateTransformations")
        if not transformations:
            return None
        first_level = transformations[0] if isinstance(transformations, list) else None
        if not isinstance(first_level, list):
            return None
        scale = next(
            (
                transform.get("scale")
                for transform in first_level
                if isinstance(transform, dict) and transform.get("type") == "scale"
            ),
            None,
        )
        if not isinstance(scale, Sequence) or len(scale) != len(self._axes):
            return None
        y_scale = float(scale[self._axes.index("y")])
        axes_metadata = self._metadata.get("axes") or []
        y_axis = axes_metadata[self._axes.index("y")] if axes_metadata else None
        unit = y_axis.get("unit") if isinstance(y_axis, dict) else None
        conversions = {
            "micrometer": 1.0,
            "micrometre": 1.0,
            "µm": 1.0,
            "um": 1.0,
            "nanometer": 0.001,
            "nanometre": 0.001,
            "nm": 0.001,
            "millimeter": 1000.0,
            "millimetre": 1000.0,
            "mm": 1000.0,
        }
        return y_scale * conversions.get(str(unit).lower(), 1.0)

    def _get_tiff_channel_names(self):
        n_channels = self.shape[0]

        if self.extension == ".qptiff":
            try:
                from qptifffile import QPTiffFile

                qptiff = QPTiffFile(self.path)
                names = list(qptiff.get_biomarkers())
                if len(names) >= n_channels:
                    return names[:n_channels]
            except Exception:
                pass

        try:
            ome_xml = self.tif.ome_metadata
            if ome_xml is not None:
                root = ET.fromstring(ome_xml)
                namespaces = {
                    "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                }
                channels = root.findall(".//ome:Channel", namespaces)
                names = [
                    channel.attrib.get("Name", f"Channel {index}")
                    for index, channel in enumerate(channels)
                ]
                if names:
                    return names[:n_channels]
        except Exception:
            pass

        try:
            metadata = self.tif.imagej_metadata
            if metadata is not None and metadata.get("Labels") is not None:
                names = list(metadata["Labels"])
                if names:
                    return names[:n_channels]
        except Exception:
            pass

        try:
            names = []
            for index, page in enumerate(self.tif.pages[:n_channels]):
                description = str(page.description)
                if "Name=" in description:
                    name = description.split("Name=")[1].split("\n")[0].strip()
                elif "ChannelName=" in description:
                    name = (
                        description.split("ChannelName=")[1]
                        .split("\n")[0]
                        .strip()
                    )
                else:
                    name = f"Channel {index}"
                names.append(name)
            if names:
                return names
        except Exception:
            pass

        return [f"Channel {index}" for index in range(n_channels)]


# Clearer name for new code while preserving the public historical import.
OrbitImage = QPTiffImage


__all__ = ["OrbitImage", "QPTiffImage"]
