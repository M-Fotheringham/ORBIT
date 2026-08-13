"""Image backends used by ORBIT.

OME-Zarr images are exposed as chunked Dask arrays. TIFF pyramids are also
opened through tifffile's Zarr adapter when available so tiled whole-slide
operations read only the requested field instead of materializing a channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import tifffile as tiff

try:
    from PIL import Image as PillowImage
except ImportError:  # tifffile remains the primary TIFF reader.
    PillowImage = None


TIFF_COMPRESSION_LZW = 5


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


def _imagecodecs_lzw_available() -> bool:
    """Return whether imagecodecs can actually call its LZW decoder.

    Frozen applications can import the top-level ``imagecodecs`` package while
    omitting its compiled codec extension. In that state ``lzw_decode`` is a
    delayed stub that raises only when pixel data are read.
    """
    try:
        import imagecodecs
    except ImportError:
        return False

    codec = getattr(imagecodecs, "LZW", None)
    available = getattr(codec, "available", None)
    if available is not None and not bool(available):
        return False

    try:
        decoder = getattr(imagecodecs, "lzw_decode")
        decoder(b"")
    except Exception as error:
        message = str(error).lower()
        return not (
            isinstance(error, (ImportError, AttributeError))
            or type(error).__name__ == "DelayedImportError"
            or "could not import name" in message
        )
    return True


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
        self._tiff_zarr_levels = []
        self._tiff_zarr_stores = []
        self._use_pillow_tiff_reader = False
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
        try:
            self.series = self._highest_resolution_tiff_series()
            self._validate_tiff_data_bounds()
            self.shape = tuple(int(value) for value in self.series.shape)
            if len(self.shape) != 3:
                raise ValueError(
                    "ORBIT currently expects TIFF images in C-Y-X order; "
                    f"got shape {self.shape}."
                )
            self.dtype = self.series.dtype
            self._axes = ("c", "y", "x")
            self.channel_names = self._get_tiff_channel_names()
            self._use_pillow_tiff_reader = self._needs_pillow_tiff_reader()
            if not self._use_pillow_tiff_reader:
                self._open_tiff_zarr_levels()
        except Exception:
            self.close()
            raise

    def _validate_tiff_data_bounds(self):
        """Reject truncated TIFFs before a native codec reads invalid strips.

        Some native TIFF decoders terminate the process instead of raising a
        Python exception when a strip offset points beyond the end of a
        truncated file. TIFF metadata provide the compressed strip/tile byte
        ranges, so ORBIT can detect that condition safely before displaying or
        segmenting the image.
        """
        for level_index, level in enumerate(self._tiff_levels):
            for page_index, page in enumerate(level.pages):
                offsets = tuple(map(int, getattr(page, "dataoffsets", ())))
                bytecounts = tuple(map(int, getattr(page, "databytecounts", ())))
                if len(offsets) != len(bytecounts):
                    raise ValueError(
                        f"TIFF pixel-data metadata are corrupt in {self.path.name}: "
                        f"page {page_index} has {len(offsets)} offsets but "
                        f"{len(bytecounts)} byte counts. Re-copy or re-export "
                        "the image from its original source."
                    )

                parent = getattr(page, "parent", None)
                filehandle = getattr(parent, "filehandle", None)
                if filehandle is None:
                    filehandle = self.tif.filehandle
                file_size = int(filehandle.size)
                file_path = Path(filehandle.path)
                for segment_index, (offset, bytecount) in enumerate(
                    zip(offsets, bytecounts)
                ):
                    if offset < 0 or bytecount < 0 or offset + bytecount > file_size:
                        required_size = offset + bytecount
                        raise ValueError(
                            f"TIFF pixel data are truncated in {file_path.name}: "
                            f"level {level_index}, page {page_index}, segment "
                            f"{segment_index} ends at byte {required_size:,}, but "
                            f"the file contains only {file_size:,} bytes. Re-copy "
                            "or re-export the complete TIFF before segmenting it."
                        )

    def _needs_pillow_tiff_reader(self) -> bool:
        """Use Pillow only when an LZW TIFF lacks a working imagecodecs decoder."""
        if PillowImage is None:
            return False
        uses_lzw = any(
            int(page.compression) == TIFF_COMPRESSION_LZW
            for level in self._tiff_levels
            for page in level.pages
        )
        return uses_lzw and not _imagecodecs_lzw_available()

    def _pillow_frame_index(self, page) -> int:
        """Map a tifffile page in the selected series to Pillow's frame index."""
        target_offset = getattr(page, "offset", None)
        if target_offset is not None:
            for frame_index, candidate in enumerate(self.tif.pages):
                if getattr(candidate, "offset", None) == target_offset:
                    return frame_index

        page_index = getattr(page, "index", None)
        if isinstance(page_index, int) and 0 <= page_index < len(self.tif.pages):
            return page_index
        raise RuntimeError(
            "Pillow could not map this TIFF pyramid page to a readable frame."
        )

    def _read_tiff_channel_with_pillow(
        self,
        channel: int,
        level: int = 0,
        region: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Decode one TIFF channel with Pillow, including 32-bit float LZW TIFFs."""
        if PillowImage is None:
            raise RuntimeError("Pillow is not installed for TIFF fallback decoding.")
        if not 0 <= int(level) < len(self._tiff_levels):
            raise IndexError(f"TIFF pyramid level {level} is not available.")

        pages = list(self._tiff_levels[int(level)].pages)
        if not pages:
            raise RuntimeError("The selected TIFF level does not contain image pages.")

        sample_index = None
        if len(pages) > int(channel):
            page = pages[int(channel)]
        elif len(pages) == 1:
            page = pages[0]
            sample_index = int(channel)
        else:
            raise IndexError(
                f"TIFF channel {channel} could not be mapped to an image page."
            )

        frame_index = self._pillow_frame_index(page)
        with PillowImage.open(self.path) as image:
            image.seek(frame_index)
            if region is not None:
                y0, x0, height, width = region
                image = image.crop((x0, y0, x0 + width, y0 + height))
            values = np.array(image, copy=True)

        if sample_index is not None and values.ndim == 3:
            if values.shape[-1] > sample_index:
                values = values[..., sample_index]
            elif values.shape[0] > sample_index:
                values = values[sample_index]

        values = np.squeeze(values)
        if values.ndim != 2:
            raise ValueError(
                "Pillow TIFF fallback expected a two-dimensional channel; "
                f"got {values.shape}."
            )
        return values

    def _open_tiff_zarr_levels(self):
        """Expose TIFF pyramid levels as sliceable arrays for FOV-sized I/O."""
        try:
            import zarr
        except ImportError:
            return

        stores = []
        arrays = []
        try:
            for level in self._tiff_levels:
                store = level.aszarr()
                stores.append(store)
                arrays.append(zarr.open(store, mode="r"))
        except Exception:
            for store in stores:
                close = getattr(store, "close", None)
                if callable(close):
                    close()
            return
        self._tiff_zarr_stores = stores
        self._tiff_zarr_levels = arrays

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
        if not 0 <= int(level) < len(self._tiff_levels):
            raise IndexError(f"TIFF pyramid level {level} is not available.")
        if self._use_pillow_tiff_reader:
            return self._read_tiff_channel_with_pillow(channel, level=level)
        try:
            if self._tiff_zarr_levels:
                return self._tiff_zarr_levels[level][channel]
            return self._tiff_levels[level].asarray(key=channel)
        except Exception as original_error:
            try:
                return self._read_tiff_channel_with_pillow(channel, level=level)
            except Exception:
                raise original_error

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
        if self.is_ome_zarr:
            level_shape = self._levels[level].shape
        else:
            if not 0 <= int(level) < len(self._tiff_levels):
                raise IndexError(f"TIFF pyramid level {level} is not available.")
            level_shape = self._tiff_levels[level].shape
        image_height, image_width = map(int, level_shape[-2:])
        if y0 + height > image_height or x0 + width > image_width:
            raise ValueError(
                f"Requested region x={x0}:{x0 + width}, y={y0}:{y0 + height} "
                f"exceeds image dimensions {(image_width, image_height)}."
            )
        if not self.is_ome_zarr and self._use_pillow_tiff_reader:
            return self._read_tiff_channel_with_pillow(
                channel,
                level=level,
                region=(y0, x0, height, width),
            )

        channel_data = self.get_channel(channel, level=level)
        try:
            return _compute(channel_data[y0 : y0 + height, x0 : x0 + width])
        except Exception as original_error:
            if self.is_ome_zarr:
                raise
            try:
                return self._read_tiff_channel_with_pillow(
                    channel,
                    level=level,
                    region=(y0, x0, height, width),
                )
            except Exception:
                raise original_error

    def get_multiscale_channel(self, channel: int = 0) -> list:
        """Return a lazy OME-Zarr pyramid for direct use by Napari."""
        self._validate_channel(channel)
        if self.is_ome_zarr:
            return [level[channel] for level in self._levels]
        if self._use_pillow_tiff_reader:
            return [self._read_tiff_channel_with_pillow(channel, level=0)]
        if self._tiff_zarr_levels:
            return [level[channel] for level in self._tiff_zarr_levels]
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
            if self._use_pillow_tiff_reader:
                overview = self._read_tiff_channel_with_pillow(
                    channel,
                    level=level_index,
                )
            else:
                try:
                    overview = _compute(levels[level_index].asarray(key=channel))
                except Exception as original_error:
                    try:
                        overview = self._read_tiff_channel_with_pillow(
                            channel,
                            level=level_index,
                        )
                    except Exception:
                        raise original_error

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
        for store in self._tiff_zarr_stores:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        self._tiff_zarr_stores = []
        self._tiff_zarr_levels = []
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
