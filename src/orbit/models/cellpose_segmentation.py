"""Cellpose-SAM segmentation and cell-level measurement generation."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from orbit.fov import DEFAULT_MINIMUM_DAPI_FRACTION
from orbit.models.tiled_segmentation import (
    AstroPathMaskStitcher,
    DEFAULT_SEGMENTATION_FOV_OVERLAP,
    DEFAULT_SEGMENTATION_FOV_SIZE,
    SegmentationFOV,
    select_dapi_positive_fovs,
    tiled_segmentation_fovs,
)
from orbit.threshold import compartment_mask_for_rows

CELLPOSE_SAM_MODEL = "cpsam_v2"
CELLPOSE_SAM_MODEL_SIZE_BYTES = 1_233_586_851
CELLPOSE_MODEL_DIRECTORY = "cellpose_models"
ORBIT_CELLPOSE_MODEL_ENV = "ORBIT_CPSAM_V2_PATH"
DEFAULT_PIXEL_SIZE_UM = 0.5064
DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM = 1.0


def bundled_cellpose_sam_model_path() -> Path | None:
    """Return ORBIT's installed or explicitly supplied model, when available.

    The Windows installer downloads and verifies the model into the application
    directory. Standalone builds therefore never need to write to Cellpose's
    per-user cache or download from a segmentation worker. An explicit
    ``ORBIT_CPSAM_V2_PATH`` override and the legacy source-tree staging path are
    retained for development and backwards compatibility.
    """
    override = os.environ.get(ORBIT_CELLPOSE_MODEL_ENV)
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())

    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        candidates.append(
            Path(pyinstaller_root) / CELLPOSE_MODEL_DIRECTORY / CELLPOSE_SAM_MODEL
        )
    candidates.append(
        Path(sys.executable).resolve().parent
        / CELLPOSE_MODEL_DIRECTORY
        / CELLPOSE_SAM_MODEL
    )
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "build"
        / CELLPOSE_MODEL_DIRECTORY
        / CELLPOSE_SAM_MODEL
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if (
                resolved.is_file()
                and resolved.stat().st_size == CELLPOSE_SAM_MODEL_SIZE_BYTES
            ):
                return resolved
        except OSError:
            continue
    return None


@contextmanager
def _writable_cellpose_streams():
    """Supply writable streams when a windowed executable has no console.

    PyInstaller sets ``sys.stdout`` and ``sys.stderr`` to ``None`` for ORBIT's
    ``console=False`` build. Cellpose uses tqdm while downloading a model that
    is not yet cached, and tqdm requires a stream with ``write`` and ``flush``.
    Source and console builds keep their original streams unchanged.
    """
    replacements = []
    try:
        for stream_name in ("stdout", "stderr"):
            original = getattr(sys, stream_name, None)
            if (
                callable(getattr(original, "write", None))
                and callable(getattr(original, "flush", None))
            ):
                continue
            replacement = open(os.devnull, "w", encoding="utf-8")
            setattr(sys, stream_name, replacement)
            replacements.append((stream_name, original, replacement))
        yield
    finally:
        for stream_name, original, replacement in reversed(replacements):
            setattr(sys, stream_name, original)
            replacement.close()


def is_dapi_channel(channel_name: str) -> bool:
    """Return whether a channel name identifies DAPI."""
    return "dapi" in str(channel_name).strip().lower()


def membrane_marker_names(
    channel_names: Iterable[str],
    nuclear_channel_name: str | None = None,
) -> list[str]:
    """Return channels that can be selected as membrane guides.

    DAPI is never offered as a membrane marker. A manually selected nuclear
    channel is also excluded, which supports TIFFs whose channels have generic
    labels such as ``Channel 0``.
    """
    nuclear_name = (
        None if nuclear_channel_name is None else str(nuclear_channel_name)
    )
    return [
        str(name)
        for name in channel_names
        if not is_dapi_channel(name) and str(name) != nuclear_name
    ]


def dapi_channel_name(channel_names: Iterable[str]) -> str | None:
    """Return the first DAPI channel, if one is present."""
    return next(
        (str(name) for name in channel_names if is_dapi_channel(name)),
        None,
    )


def resolve_nuclear_channel_name(
    channel_names: Iterable[str],
    nuclear_channel_name: str | None = None,
) -> str | None:
    """Resolve an explicit nuclear channel, defaulting to DAPI when present."""
    names = [str(name) for name in channel_names]
    if nuclear_channel_name is None:
        return dapi_channel_name(names)
    requested = str(nuclear_channel_name)
    if requested not in names:
        raise ValueError(
            f"Selected nuclear channel '{requested}' is not present. Available "
            "channels: " + ", ".join(names)
        )
    return requested


def cuda_compatible_gpu_available() -> bool:
    """Return whether PyTorch can access an NVIDIA CUDA GPU."""
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return False
    return bool(
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and torch.version.cuda is not None
    )


def output_paths_for_image(image_path: str | Path) -> tuple[Path, Path]:
    """Return deterministic cell-data and mask paths beside an image."""
    image_path = Path(image_path).expanduser().resolve()
    return segmentation_export_paths(image_path.parent, image_path)


def segmentation_export_paths(
    destination_directory: str | Path,
    image_path: str | Path,
    filename_stem: str | None = None,
) -> tuple[Path, Path]:
    """Return cell-data and mask paths in a selected export directory."""
    destination = Path(destination_directory).expanduser().resolve()
    base_name = Path(image_path).stem if filename_stem is None else str(filename_stem)
    base_name = Path(base_name).name.strip()
    if not base_name:
        raise ValueError("The segmentation export filename cannot be empty.")
    return (
        destination / f"{base_name}_orbit_cellpose_cells.tsv",
        destination / f"{base_name}_orbit_cellpose_masks.tif",
    )


def _normalized_channel(channel: np.ndarray) -> np.ndarray:
    """Robustly normalize one fluorescence channel to floating-point 0..1."""
    values = np.asarray(channel)
    if values.ndim != 2:
        raise ValueError(
            f"Cellpose segmentation requires 2D channels; got {values.shape}."
        )
    sample_stride = max(
        int(np.ceil(np.sqrt(values.size / 1_000_000))),
        1,
    )
    sample = values[::sample_stride, ::sample_stride]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = np.array(values, dtype=np.float32, copy=True)
    np.nan_to_num(
        normalized,
        copy=False,
        nan=low,
        posinf=high,
        neginf=low,
    )
    np.clip(normalized, low, high, out=normalized)
    normalized -= np.float32(low)
    normalized /= np.float32(high - low)
    return normalized


def build_cellpose_input(
    image,
    selected_marker_names: Iterable[str],
    nuclear_channel_name: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Build a Y-X-C Cellpose input from membrane and nuclear channels."""
    channel_names = [str(name) for name in image.get_channel_names()]
    channel_indices = {name: index for index, name in enumerate(channel_names)}
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    nuclear_name = resolve_nuclear_channel_name(
        channel_names,
        nuclear_channel_name,
    )
    if not selected:
        raise ValueError("Select at least one membrane marker before segmenting.")
    if nuclear_name is not None and nuclear_name in selected:
        raise ValueError(
            f"'{nuclear_name}' is selected as the nuclear channel and cannot "
            "also be used as a membrane marker."
        )
    if any(is_dapi_channel(name) for name in selected):
        raise ValueError(
            "DAPI cannot be selected as a membrane marker. Choose it as the "
            "nuclear channel instead."
        )

    missing = [name for name in selected if name not in channel_indices]
    if missing:
        raise ValueError(
            f"{Path(image.path).name} does not contain selected marker(s): "
            + ", ".join(missing)
        )

    merged = None
    for name in selected:
        normalized = _normalized_channel(
            image.get_channel(channel_indices[name])
        )
        if merged is None:
            merged = normalized
        elif normalized.shape != merged.shape:
            raise ValueError(
                f"Channel '{name}' has dimensions {normalized.shape}, expected "
                f"{merged.shape}."
            )
        else:
            merged += normalized
    merged /= np.float32(len(selected))

    # Keeping the assembled input as uint8 substantially reduces whole-image
    # memory. Cellpose converts it to float32 as part of its own normalization.
    model_input = np.zeros((*merged.shape, 3), dtype=np.uint8)
    merged *= np.float32(255.0)
    model_input[..., 0] = merged

    if nuclear_name is not None:
        nuclear = _normalized_channel(
            image.get_channel(channel_indices[nuclear_name])
        )
        if nuclear.shape != merged.shape:
            raise ValueError(
                f"Nuclear channel '{nuclear_name}' has dimensions "
                f"{nuclear.shape}, expected "
                f"{merged.shape}."
            )
        nuclear *= np.float32(255.0)
        model_input[..., 1] = nuclear

    # Cellpose 4 accepts arbitrary channel order but its network consumes up to
    # three channels. Keeping membrane and nuclear guidance separate preserves
    # both signals; the unused third channel is explicitly zero.
    return model_input, nuclear_name


def create_cellpose_sam_model(gpu: bool = True):
    """Load Cellpose-SAM for CUDA batch work or CPU FOV previews."""
    if gpu and not cuda_compatible_gpu_available():
        raise RuntimeError(
            "Cellpose-SAM segmentation requires a CUDA-compatible GPU, but "
            "none was detected by PyTorch."
        )
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError(
            "Cellpose is not installed. Install this ORBIT version with "
            "'python -m pip install -e .' and try again."
        ) from error

    model_path = bundled_cellpose_sam_model_path()
    frozen_build = bool(
        getattr(sys, "frozen", False) or "__compiled__" in globals()
    )
    if model_path is None and frozen_build:
        raise RuntimeError(
            "This ORBIT installation is missing its verified cpsam_v2 model. "
            "Reinstall ORBIT while connected to the internet so Setup can "
            "download and verify the model."
        )

    # Source checkouts retain Cellpose's normal cache/download fallback when no
    # local model exists. Standalone builds always pass the installer-provided
    # path and therefore never download from the segmentation worker.
    with _writable_cellpose_streams():
        return models.CellposeModel(
            gpu=bool(gpu),
            pretrained_model=(
                str(model_path) if model_path is not None else CELLPOSE_SAM_MODEL
            ),
            use_bfloat16=False,
        )


class _RegionImage:
    """Lazy image adapter that reads only the requested FOV for each channel."""

    def __init__(self, source, y0: int, x0: int, height: int, width: int):
        self.path = source.path
        self._source = source
        self._y0, self._x0 = int(y0), int(x0)
        self._height, self._width = int(height), int(width)
        self._names = [str(name) for name in source.get_channel_names()]

    def get_channel_names(self):
        return list(self._names)

    def get_channel(self, index: int):
        return self._source.get_region(
            int(index),
            self._y0,
            self._x0,
            self._height,
            self._width,
        )


def _safe_channel_labels(channel_names: Iterable[str]) -> list[str]:
    """Create unique, non-empty labels for measurement-table columns."""
    labels = []
    counts: dict[str, int] = {}
    for index, raw_name in enumerate(channel_names, start=1):
        base = str(raw_name).strip() or f"Channel {index}"
        counts[base] = counts.get(base, 0) + 1
        labels.append(
            base if counts[base] == 1 else f"{base} ({counts[base]})"
        )
    return labels


def _intensity_statistics_by_label(
    channel: np.ndarray,
    masks: np.ndarray,
    maximum_label: int,
    chunk_rows: int = 512,
    selected_pixels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate finite-pixel intensity statistics for all or selected pixels."""
    if selected_pixels is not None:
        selected_pixels = np.asarray(selected_pixels, dtype=bool)
        if selected_pixels.shape != masks.shape:
            raise ValueError(
                "The compartment selection and segmentation mask dimensions "
                f"differ ({selected_pixels.shape} versus {masks.shape})."
            )
    counts = np.zeros(maximum_label + 1, dtype=np.uint64)
    sums = np.zeros(maximum_label + 1, dtype=np.float64)
    squared_sums = np.zeros(maximum_label + 1, dtype=np.float64)
    minima = np.full(maximum_label + 1, np.inf, dtype=np.float64)
    maxima = np.full(maximum_label + 1, -np.inf, dtype=np.float64)

    for y0 in range(0, masks.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, masks.shape[0])
        labels = np.asarray(masks[y0:y1]).reshape(-1)
        values = np.asarray(channel[y0:y1]).reshape(-1)
        valid = (labels > 0) & np.isfinite(values)
        if selected_pixels is not None:
            valid &= selected_pixels[y0:y1].reshape(-1)
        if not np.any(valid):
            continue
        labels = labels[valid].astype(np.int64, copy=False)
        values = values[valid].astype(np.float64, copy=False)
        counts += np.bincount(labels, minlength=maximum_label + 1).astype(
            np.uint64,
            copy=False,
        )
        sums += np.bincount(
            labels,
            weights=values,
            minlength=maximum_label + 1,
        )
        squared_sums += np.bincount(
            labels,
            weights=values * values,
            minlength=maximum_label + 1,
        )
        np.minimum.at(minima, labels, values)
        np.maximum.at(maxima, labels, values)

    means = np.full(maximum_label + 1, np.nan, dtype=np.float64)
    deviations = np.full(maximum_label + 1, np.nan, dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present]
    variance = np.zeros(maximum_label + 1, dtype=np.float64)
    variance[present] = (
        squared_sums[present] / counts[present] - means[present] ** 2
    )
    deviations[present] = np.sqrt(np.maximum(variance[present], 0.0))
    minima[~present] = np.nan
    maxima[~present] = np.nan
    return means, deviations, minima, maxima


def _segmentation_compartment_masks(
    masks: np.ndarray,
    pixel_size_um: float,
    membrane_width_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition cell labels into an eroded nucleus and peripheral membrane."""
    pixel_size_um = float(pixel_size_um)
    membrane_width_um = float(membrane_width_um)
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("The pixel size must be a finite positive number.")
    if not np.isfinite(membrane_width_um) or membrane_width_um < 0:
        raise ValueError(
            "The membrane-compartment width must be a finite non-negative "
            "number."
        )

    inward_buffer_pixels = max(
        int(round(membrane_width_um / pixel_size_um)),
        0,
    )
    nucleus_pixels = compartment_mask_for_rows(
        masks,
        0,
        masks.shape[0],
        "nucleus",
        inward_buffer_pixels,
    )
    cell_pixels = masks > 0
    membrane_pixels = cell_pixels & ~nucleus_pixels
    return nucleus_pixels, membrane_pixels


def _compartment_pixel_counts(
    masks: np.ndarray,
    selected_pixels: np.ndarray,
    maximum_label: int,
) -> np.ndarray:
    """Count selected pixels for each cell label."""
    labels = np.asarray(masks)[selected_pixels]
    return np.bincount(
        labels.astype(np.int64, copy=False),
        minlength=maximum_label + 1,
    )


def measure_segmented_cells(
    masks: np.ndarray,
    image,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    membrane_width_um: float = DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM,
) -> pd.DataFrame:
    """Create morphology and compartment fluorescence data for every cell.

    Cellpose-SAM returns one whole-cell label mask. ORBIT therefore defines the
    membrane compartment as the outer ``membrane_width_um`` of each cell and
    the nucleus compartment as the remaining eroded interior. Together these
    two disjoint compartments cover the complete Cellpose cell mask.
    """
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError(f"Cellpose returned a non-2D mask: {masks.shape}.")
    if not np.any(masks > 0):
        raise ValueError("Cellpose did not identify any cells in the image.")

    properties = regionprops_table(
        masks,
        properties=(
            "label",
            "centroid",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "extent",
            "equivalent_diameter_area",
            "major_axis_length",
            "minor_axis_length",
            "bbox",
        ),
    )
    cells = pd.DataFrame(properties).rename(columns={
        "label": "Cell ID",
        "centroid-0": "Centroid Y px",
        "centroid-1": "Centroid X px",
        "area": "Area px",
        "perimeter": "Perimeter px",
        "eccentricity": "Eccentricity",
        "solidity": "Solidity",
        "extent": "Extent",
        "equivalent_diameter_area": "Equivalent diameter px",
        "major_axis_length": "Major axis length px",
        "minor_axis_length": "Minor axis length px",
        "bbox-0": "Bounding box Y min px",
        "bbox-1": "Bounding box X min px",
        "bbox-2": "Bounding box Y max px",
        "bbox-3": "Bounding box X max px",
    })
    cells["Cell ID"] = cells["Cell ID"].astype(np.int64)
    cells.insert(
        1,
        "Centroid X µm",
        cells["Centroid X px"] * float(pixel_size_um),
    )
    cells.insert(
        2,
        "Centroid Y µm",
        cells["Centroid Y px"] * float(pixel_size_um),
    )
    cells["Area µm²"] = cells["Area px"] * float(pixel_size_um) ** 2
    cells["Perimeter µm"] = cells["Perimeter px"] * float(pixel_size_um)
    cells["Equivalent diameter µm"] = (
        cells["Equivalent diameter px"] * float(pixel_size_um)
    )
    cells["Major axis length µm"] = (
        cells["Major axis length px"] * float(pixel_size_um)
    )
    cells["Minor axis length µm"] = (
        cells["Minor axis length px"] * float(pixel_size_um)
    )

    cell_ids = cells["Cell ID"].to_numpy(dtype=np.int64)
    maximum_label = int(masks.max())
    nucleus_pixels, membrane_pixels = _segmentation_compartment_masks(
        masks,
        pixel_size_um=pixel_size_um,
        membrane_width_um=membrane_width_um,
    )
    nucleus_counts = _compartment_pixel_counts(
        masks,
        nucleus_pixels,
        maximum_label,
    )
    membrane_counts = _compartment_pixel_counts(
        masks,
        membrane_pixels,
        maximum_label,
    )
    cells["Nucleus Area px"] = nucleus_counts[cell_ids]
    cells["Nucleus Area µm²"] = (
        cells["Nucleus Area px"] * float(pixel_size_um) ** 2
    )
    cells["Membrane Area px"] = membrane_counts[cell_ids]
    cells["Membrane Area µm²"] = (
        cells["Membrane Area px"] * float(pixel_size_um) ** 2
    )

    channel_names = [str(name) for name in image.get_channel_names()]
    channel_labels = _safe_channel_labels(channel_names)
    for channel_index, channel_label in enumerate(channel_labels):
        channel = image.get_channel(channel_index)
        for compartment_name, selected_pixels in (
            ("Cell", None),
            ("Nucleus", nucleus_pixels),
            ("Membrane", membrane_pixels),
        ):
            means, deviations, minima, maxima = _intensity_statistics_by_label(
                channel,
                masks,
                maximum_label,
                selected_pixels=selected_pixels,
            )
            prefix = f"{channel_label}: {compartment_name}"
            cells[f"{prefix} Mean"] = means[cell_ids]
            cells[f"{prefix} Std Dev"] = deviations[cell_ids]
            cells[f"{prefix} Min"] = minima[cell_ids]
            cells[f"{prefix} Max"] = maxima[cell_ids]

    return cells


def _offset_fov_measurements(
    cells: pd.DataFrame,
    y0: int,
    x0: int,
    pixel_size_um: float,
) -> pd.DataFrame:
    cells = cells.copy()
    for column in ("Centroid X px", "Bounding box X min px", "Bounding box X max px"):
        if column in cells:
            cells[column] = cells[column] + int(x0)
    for column in ("Centroid Y px", "Bounding box Y min px", "Bounding box Y max px"):
        if column in cells:
            cells[column] = cells[column] + int(y0)
    if "Centroid X px" in cells:
        cells["Centroid X µm"] = cells["Centroid X px"] * float(pixel_size_um)
    if "Centroid Y px" in cells:
        cells["Centroid Y µm"] = cells["Centroid Y px"] * float(pixel_size_um)
    return cells


def segment_fov_preview(
    image,
    selected_marker_names: Iterable[str],
    y0: int,
    x0: int,
    height: int,
    width: int,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    model=None,
    *,
    nuclear_channel_name: str | None = None,
    membrane_width_um: float = DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM,
) -> dict:
    """Segment only the displayed FOV on CPU without changing project data."""
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    region_image = _RegionImage(image, int(y0), int(x0), int(height), int(width))
    model_input, nuclear_name = build_cellpose_input(
        region_image,
        selected,
        nuclear_channel_name=nuclear_channel_name,
    )
    if model is None:
        model = create_cellpose_sam_model(gpu=False)
    masks, _flows, _styles = model.eval(
        model_input,
        channel_axis=-1,
        normalize=True,
        diameter=None,
        batch_size=8,
        tile_overlap=0.1,
    )
    masks = np.asarray(masks, dtype=np.uint32)
    cell_data = measure_segmented_cells(
        masks,
        region_image,
        pixel_size_um=pixel_size_um,
        membrane_width_um=membrane_width_um,
    )
    cell_data = _offset_fov_measurements(
        cell_data, y0=int(y0), x0=int(x0), pixel_size_um=pixel_size_um
    )
    return {
        "image_path": str(Path(image.path).resolve()),
        "x0": int(x0),
        "y0": int(y0),
        "width": int(width),
        "height": int(height),
        "masks": masks,
        "cell_data": cell_data,
        "cell_count": len(cell_data),
        "marker_names": selected,
        "nuclear_channel_name": nuclear_name,
        "model_name": CELLPOSE_SAM_MODEL,
        "compute_device": "cpu",
        "scope": "current_fov",
        "pixel_size_um": float(pixel_size_um),
        "membrane_width_um": float(membrane_width_um),
    }


def merge_fov_segmentation(
    *,
    preview: dict,
    image_shape: tuple[int, int],
    existing_masks: np.ndarray | None,
    existing_cell_data: pd.DataFrame | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Accept a preview by replacing cells intersecting its rectangular FOV."""
    height, width = map(int, image_shape)
    y0, x0 = int(preview["y0"]), int(preview["x0"])
    local_masks = np.asarray(preview["masks"], dtype=np.uint32)
    y1, x1 = y0 + local_masks.shape[0], x0 + local_masks.shape[1]
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError("The segmentation preview lies outside the source image.")
    if existing_masks is None:
        merged_masks = np.zeros((height, width), dtype=np.uint32)
    else:
        if tuple(existing_masks.shape) != (height, width):
            raise ValueError("The existing segmentation dimensions do not match the image.")
        merged_masks = np.array(existing_masks, dtype=np.uint32, copy=True)

    replaced_labels = np.unique(merged_masks[y0:y1, x0:x1])
    replaced_labels = replaced_labels[replaced_labels > 0]
    for label in replaced_labels:
        merged_masks[merged_masks == label] = 0

    maximum_label = int(merged_masks.max(initial=0))
    relabelled = np.zeros_like(local_masks, dtype=np.uint32)
    local_labels = np.unique(local_masks)
    local_labels = local_labels[local_labels > 0]
    label_mapping = {
        int(label): maximum_label + index
        for index, label in enumerate(local_labels, start=1)
    }
    for old_label, new_label in label_mapping.items():
        relabelled[local_masks == old_label] = new_label
    merged_masks[y0:y1, x0:x1] = relabelled

    new_cells = preview["cell_data"].copy()
    if "Cell ID" in new_cells:
        new_cells["Cell ID"] = (
            pd.to_numeric(new_cells["Cell ID"], errors="raise")
            .astype(int)
            .map(label_mapping)
            .astype(np.int64)
        )
    if existing_cell_data is None:
        merged_cells = new_cells
    else:
        retained = existing_cell_data.copy()
        identifier_column = next(
            (
                column for column in retained.columns
                if str(column).strip().lower().replace("_", " ").replace("-", " ")
                in {"cell id", "object id", "mask id", "label id"}
            ),
            None,
        )
        if identifier_column is not None and replaced_labels.size:
            ids = pd.to_numeric(retained[identifier_column], errors="coerce")
            retained = retained.loc[~ids.isin(replaced_labels)].copy()
        elif "Centroid X px" in retained and "Centroid Y px" in retained:
            xs = pd.to_numeric(retained["Centroid X px"], errors="coerce")
            ys = pd.to_numeric(retained["Centroid Y px"], errors="coerce")
            retained = retained.loc[
                ~((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1))
            ].copy()
        elif "Centroid X µm" in retained and "Centroid Y µm" in retained:
            scale = float(preview.get("pixel_size_um", DEFAULT_PIXEL_SIZE_UM))
            xs = pd.to_numeric(retained["Centroid X µm"], errors="coerce") / scale
            ys = pd.to_numeric(retained["Centroid Y µm"], errors="coerce") / scale
            retained = retained.loc[
                ~((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1))
            ].copy()
        merged_cells = pd.concat([retained, new_cells], ignore_index=True, sort=False)
    if "Cell ID" in merged_cells:
        merged_cells = merged_cells.sort_values("Cell ID").reset_index(drop=True)
    return merged_masks, merged_cells


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=target.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _write_segmentation_outputs(
    cell_path: Path,
    mask_path: Path,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    marker_names: Iterable[str],
    nuclear_channel_name: str | None,
) -> tuple[Path, Path]:
    """Atomically write one cell-data/mask output pair."""
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cell_path = _temporary_path(cell_path)
    temporary_mask_path = _temporary_path(mask_path)
    try:
        cell_data.to_csv(temporary_cell_path, sep="\t", index=False)
        tifffile.imwrite(
            temporary_mask_path,
            np.asarray(masks, dtype=np.uint32),
            bigtiff=True,
            metadata={
                "axes": "YX",
                "ORBIT segmentation model": CELLPOSE_SAM_MODEL,
                "ORBIT membrane markers": list(marker_names),
                "ORBIT nuclear marker": nuclear_channel_name or "",
            },
        )
        os.replace(temporary_cell_path, cell_path)
        os.replace(temporary_mask_path, mask_path)
    finally:
        temporary_cell_path.unlink(missing_ok=True)
        temporary_mask_path.unlink(missing_ok=True)
    return cell_path, mask_path


def save_segmentation_outputs(
    image_path: str | Path,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    marker_names: Iterable[str],
    nuclear_channel_name: str | None,
) -> tuple[Path, Path]:
    """Atomically replace ORBIT's generated mask and cell-data outputs."""
    cell_path, mask_path = output_paths_for_image(image_path)
    return _write_segmentation_outputs(
        cell_path,
        mask_path,
        masks,
        cell_data,
        marker_names,
        nuclear_channel_name,
    )


def export_segmentation_outputs(
    destination_directory: str | Path,
    image_path: str | Path,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    marker_names: Iterable[str] = (),
    nuclear_channel_name: str | None = None,
    filename_stem: str | None = None,
) -> tuple[Path, Path]:
    """Export a Cellpose data table and label mask to a chosen directory."""
    destination = Path(destination_directory).expanduser().resolve()
    if not destination.is_dir():
        raise NotADirectoryError(
            f"Segmentation export directory not found: {destination}"
        )
    cell_path, mask_path = segmentation_export_paths(
        destination,
        image_path,
        filename_stem=filename_stem,
    )
    return _write_segmentation_outputs(
        cell_path,
        mask_path,
        masks,
        cell_data,
        marker_names,
        nuclear_channel_name,
    )


def segment_image(
    image,
    selected_marker_names: Iterable[str],
    model,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    *,
    nuclear_channel_name: str | None = None,
    membrane_width_um: float = DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM,
    fovs: Iterable[SegmentationFOV] | None = None,
    fov_size: int = DEFAULT_SEGMENTATION_FOV_SIZE,
    fov_overlap: float = DEFAULT_SEGMENTATION_FOV_OVERLAP,
    dapi_positive_only: bool = True,
    minimum_dapi_fraction: float = DEFAULT_MINIMUM_DAPI_FRACTION,
    progress_callback: Callable[[dict], None] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> dict:
    """Segment one image as overlapping FOVs and stitch it on disk."""
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    channel_names = [str(name) for name in image.get_channel_names()]
    nuclear_name = resolve_nuclear_channel_name(
        channel_names,
        nuclear_channel_name,
    )
    if fovs is None:
        candidates = tiled_segmentation_fovs(
            image.get_shape()[-2:],
            fov_size=fov_size,
            overlap=fov_overlap,
        )
        if dapi_positive_only and nuclear_name is not None:
            fovs = select_dapi_positive_fovs(
                image,
                candidates,
                channel_names.index(nuclear_name),
                minimum_dapi_fraction=minimum_dapi_fraction,
            )
        else:
            fovs = candidates
        candidate_count = len(candidates)
    else:
        fovs = list(fovs)
        candidate_count = len(
            tiled_segmentation_fovs(
                image.get_shape()[-2:],
                fov_size=fov_size,
                overlap=fov_overlap,
            )
        )
    fovs = list(fovs)
    if not fovs:
        raise ValueError(
            f"No nuclear-positive segmentation FOVs were found in "
            f"{Path(image.path).name}."
        )

    cell_path, mask_path = output_paths_for_image(image.path)
    stitcher = AstroPathMaskStitcher(
        image_shape=image.get_shape()[-2:],
        cell_path=cell_path,
        mask_path=mask_path,
        model_name=CELLPOSE_SAM_MODEL,
        marker_names=selected,
        nuclear_channel_name=nuclear_name,
    )
    finalized = False
    try:
        for current, fov in enumerate(fovs, start=1):
            region_image = _RegionImage(
                image,
                fov.y0,
                fov.x0,
                fov.height,
                fov.width,
            )
            model_input, _nuclear_name = build_cellpose_input(
                region_image,
                selected,
                nuclear_channel_name=nuclear_name,
            )
            masks, _flows, _styles = model.eval(
                model_input,
                channel_axis=-1,
                normalize=True,
                diameter=None,
                batch_size=8,
                tile_overlap=0.1,
            )
            masks = np.asarray(masks, dtype=np.uint32)
            if np.any(masks > 0):
                cells = measure_segmented_cells(
                    masks,
                    region_image,
                    pixel_size_um=pixel_size_um,
                    membrane_width_um=membrane_width_um,
                )
                stitcher.add_fov(
                    fov,
                    masks,
                    cells,
                    pixel_size_um=pixel_size_um,
                )
            if progress_callback is not None:
                global_current = int(progress_offset) + current
                progress_callback({
                    "phase": "segmenting",
                    "current": global_current,
                    "total": int(progress_total or len(fovs)),
                    "message": (
                        f"Segmenting and stitching {Path(image.path).name}: FOV "
                        f"{current:,}/{len(fovs):,}"
                    ),
                })
        if progress_callback is not None:
            progress_callback({
                "phase": "finalizing",
                "message": f"Finalizing {Path(image.path).name} outputs...",
            })
        cell_path, mask_path, cell_count = stitcher.finalize()
        finalized = True
    finally:
        if not finalized:
            stitcher.abort()

    return {
        "image_path": str(Path(image.path).resolve()),
        "cell_data_path": str(cell_path),
        "segmentation_mask_path": str(mask_path),
        "cell_count": int(cell_count),
        "marker_names": selected,
        "nuclear_channel_name": nuclear_name,
        "membrane_width_um": float(membrane_width_um),
        "model_name": CELLPOSE_SAM_MODEL,
        "compute_device": "cuda",
        "scope": "tiled_whole_image",
        "fov_size": int(fov_size),
        "fov_overlap": float(fov_overlap),
        "candidate_fov_count": int(candidate_count),
        "selected_fov_count": len(fovs),
        "dapi_positive_only": bool(dapi_positive_only),
        "nuclear_positive_only": bool(dapi_positive_only),
        "minimum_dapi_fraction": float(minimum_dapi_fraction),
        "stitching_method": "AstroPath-style primary regions",
    }


def segment_project_images(
    images: Iterable,
    selected_marker_names: Iterable[str],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    nuclear_channel_name: str | None = None,
    membrane_width_um: float = DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM,
    fov_size: int = DEFAULT_SEGMENTATION_FOV_SIZE,
    fov_overlap: float = DEFAULT_SEGMENTATION_FOV_OVERLAP,
    dapi_positive_only: bool = True,
    minimum_dapi_fraction: float = DEFAULT_MINIMUM_DAPI_FRACTION,
) -> list[dict]:
    """Tile, optionally nuclear-filter, segment, and stitch project images."""
    images = list(images)
    selected_markers = list(
        dict.fromkeys(str(name) for name in selected_marker_names)
    )
    if not images:
        raise ValueError("Load at least one image before segmenting.")
    if not selected_markers:
        raise ValueError("Select at least one membrane marker before segmenting.")

    candidates_by_image = [
        tiled_segmentation_fovs(
            image.get_shape()[-2:],
            fov_size=fov_size,
            overlap=fov_overlap,
        )
        for image in images
    ]
    total_candidates = sum(map(len, candidates_by_image))
    scanned_candidates = 0
    selected_by_image = []
    for image, candidates in zip(images, candidates_by_image):
        channel_names = [str(name) for name in image.get_channel_names()]
        nuclear_name = resolve_nuclear_channel_name(
            channel_names,
            nuclear_channel_name,
        )
        if dapi_positive_only and nuclear_name is not None:
            offset = scanned_candidates

            def report_selection(current, _total, _fov, positive_fraction):
                if progress_callback is not None:
                    progress_callback({
                        "phase": "selecting_fovs",
                        "current": offset + current,
                        "total": total_candidates,
                        "message": (
                            f"Selecting {nuclear_name}-positive FOVs: "
                            f"{offset + current:,}/{total_candidates:,} "
                            f"({positive_fraction:.1%} positive pixels)"
                        ),
                    })

            selected_fovs = select_dapi_positive_fovs(
                image,
                candidates,
                channel_names.index(nuclear_name),
                minimum_dapi_fraction=minimum_dapi_fraction,
                progress_callback=report_selection,
            )
        else:
            selected_fovs = list(candidates)
            if progress_callback is not None:
                progress_callback({
                    "phase": "selecting_fovs",
                    "current": scanned_candidates + len(candidates),
                    "total": total_candidates,
                    "message": (
                        f"Using all {len(candidates):,} FOVs for "
                        f"{Path(image.path).name}"
                        + (" (no nuclear channel selected)" if nuclear_name is None else "")
                    ),
                })
        selected_by_image.append(selected_fovs)
        scanned_candidates += len(candidates)

    total_selected = sum(map(len, selected_by_image))
    if total_selected == 0:
        raise ValueError(
            "No FOV met the minimum nuclear-positive pixel requirement. "
            "Disable nuclear-positive FOV selection or lower the selection "
            "requirement."
        )
    if progress_callback is not None:
        progress_callback({
            "phase": "loading_model",
            "message": f"Loading Cellpose-SAM model {CELLPOSE_SAM_MODEL}...",
        })
    model = create_cellpose_sam_model(gpu=True)
    if progress_callback is not None:
        progress_callback({
            "phase": "model_loaded",
            "message": (
                f"Cellpose-SAM model {CELLPOSE_SAM_MODEL} loaded; starting "
                "tiled segmentation..."
            ),
        })
    results = []
    segmented_fovs = 0
    for image, selected_fovs in zip(images, selected_by_image):
        results.append(
            segment_image(
                image,
                selected_markers,
                model,
                pixel_size_um=image.get_pixel_size_um(
                    default=pixel_size_um
                ),
                nuclear_channel_name=nuclear_channel_name,
                membrane_width_um=membrane_width_um,
                fovs=selected_fovs,
                fov_size=fov_size,
                fov_overlap=fov_overlap,
                dapi_positive_only=dapi_positive_only,
                minimum_dapi_fraction=minimum_dapi_fraction,
                progress_callback=progress_callback,
                progress_offset=segmented_fovs,
                progress_total=total_selected,
            )
        )
        segmented_fovs += len(selected_fovs)
    return results


def segment_project_image_paths(
    image_paths: Iterable[str | Path],
    selected_marker_names: Iterable[str],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    nuclear_channel_name: str | None = None,
    membrane_width_um: float = DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM,
    fov_size: int = DEFAULT_SEGMENTATION_FOV_SIZE,
    fov_overlap: float = DEFAULT_SEGMENTATION_FOV_OVERLAP,
    dapi_positive_only: bool = True,
    minimum_dapi_fraction: float = DEFAULT_MINIMUM_DAPI_FRACTION,
) -> list[dict]:
    """Open, segment, and close images in the calling execution context.

    The viewer keeps image readers alive for Napari. Reusing those same TIFF
    readers in a background worker lets the viewer and segmentation code access
    a single ``tifffile`` Zarr store and its native decoder concurrently. On
    Windows that can terminate the process without a Python exception. Opening
    independent readers here gives the segmentation worker sole ownership of
    its TIFF handles while retaining the existing OME-Zarr behaviour.
    """
    from orbit.image import QPTiffImage

    paths = [Path(path).expanduser().resolve() for path in image_paths]
    if not paths:
        raise ValueError("Load at least one image before segmenting.")

    images = []
    try:
        for path in paths:
            images.append(QPTiffImage(path))
        return segment_project_images(
            images,
            selected_marker_names,
            pixel_size_um=pixel_size_um,
            progress_callback=progress_callback,
            nuclear_channel_name=nuclear_channel_name,
            membrane_width_um=membrane_width_um,
            fov_size=fov_size,
            fov_overlap=fov_overlap,
            dapi_positive_only=dapi_positive_only,
            minimum_dapi_fraction=minimum_dapi_fraction,
        )
    finally:
        for image in reversed(images):
            image.close()


__all__ = [
    "CELLPOSE_SAM_MODEL",
    "DEFAULT_MEMBRANE_COMPARTMENT_WIDTH_UM",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_SEGMENTATION_FOV_OVERLAP",
    "DEFAULT_SEGMENTATION_FOV_SIZE",
    "build_cellpose_input",
    "bundled_cellpose_sam_model_path",
    "create_cellpose_sam_model",
    "cuda_compatible_gpu_available",
    "dapi_channel_name",
    "export_segmentation_outputs",
    "is_dapi_channel",
    "measure_segmented_cells",
    "merge_fov_segmentation",
    "membrane_marker_names",
    "output_paths_for_image",
    "save_segmentation_outputs",
    "resolve_nuclear_channel_name",
    "segmentation_export_paths",
    "segment_image",
    "segment_fov_preview",
    "segment_project_image_paths",
    "segment_project_images",
]
