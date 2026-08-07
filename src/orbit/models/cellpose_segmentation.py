"""Cellpose-SAM segmentation and cell-level measurement generation."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

CELLPOSE_SAM_MODEL = "cpsam_v2"
DEFAULT_PIXEL_SIZE_UM = 0.5064


def is_dapi_channel(channel_name: str) -> bool:
    """Return whether a channel name identifies DAPI."""
    return "dapi" in str(channel_name).strip().lower()


def membrane_marker_names(channel_names: Iterable[str]) -> list[str]:
    """Return channels that can be selected as membrane guides."""
    return [str(name) for name in channel_names if not is_dapi_channel(name)]


def dapi_channel_name(channel_names: Iterable[str]) -> str | None:
    """Return the first DAPI channel, if one is present."""
    return next(
        (str(name) for name in channel_names if is_dapi_channel(name)),
        None,
    )


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
    base_name = image_path.stem
    return (
        image_path.with_name(f"{base_name}_orbit_cellpose_cells.tsv"),
        image_path.with_name(f"{base_name}_orbit_cellpose_masks.tif"),
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
) -> tuple[np.ndarray, str | None]:
    """Build a Y-X-C Cellpose input from merged markers and optional DAPI."""
    channel_names = [str(name) for name in image.get_channel_names()]
    channel_indices = {name: index for index, name in enumerate(channel_names)}
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    if not selected:
        raise ValueError("Select at least one membrane marker before segmenting.")
    if any(is_dapi_channel(name) for name in selected):
        raise ValueError(
            "DAPI is supplied automatically as the nuclear channel and cannot "
            "be selected as a membrane marker."
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

    nuclear_name = dapi_channel_name(channel_names)
    if nuclear_name is not None:
        nuclear = _normalized_channel(
            image.get_channel(channel_indices[nuclear_name])
        )
        if nuclear.shape != merged.shape:
            raise ValueError(
                f"DAPI has dimensions {nuclear.shape}, expected "
                f"{merged.shape}."
            )
        nuclear *= np.float32(255.0)
        model_input[..., 1] = nuclear

    # Cellpose 4 accepts arbitrary channel order but its network consumes up to
    # three channels. Keeping membrane and nuclear guidance separate preserves
    # both signals; the unused third channel is explicitly zero.
    return model_input, nuclear_name


def create_cellpose_sam_model():
    """Load the current Cellpose-SAM model, using a GPU when available."""
    if not cuda_compatible_gpu_available():
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

    return models.CellposeModel(
        gpu=True,
        pretrained_model=CELLPOSE_SAM_MODEL,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate finite-pixel intensity statistics without copying full masks."""
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


def measure_segmented_cells(
    masks: np.ndarray,
    image,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
) -> pd.DataFrame:
    """Create morphology and all-channel fluorescence data for every cell."""
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
    channel_names = [str(name) for name in image.get_channel_names()]
    channel_labels = _safe_channel_labels(channel_names)
    for channel_index, channel_label in enumerate(channel_labels):
        channel = image.get_channel(channel_index)
        means, deviations, minima, maxima = _intensity_statistics_by_label(
            channel,
            masks,
            maximum_label,
        )
        cells[f"{channel_label}: Cell Mean"] = means[cell_ids]
        cells[f"{channel_label}: Cell Std Dev"] = deviations[cell_ids]
        cells[f"{channel_label}: Cell Min"] = minima[cell_ids]
        cells[f"{channel_label}: Cell Max"] = maxima[cell_ids]

    return cells


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=target.suffix,
    )
    os.close(descriptor)
    return Path(name)


def save_segmentation_outputs(
    image_path: str | Path,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    marker_names: Iterable[str],
    nuclear_channel_name: str | None,
) -> tuple[Path, Path]:
    """Atomically replace ORBIT's generated mask and cell-data outputs."""
    cell_path, mask_path = output_paths_for_image(image_path)
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


def segment_image(
    image,
    selected_marker_names: Iterable[str],
    model,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
) -> dict:
    """Segment one loaded image and replace its generated ORBIT outputs."""
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    model_input, nuclear_name = build_cellpose_input(image, selected)
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
        image,
        pixel_size_um=pixel_size_um,
    )
    cell_path, mask_path = save_segmentation_outputs(
        image.path,
        masks,
        cell_data,
        selected,
        nuclear_name,
    )
    return {
        "image_path": str(Path(image.path).resolve()),
        "cell_data_path": str(cell_path),
        "segmentation_mask_path": str(mask_path),
        "cell_count": len(cell_data),
        "marker_names": selected,
        "nuclear_channel_name": nuclear_name,
        "model_name": CELLPOSE_SAM_MODEL,
    }


def segment_project_images(
    images: Iterable,
    selected_marker_names: Iterable[str],
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run one shared Cellpose-SAM model over every image in a project."""
    images = list(images)
    selected = list(dict.fromkeys(str(name) for name in selected_marker_names))
    if not images:
        raise ValueError("Load at least one image before segmenting.")
    if not selected:
        raise ValueError("Select at least one membrane marker before segmenting.")

    if progress_callback is not None:
        progress_callback(
            f"Loading Cellpose-SAM model {CELLPOSE_SAM_MODEL}..."
        )
    model = create_cellpose_sam_model()
    results = []
    for index, image in enumerate(images, start=1):
        if progress_callback is not None:
            progress_callback(
                f"Segmenting {Path(image.path).name} ({index}/{len(images)})..."
            )
        results.append(
            segment_image(
                image,
                selected,
                model,
                pixel_size_um=pixel_size_um,
            )
        )
    return results


__all__ = [
    "CELLPOSE_SAM_MODEL",
    "DEFAULT_PIXEL_SIZE_UM",
    "build_cellpose_input",
    "create_cellpose_sam_model",
    "cuda_compatible_gpu_available",
    "dapi_channel_name",
    "is_dapi_channel",
    "measure_segmented_cells",
    "membrane_marker_names",
    "output_paths_for_image",
    "save_segmentation_outputs",
    "segment_image",
    "segment_project_images",
]
