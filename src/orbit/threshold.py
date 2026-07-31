from __future__ import annotations

import numpy as np
import pandas as pd


def intensity_threshold_from_slider(
    image: np.ndarray,
    slider_value: int,
    slider_maximum: int = 1000,
) -> float:
    """Map a slider position onto the display's 1st-to-99th percentile range."""
    if slider_maximum <= 0:
        raise ValueError("The threshold slider maximum must be positive.")

    values = np.asarray(image)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("The displayed channel contains no finite pixel values.")

    low, high = np.percentile(finite_values, (1, 99))
    position = np.clip(float(slider_value) / slider_maximum, 0.0, 1.0)
    return float(low + position * (high - low))


def _grow_counts(counts: np.ndarray, required_size: int) -> np.ndarray:
    if required_size <= counts.size:
        return counts
    expanded = np.zeros(required_size, dtype=np.uint64)
    expanded[: counts.size] = counts
    return expanded


def pixel_counts_by_mask_label(
    channel: np.ndarray,
    masks: np.ndarray,
    intensity_threshold: float,
    chunk_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Count total and above-threshold pixels for each positive mask label."""
    channel = np.asarray(channel)
    masks = np.asarray(masks)
    if channel.ndim != 2 or masks.ndim != 2:
        raise ValueError("Threshold phenotyping requires two-dimensional arrays.")
    if channel.shape != masks.shape:
        raise ValueError(
            "The fluorescence channel and segmentation mask dimensions differ "
            f"({channel.shape} versus {masks.shape})."
        )
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    total_counts = np.zeros(1, dtype=np.uint64)
    positive_counts = np.zeros(1, dtype=np.uint64)

    for y0 in range(0, masks.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, masks.shape[0])
        mask_chunk = np.asarray(masks[y0:y1])
        channel_chunk = np.asarray(channel[y0:y1])

        valid = np.isfinite(mask_chunk) & (mask_chunk > 0)
        if not np.any(valid):
            continue

        labels = mask_chunk[valid].astype(np.int64, copy=False)
        if np.any(labels < 0):
            raise ValueError("Segmentation mask labels must be non-negative.")

        maximum_label = int(labels.max())
        required_size = maximum_label + 1
        total_counts = _grow_counts(total_counts, required_size)
        positive_counts = _grow_counts(positive_counts, required_size)

        chunk_totals = np.bincount(labels)
        total_counts[: chunk_totals.size] += chunk_totals.astype(
            np.uint64, copy=False
        )

        above_threshold = np.isfinite(channel_chunk[valid]) & (
            channel_chunk[valid] > intensity_threshold
        )
        if np.any(above_threshold):
            chunk_positive = np.bincount(labels[above_threshold])
            positive_counts[: chunk_positive.size] += chunk_positive.astype(
                np.uint64, copy=False
            )

    return total_counts, positive_counts


def _numeric_identifier_candidates(cell_data: pd.DataFrame) -> list[np.ndarray]:
    candidates = []
    preferred_names = {
        "cell id",
        "cellid",
        "object id",
        "objectid",
        "mask id",
        "maskid",
        "label id",
        "labelid",
    }
    for column in cell_data.columns:
        normalized = (
            str(column).strip().lower().replace("_", " ").replace("-", " ")
        )
        normalized_compact = normalized.replace(" ", "")
        if (
            normalized not in preferred_names
            and normalized_compact not in preferred_names
        ):
            continue
        values = pd.to_numeric(cell_data[column], errors="coerce").to_numpy(
            dtype=float
        )
        rounded = np.rint(values)
        valid = (
            np.isfinite(values)
            & (rounded > 0)
            & np.isclose(values, rounded)
        )
        identifier = np.zeros(len(cell_data), dtype=np.int64)
        identifier[valid] = rounded[valid].astype(np.int64)
        candidates.append(identifier)
    return candidates


def map_cell_rows_to_mask_labels(
    cell_data: pd.DataFrame,
    masks: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    total_counts: np.ndarray,
) -> np.ndarray:
    """Resolve each table row to a non-empty segmentation-mask label."""
    row_count = len(cell_data)
    x = np.asarray(centroid_x, dtype=float)
    y = np.asarray(centroid_y, dtype=float)
    if x.shape != (row_count,) or y.shape != (row_count,):
        raise ValueError("Centroid arrays must contain one value per cell-data row.")

    def labels_exist(values: np.ndarray) -> np.ndarray:
        valid = (values > 0) & (values < total_counts.size)
        existing = np.zeros(values.shape, dtype=bool)
        existing[valid] = total_counts[values[valid]] > 0
        return existing

    resolved = np.zeros(row_count, dtype=np.int64)

    for candidate in _numeric_identifier_candidates(cell_data):
        unresolved = ~labels_exist(resolved)
        candidate_valid = labels_exist(candidate)
        use = unresolved & candidate_valid
        resolved[use] = candidate[use]
        if labels_exist(resolved).all():
            return resolved

    rounded_x = np.rint(x)
    rounded_y = np.rint(y)
    coordinate_valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (rounded_x >= 0)
        & (rounded_x < masks.shape[1])
        & (rounded_y >= 0)
        & (rounded_y < masks.shape[0])
    )
    centroid_labels = np.zeros(row_count, dtype=np.int64)
    centroid_labels[coordinate_valid] = np.asarray(masks)[
        rounded_y[coordinate_valid].astype(np.int64),
        rounded_x[coordinate_valid].astype(np.int64),
    ].astype(np.int64, copy=False)
    unresolved = ~labels_exist(resolved)
    use = unresolved & labels_exist(centroid_labels)
    resolved[use] = centroid_labels[use]

    sequential_labels = np.arange(1, row_count + 1, dtype=np.int64)
    unresolved = ~labels_exist(resolved)
    use = unresolved & labels_exist(sequential_labels)
    resolved[use] = sequential_labels[use]

    unresolved = ~labels_exist(resolved)
    if np.any(unresolved):
        raise ValueError(
            "Could not match "
            f"{int(np.count_nonzero(unresolved)):,} cell-data rows to mask labels. "
            "Provide numeric Cell ID/Object ID values or centroids that fall "
            "inside their segmented cells."
        )
    return resolved


def phenotype_cells_by_threshold(
    channel: np.ndarray,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    intensity_threshold: float,
    positive_pixel_fraction: float,
    chunk_rows: int = 2048,
) -> dict[str, np.ndarray]:
    """Assign cells using the fraction of their pixels above an intensity cutoff."""
    if not 0.0 <= positive_pixel_fraction <= 1.0:
        raise ValueError("The positive-pixel fraction must be between 0 and 1.")

    total_counts, positive_counts = pixel_counts_by_mask_label(
        channel,
        masks,
        intensity_threshold,
        chunk_rows=chunk_rows,
    )
    mask_labels = map_cell_rows_to_mask_labels(
        cell_data,
        masks,
        centroid_x,
        centroid_y,
        total_counts,
    )
    fractions_by_label = np.divide(
        positive_counts,
        total_counts,
        out=np.zeros_like(positive_counts, dtype=float),
        where=total_counts > 0,
    )
    fractions = fractions_by_label[mask_labels]
    return {
        "mask_label": mask_labels,
        "positive_fraction": fractions,
        "positive": fractions > positive_pixel_fraction,
    }
