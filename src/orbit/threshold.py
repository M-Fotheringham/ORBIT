from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


DEFAULT_INWARD_BUFFER_PIXELS = 4


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
    expanded = np.zeros(required_size, dtype=counts.dtype)
    expanded[: counts.size] = counts
    return expanded


def _inner_boundaries(labels: np.ndarray) -> np.ndarray:
    """Find boundary pixels inside each positive labelled region."""
    labels = np.asarray(labels)
    cell_pixels = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)

    vertical_change = labels[1:, :] != labels[:-1, :]
    boundary[1:, :] |= cell_pixels[1:, :] & vertical_change
    boundary[:-1, :] |= cell_pixels[:-1, :] & vertical_change

    horizontal_change = labels[:, 1:] != labels[:, :-1]
    boundary[:, 1:] |= cell_pixels[:, 1:] & horizontal_change
    boundary[:, :-1] |= cell_pixels[:, :-1] & horizontal_change

    boundary[0, :] |= cell_pixels[0, :]
    boundary[-1, :] |= cell_pixels[-1, :]
    boundary[:, 0] |= cell_pixels[:, 0]
    boundary[:, -1] |= cell_pixels[:, -1]
    return boundary


def compartment_mask_for_rows(
    masks: np.ndarray,
    y0: int,
    y1: int,
    compartment: str,
    inward_buffer_pixels: int,
) -> np.ndarray:
    """Return the selected compartment for a row block of a labelled mask."""
    masks = np.asarray(masks)
    if compartment == "all":
        return masks[y0:y1] > 0
    if compartment not in {"nucleus", "cytoplasm_membrane"}:
        raise ValueError(f"Unsupported threshold compartment: {compartment}")
    if inward_buffer_pixels < 0:
        raise ValueError("The inward-buffer distance cannot be negative.")

    # Include a halo larger than the requested distance so the artificial
    # horizontal chunk edges cannot influence the rows returned to the caller.
    halo = int(inward_buffer_pixels) + 2
    halo_y0 = max(y0 - halo, 0)
    halo_y1 = min(y1 + halo, masks.shape[0])
    mask_chunk = np.asarray(masks[halo_y0:halo_y1])
    cell_pixels = mask_chunk > 0
    boundary = _inner_boundaries(mask_chunk)

    if np.any(boundary):
        distance_from_boundary = distance_transform_edt(~boundary)
    else:
        distance_from_boundary = np.full(mask_chunk.shape, np.inf)

    if compartment == "nucleus":
        selected = cell_pixels & (
            distance_from_boundary > inward_buffer_pixels
        )
    else:
        selected = cell_pixels & (
            distance_from_boundary <= inward_buffer_pixels
        )

    row_start = y0 - halo_y0
    row_stop = row_start + (y1 - y0)
    return selected[row_start:row_stop]


def _pixel_statistics_by_mask_label(
    channel: np.ndarray,
    masks: np.ndarray,
    intensity_threshold: float,
    compartment: str = "all",
    inward_buffer_pixels: int = DEFAULT_INWARD_BUFFER_PIXELS,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate threshold and intensity statistics by mask label."""
    channel = np.asarray(channel)
    masks = np.asarray(masks)
    if channel.ndim != 2 or masks.ndim != 2:
        raise ValueError("Threshold phenotyping requires two-dimensional arrays.")
    if channel.shape != masks.shape:
        raise ValueError(
            "The fluorescence channel and segmentation mask dimensions differ "
            f"({channel.shape} versus {masks.shape})."
        )
    if compartment not in {"all", "nucleus", "cytoplasm_membrane"}:
        raise ValueError(f"Unsupported threshold compartment: {compartment}")
    if inward_buffer_pixels < 0:
        raise ValueError("The inward-buffer distance cannot be negative.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    whole_cell_counts = np.zeros(1, dtype=np.uint64)
    denominator_counts = np.zeros(1, dtype=np.uint64)
    positive_counts = np.zeros(1, dtype=np.uint64)
    finite_intensity_counts = np.zeros(1, dtype=np.uint64)
    intensity_sums = np.zeros(1, dtype=np.float64)

    for y0 in range(0, masks.shape[0], chunk_rows):
        y1 = min(y0 + chunk_rows, masks.shape[0])
        mask_chunk = np.asarray(masks[y0:y1])
        channel_chunk = np.asarray(channel[y0:y1])

        whole_cell_valid = np.isfinite(mask_chunk) & (mask_chunk > 0)
        if not np.any(whole_cell_valid):
            continue

        whole_cell_labels = mask_chunk[whole_cell_valid].astype(
            np.int64, copy=False
        )
        if np.any(whole_cell_labels < 0):
            raise ValueError("Segmentation mask labels must be non-negative.")

        maximum_label = int(whole_cell_labels.max())
        required_size = maximum_label + 1
        whole_cell_counts = _grow_counts(whole_cell_counts, required_size)
        denominator_counts = _grow_counts(denominator_counts, required_size)
        positive_counts = _grow_counts(positive_counts, required_size)
        finite_intensity_counts = _grow_counts(
            finite_intensity_counts, required_size
        )
        intensity_sums = _grow_counts(intensity_sums, required_size)

        chunk_whole_cell = np.bincount(whole_cell_labels)
        whole_cell_counts[: chunk_whole_cell.size] += chunk_whole_cell.astype(
            np.uint64, copy=False
        )

        denominator_valid = whole_cell_valid
        if compartment != "all":
            denominator_valid = denominator_valid & compartment_mask_for_rows(
                masks,
                y0,
                y1,
                compartment,
                inward_buffer_pixels,
            )
        if not np.any(denominator_valid):
            continue

        denominator_labels = mask_chunk[denominator_valid].astype(
            np.int64, copy=False
        )
        chunk_denominator = np.bincount(denominator_labels)
        denominator_counts[: chunk_denominator.size] += chunk_denominator.astype(
            np.uint64, copy=False
        )

        denominator_intensities = channel_chunk[denominator_valid]
        finite_intensity = np.isfinite(denominator_intensities)
        if np.any(finite_intensity):
            finite_labels = denominator_labels[finite_intensity]
            chunk_finite_counts = np.bincount(finite_labels)
            finite_intensity_counts[
                : chunk_finite_counts.size
            ] += chunk_finite_counts.astype(np.uint64, copy=False)
            chunk_intensity_sums = np.bincount(
                finite_labels,
                weights=denominator_intensities[finite_intensity].astype(
                    np.float64, copy=False
                ),
            )
            intensity_sums[
                : chunk_intensity_sums.size
            ] += chunk_intensity_sums

        above_threshold = finite_intensity & (
            denominator_intensities > intensity_threshold
        )
        if np.any(above_threshold):
            chunk_positive = np.bincount(
                denominator_labels[above_threshold]
            )
            positive_counts[: chunk_positive.size] += chunk_positive.astype(
                np.uint64, copy=False
            )

    return (
        whole_cell_counts,
        denominator_counts,
        positive_counts,
        finite_intensity_counts,
        intensity_sums,
    )


def pixel_counts_by_mask_label(
    channel: np.ndarray,
    masks: np.ndarray,
    intensity_threshold: float,
    compartment: str = "all",
    inward_buffer_pixels: int = DEFAULT_INWARD_BUFFER_PIXELS,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count whole-cell, denominator, and positive pixels by mask label."""
    statistics = _pixel_statistics_by_mask_label(
        channel,
        masks,
        intensity_threshold,
        compartment=compartment,
        inward_buffer_pixels=inward_buffer_pixels,
        chunk_rows=chunk_rows,
    )
    return statistics[:3]


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
    nearest_search_radius: int = 75,
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
    if np.any(unresolved) and nearest_search_radius > 0:
        for row_index in np.flatnonzero(unresolved):
            if not np.isfinite(x[row_index]) or not np.isfinite(y[row_index]):
                continue
            center_x = int(round(x[row_index]))
            center_y = int(round(y[row_index]))
            x0 = max(center_x - nearest_search_radius, 0)
            x1 = min(center_x + nearest_search_radius + 1, masks.shape[1])
            y0 = max(center_y - nearest_search_radius, 0)
            y1 = min(center_y + nearest_search_radius + 1, masks.shape[0])
            if x0 >= x1 or y0 >= y1:
                continue

            window = np.asarray(masks[y0:y1, x0:x1])
            rows, columns = np.nonzero(window > 0)
            if not len(rows):
                continue
            candidate_labels = window[rows, columns].astype(
                np.int64, copy=False
            )
            candidate_valid = labels_exist(candidate_labels)
            if not np.any(candidate_valid):
                continue
            rows = rows[candidate_valid]
            columns = columns[candidate_valid]
            candidate_labels = candidate_labels[candidate_valid]
            distance_squared = (
                (x0 + columns - x[row_index]) ** 2
                + (y0 + rows - y[row_index]) ** 2
            )
            resolved[row_index] = candidate_labels[
                int(np.argmin(distance_squared))
            ]

    unresolved = ~labels_exist(resolved)
    if np.any(unresolved):
        raise ValueError(
            "Could not match "
            f"{int(np.count_nonzero(unresolved)):,} cell-data rows to mask labels. "
            "Provide numeric Cell ID/Object ID values or centroids within "
            f"{nearest_search_radius} pixels of their segmented cells."
        )
    return resolved


def cell_statistics_by_threshold(
    channel: np.ndarray,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    intensity_threshold: float,
    compartment: str = "all",
    inward_buffer_pixels: int = DEFAULT_INWARD_BUFFER_PIXELS,
    chunk_rows: int = 512,
) -> dict[str, np.ndarray]:
    """Return all-image per-cell intensity and positive-pixel statistics."""
    (
        whole_cell_counts,
        denominator_counts,
        positive_counts,
        finite_intensity_counts,
        intensity_sums,
    ) = _pixel_statistics_by_mask_label(
        channel,
        masks,
        intensity_threshold,
        compartment=compartment,
        inward_buffer_pixels=inward_buffer_pixels,
        chunk_rows=chunk_rows,
    )
    mask_labels = map_cell_rows_to_mask_labels(
        cell_data,
        masks,
        centroid_x,
        centroid_y,
        whole_cell_counts,
    )
    mean_intensity_by_label = np.divide(
        intensity_sums,
        finite_intensity_counts,
        out=np.full(intensity_sums.shape, np.nan, dtype=float),
        where=finite_intensity_counts > 0,
    )
    fractions_by_label = np.divide(
        positive_counts,
        denominator_counts,
        out=np.zeros_like(positive_counts, dtype=float),
        where=denominator_counts > 0,
    )
    return {
        "mask_label": mask_labels,
        "denominator_pixels": denominator_counts[mask_labels],
        "mean_intensity": mean_intensity_by_label[mask_labels],
        "positive_fraction": fractions_by_label[mask_labels],
    }


def select_automated_training_indices(
    positive_fractions: np.ndarray,
    positive_pixel_fraction: float,
    low_negative_count: int = 15,
    mid_negative_count: int = 10,
    positive_count: int = 25,
    top_positive_fraction: float = 0.30,
    fluorescence_values: np.ndarray | None = None,
    excluded_indices: np.ndarray | None = None,
    random_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Select balanced threshold-derived training rows for automation.

    Negative cells are ranked by mean fluorescence within the cells that fail
    the positive-pixel threshold. Samples are drawn without replacement from
    the lowest 50% and the 51st-to-80th percentile band. Positives are sampled
    without replacement from the top positive-pixel-fraction cells.
    """
    fractions = np.asarray(positive_fractions, dtype=float)
    if fractions.ndim != 1:
        raise ValueError("Positive-pixel fractions must be one-dimensional.")
    fluorescence = (
        fractions
        if fluorescence_values is None
        else np.asarray(fluorescence_values, dtype=float)
    )
    if fluorescence.ndim != 1 or fluorescence.shape != fractions.shape:
        raise ValueError(
            "Per-cell fluorescence values must match positive-pixel fractions."
        )
    if not 0.0 <= positive_pixel_fraction <= 1.0:
        raise ValueError("The positive-pixel fraction must be between 0 and 1.")
    if low_negative_count < 0 or mid_negative_count < 0 or positive_count < 0:
        raise ValueError("Automated training counts cannot be negative.")
    if not 0.0 < top_positive_fraction <= 1.0:
        raise ValueError("The top-positive fraction must be above 0 and at most 1.")

    excluded = np.zeros(fractions.size, dtype=bool)
    if excluded_indices is not None:
        excluded_indices = np.asarray(excluded_indices, dtype=np.int64)
        valid_excluded = excluded_indices[
            (excluded_indices >= 0) & (excluded_indices < fractions.size)
        ]
        excluded[valid_excluded] = True

    finite = np.isfinite(fractions)
    finite_fluorescence = finite & np.isfinite(fluorescence)
    threshold_negative = np.flatnonzero(
        finite_fluorescence & (fractions <= positive_pixel_fraction)
    )
    if threshold_negative.size == 0 and (low_negative_count or mid_negative_count):
        raise ValueError(
            "No cells fall below the automated positive-pixel threshold. "
            "Raise the intensity or positive-pixel threshold in Edit mode."
        )

    negative_order = threshold_negative[
        np.argsort(fluorescence[threshold_negative], kind="stable")
    ]
    low_stop = int(np.ceil(negative_order.size * 0.50))
    mid_stop = int(np.ceil(negative_order.size * 0.80))
    low_negative_pool = negative_order[:low_stop]
    mid_negative_pool = negative_order[low_stop:mid_stop]
    low_negative_pool = low_negative_pool[~excluded[low_negative_pool]]
    mid_negative_pool = mid_negative_pool[~excluded[mid_negative_pool]]

    if low_negative_pool.size < low_negative_count:
        raise ValueError(
            f"Automated phenotyping needs {low_negative_count} negative "
            "training cells from the lowest-fluorescence 50% of negatively "
            f"thresholded cells, but only {low_negative_pool.size} are available."
        )
    if mid_negative_pool.size < mid_negative_count:
        raise ValueError(
            f"Automated phenotyping needs {mid_negative_count} negative "
            "training cells from the 51st-to-80th percentile range of "
            "negatively thresholded cells, but only "
            f"{mid_negative_pool.size} are available."
        )

    rng = np.random.default_rng(random_seed)
    low_negative_indices = (
        np.sort(
            rng.choice(
                low_negative_pool,
                size=low_negative_count,
                replace=False,
            )
        )
        if low_negative_count
        else np.array([], dtype=np.int64)
    )
    mid_negative_indices = (
        np.sort(
            rng.choice(
                mid_negative_pool,
                size=mid_negative_count,
                replace=False,
            )
        )
        if mid_negative_count
        else np.array([], dtype=np.int64)
    )
    negative_indices = np.sort(
        np.concatenate((low_negative_indices, mid_negative_indices))
    )

    threshold_positive = np.flatnonzero(
        finite & (fractions > positive_pixel_fraction)
    )
    if threshold_positive.size == 0 and positive_count:
        raise ValueError(
            "No cells exceed the automated positive-pixel threshold. "
            "Lower the intensity or positive-pixel threshold in Edit mode."
        )
    positive_order = threshold_positive[
        np.argsort(fractions[threshold_positive], kind="stable")[::-1]
    ]
    top_count = int(np.ceil(positive_order.size * top_positive_fraction))
    top_positive = positive_order[:top_count]
    blocked = excluded.copy()
    blocked[negative_indices] = True
    positive_pool = top_positive[~blocked[top_positive]]
    if positive_pool.size < positive_count:
        raise ValueError(
            f"Automated phenotyping needs {positive_count} positive training "
            f"cells in the top {top_positive_fraction:.0%} of threshold-positive "
            f"cells, but only {positive_pool.size} are available."
        )

    positive_indices = (
        np.sort(
            rng.choice(positive_pool, size=positive_count, replace=False)
        )
        if positive_count
        else np.array([], dtype=np.int64)
    )
    return {
        "negative": negative_indices.astype(np.int64, copy=False),
        "negative_low": low_negative_indices.astype(np.int64, copy=False),
        "negative_mid": mid_negative_indices.astype(np.int64, copy=False),
        "negative_low_pool": low_negative_pool.astype(np.int64, copy=False),
        "negative_mid_pool": mid_negative_pool.astype(np.int64, copy=False),
        "positive": positive_indices.astype(np.int64, copy=False),
        "positive_pool": positive_pool.astype(np.int64, copy=False),
    }


def select_automated_refinement_indices(
    predicted_positive: np.ndarray,
    positive_probabilities: np.ndarray,
    fluorescence_values: np.ndarray,
    positive_count: int = 5,
    negative_count: int = 5,
    maximum_call_probability: float = 0.60,
    fluorescence_tail_fraction: float = 0.10,
    excluded_indices: np.ndarray | None = None,
    random_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Select low-confidence, fluorescence-discordant refinement cells.

    Five high-fluorescence cells from the top fluorescence decile of
    low-confidence negative model calls become positive training examples.
    Five low-fluorescence cells from the bottom fluorescence decile of
    low-confidence positive model calls become negative training examples.
    Existing or explicitly excluded training rows are never selected again.
    """
    calls = np.asarray(predicted_positive, dtype=bool)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    fluorescence = np.asarray(fluorescence_values, dtype=float)
    if calls.ndim != 1:
        raise ValueError("Model calls must be one-dimensional.")
    if probabilities.ndim != 1 or probabilities.shape != calls.shape:
        raise ValueError(
            "Positive probabilities must contain one value per model call."
        )
    if fluorescence.ndim != 1 or fluorescence.shape != calls.shape:
        raise ValueError(
            "Per-cell fluorescence values must contain one value per model call."
        )
    if positive_count < 0 or negative_count < 0:
        raise ValueError("Automated refinement counts cannot be negative.")
    if not 0.50 < maximum_call_probability <= 1.0:
        raise ValueError(
            "The maximum call probability must be above 0.50 and at most 1."
        )
    if not 0.0 < fluorescence_tail_fraction <= 1.0:
        raise ValueError(
            "The fluorescence-tail fraction must be above 0 and at most 1."
        )

    excluded = np.zeros(calls.size, dtype=bool)
    if excluded_indices is not None:
        excluded_indices = np.asarray(excluded_indices, dtype=np.int64)
        valid_excluded = excluded_indices[
            (excluded_indices >= 0) & (excluded_indices < calls.size)
        ]
        excluded[valid_excluded] = True

    valid = (
        np.isfinite(probabilities)
        & np.isfinite(fluorescence)
        & (probabilities >= 0.0)
        & (probabilities <= 1.0)
        & ~excluded
    )
    call_probabilities = np.where(calls, probabilities, 1.0 - probabilities)
    low_confidence = valid & (
        call_probabilities < maximum_call_probability
    )

    called_negative = np.flatnonzero(low_confidence & ~calls)
    called_positive = np.flatnonzero(low_confidence & calls)

    high_fluorescence_order = called_negative[
        np.argsort(fluorescence[called_negative], kind="stable")
    ]
    high_tail_count = int(np.ceil(
        high_fluorescence_order.size * fluorescence_tail_fraction
    ))
    high_fluorescence_pool = high_fluorescence_order[-high_tail_count:]

    low_fluorescence_order = called_positive[
        np.argsort(fluorescence[called_positive], kind="stable")
    ]
    low_tail_count = int(np.ceil(
        low_fluorescence_order.size * fluorescence_tail_fraction
    ))
    low_fluorescence_pool = low_fluorescence_order[:low_tail_count]

    if high_fluorescence_pool.size < positive_count:
        raise ValueError(
            "Automated refinement needs "
            f"{positive_count} cells from the top "
            f"{fluorescence_tail_fraction:.0%} of fluorescence among "
            "low-confidence negative calls, but only "
            f"{high_fluorescence_pool.size} are available."
        )
    if low_fluorescence_pool.size < negative_count:
        raise ValueError(
            "Automated refinement needs "
            f"{negative_count} cells from the bottom "
            f"{fluorescence_tail_fraction:.0%} of fluorescence among "
            "low-confidence positive calls, but only "
            f"{low_fluorescence_pool.size} are available."
        )

    rng = np.random.default_rng(random_seed)
    positive_indices = (
        np.sort(
            rng.choice(
                high_fluorescence_pool,
                size=positive_count,
                replace=False,
            )
        )
        if positive_count
        else np.array([], dtype=np.int64)
    )
    negative_indices = (
        np.sort(
            rng.choice(
                low_fluorescence_pool,
                size=negative_count,
                replace=False,
            )
        )
        if negative_count
        else np.array([], dtype=np.int64)
    )
    return {
        "positive": positive_indices.astype(np.int64, copy=False),
        "negative": negative_indices.astype(np.int64, copy=False),
        "positive_pool": high_fluorescence_pool.astype(
            np.int64, copy=False
        ),
        "negative_pool": low_fluorescence_pool.astype(
            np.int64, copy=False
        ),
    }


def phenotype_cells_by_threshold(
    channel: np.ndarray,
    masks: np.ndarray,
    cell_data: pd.DataFrame,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    intensity_threshold: float,
    positive_pixel_fraction: float,
    compartment: str = "all",
    inward_buffer_pixels: int = DEFAULT_INWARD_BUFFER_PIXELS,
    chunk_rows: int = 512,
) -> dict[str, np.ndarray]:
    """Assign cells using the fraction of their pixels above an intensity cutoff."""
    if not 0.0 <= positive_pixel_fraction <= 1.0:
        raise ValueError("The positive-pixel fraction must be between 0 and 1.")

    (
        whole_cell_counts,
        denominator_counts,
        positive_counts,
        finite_intensity_counts,
        intensity_sums,
    ) = _pixel_statistics_by_mask_label(
        channel,
        masks,
        intensity_threshold,
        compartment=compartment,
        inward_buffer_pixels=inward_buffer_pixels,
        chunk_rows=chunk_rows,
    )
    mask_labels = map_cell_rows_to_mask_labels(
        cell_data,
        masks,
        centroid_x,
        centroid_y,
        whole_cell_counts,
    )
    fractions_by_label = np.divide(
        positive_counts,
        denominator_counts,
        out=np.zeros_like(positive_counts, dtype=float),
        where=denominator_counts > 0,
    )
    fractions = fractions_by_label[mask_labels]
    mean_intensity_by_label = np.divide(
        intensity_sums,
        finite_intensity_counts,
        out=np.full(intensity_sums.shape, np.nan, dtype=float),
        where=finite_intensity_counts > 0,
    )
    denominator_pixels = denominator_counts[mask_labels]
    return {
        "mask_label": mask_labels,
        "denominator_pixels": denominator_pixels,
        "mean_intensity": mean_intensity_by_label[mask_labels],
        "positive_fraction": fractions,
        "positive": (
            (denominator_pixels > 0)
            & (fractions > positive_pixel_fraction)
        ),
    }
