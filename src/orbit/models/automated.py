"""Deterministic training-cell selection for automated phenotyping."""

from __future__ import annotations

import numpy as np


AUTOMATED_RANDOM_SEED = 42
AUTOMATED_INTENSITY_SLIDER_VALUE = 660
AUTOMATED_POSITIVE_PIXEL_PERCENT = 15
AUTOMATED_NEGATIVE_TRAINING_COUNT = 25
AUTOMATED_LOW_NEGATIVE_TRAINING_COUNT = 15
AUTOMATED_MID_NEGATIVE_TRAINING_COUNT = 10
AUTOMATED_POSITIVE_TRAINING_COUNT = 25
AUTOMATED_TOP_POSITIVE_FRACTION = 0.30
AUTOMATED_REFINEMENT_TRAINING_COUNT = 5
AUTOMATED_REFINEMENT_MAX_CALL_PROBABILITY = 0.60
AUTOMATED_SECOND_REFINEMENT_MAX_CALL_PROBABILITY = 0.55
AUTOMATED_REFINEMENT_FLUORESCENCE_FRACTION = 0.10


def random_seed_for_stage(
    stage: int,
    base_seed: int = AUTOMATED_RANDOM_SEED,
) -> int:
    """Derive a stable seed for one stage of the automated workflow."""
    if int(stage) != stage or stage < 0:
        raise ValueError("The automated-phenotyping stage must be non-negative.")
    modulus = np.iinfo(np.uint32).max + 1
    return (int(base_seed) + int(stage)) % modulus


def _selection_rng(random_seed: int | None) -> np.random.Generator:
    """Return a deterministic generator, including when no seed is supplied."""
    seed = AUTOMATED_RANDOM_SEED if random_seed is None else int(random_seed)
    return np.random.default_rng(seed)


def _excluded_rows(size: int, excluded_indices: np.ndarray | None) -> np.ndarray:
    excluded = np.zeros(size, dtype=bool)
    if excluded_indices is None:
        return excluded
    indices = np.asarray(excluded_indices, dtype=np.int64)
    valid = indices[(indices >= 0) & (indices < size)]
    excluded[valid] = True
    return excluded


def select_automated_training_indices(
    positive_fractions: np.ndarray,
    positive_pixel_fraction: float,
    low_negative_count: int = 15,
    mid_negative_count: int = 10,
    positive_count: int = 25,
    top_positive_fraction: float = 0.30,
    fluorescence_values: np.ndarray | None = None,
    excluded_indices: np.ndarray | None = None,
    random_seed: int | None = AUTOMATED_RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Select the deterministic threshold-derived initial training rows."""
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
        raise ValueError(
            "The top-positive fraction must be above 0 and at most 1."
        )

    excluded = _excluded_rows(fractions.size, excluded_indices)
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

    rng = _selection_rng(random_seed)
    low_negative_indices = (
        np.sort(rng.choice(
            low_negative_pool, size=low_negative_count, replace=False
        ))
        if low_negative_count
        else np.array([], dtype=np.int64)
    )
    mid_negative_indices = (
        np.sort(rng.choice(
            mid_negative_pool, size=mid_negative_count, replace=False
        ))
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
        np.sort(rng.choice(
            positive_pool, size=positive_count, replace=False
        ))
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
    random_seed: int | None = AUTOMATED_RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Select deterministic low-confidence refinement training rows."""
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

    excluded = _excluded_rows(calls.size, excluded_indices)
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

    rng = _selection_rng(random_seed)
    positive_indices = (
        np.sort(rng.choice(
            high_fluorescence_pool, size=positive_count, replace=False
        ))
        if positive_count
        else np.array([], dtype=np.int64)
    )
    negative_indices = (
        np.sort(rng.choice(
            low_fluorescence_pool, size=negative_count, replace=False
        ))
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
