"""Machine-learning models used by ORBIT."""

from orbit.models.automated import (
    AUTOMATED_RANDOM_SEED,
    random_seed_for_stage,
    select_automated_refinement_indices,
    select_automated_training_indices,
)
from orbit.models.random_forest import (
    DEFAULT_POSITIVE_PROBABILITY_THRESHOLD,
    MODEL_FORMAT,
    MODEL_VERSION,
    RANDOM_FOREST_ALGORITHM,
    RANDOM_FOREST_RANDOM_SEED,
    fit_random_forest,
    model_calls_and_positive_probabilities,
)

_CELLPOSE_EXPORTS = {
    "CELLPOSE_SAM_MODEL",
    "DEFAULT_POSITIVE_PROBABILITY_THRESHOLD",
    "build_cellpose_input",
    "dapi_channel_name",
    "membrane_marker_names",
    "segment_project_image_paths",
    "segment_project_images",
}


def __getattr__(name):
    """Load Cellpose dependencies only when segmentation APIs are requested."""
    if name in _CELLPOSE_EXPORTS:
        from orbit.models import cellpose_segmentation

        return getattr(cellpose_segmentation, name)
    raise AttributeError(name)

__all__ = [
    "AUTOMATED_RANDOM_SEED",
    "CELLPOSE_SAM_MODEL",
    "MODEL_FORMAT",
    "MODEL_VERSION",
    "RANDOM_FOREST_ALGORITHM",
    "RANDOM_FOREST_RANDOM_SEED",
    "build_cellpose_input",
    "dapi_channel_name",
    "fit_random_forest",
    "membrane_marker_names",
    "model_calls_and_positive_probabilities",
    "random_seed_for_stage",
    "segment_project_image_paths",
    "segment_project_images",
    "select_automated_refinement_indices",
    "select_automated_training_indices",
]
