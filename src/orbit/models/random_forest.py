"""Construction, fitting, and inference for ORBIT random-forest models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


MODEL_FORMAT = "ORBIT phenotype model"
MODEL_VERSION = 1
RANDOM_FOREST_ALGORITHM = "RandomForestClassifier"
RANDOM_FOREST_RANDOM_SEED = 42
RANDOM_FOREST_ESTIMATORS = 300
DEFAULT_POSITIVE_PROBABILITY_THRESHOLD = 0.50


def fit_random_forest(
    training_data: pd.DataFrame,
    targets: Sequence[int] | np.ndarray,
    feature_columns: Sequence[str],
    random_seed: int = RANDOM_FOREST_RANDOM_SEED,
) -> tuple[Pipeline, list[str]]:
    """Fit ORBIT's classifier and return it with the usable feature names."""
    target_array = np.asarray(targets, dtype=np.uint8)
    if target_array.ndim != 1 or target_array.size != len(training_data):
        raise ValueError(
            "Training targets must contain one value per training cell."
        )
    if set(target_array.tolist()) != {0, 1}:
        raise ValueError(
            "Random-forest training requires both positive and negative cells."
        )

    features = list(feature_columns)
    numeric_data = training_data[features].apply(
        pd.to_numeric, errors="coerce"
    )
    usable_features = [
        column
        for column in features
        if numeric_data[column].notna().any()
        and numeric_data[column].nunique(dropna=True) > 1
    ]
    if not usable_features:
        raise ValueError(
            "The training cells do not vary in any numeric feature column."
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            class_weight="balanced",
            random_state=int(random_seed),
            n_jobs=-1,
        )),
    ])
    pipeline.fit(numeric_data[usable_features], target_array)
    return pipeline, usable_features


def model_calls_and_positive_probabilities(
    pipeline: Pipeline,
    measurements: pd.DataFrame,
    positive_probability_threshold: float = DEFAULT_POSITIVE_PROBABILITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probability-thresholded calls and positive-class probabilities."""
    decision_threshold = float(positive_probability_threshold)
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("The positive probability threshold must be between 0 and 1.")
    probabilities = np.asarray(
        pipeline.predict_proba(measurements), dtype=float
    )
    classes = np.asarray(pipeline.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if positive_columns.size != 1:
        raise ValueError(
            "The phenotype model does not expose a single positive class."
        )
    positive_probability = np.clip(
        probabilities[:, int(positive_columns[0])], 0.0, 1.0
    )
    return positive_probability >= decision_threshold, positive_probability
