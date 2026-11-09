"""Construction, fitting, and inference for ORBIT random-forest models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import ydf


# YDF 0.8's default logging inspects stdout, which is absent in the
# windowed Windows application. Keep all ORBIT model operations console-free.
ydf.verbose(0)


MODEL_FORMAT = "ORBIT phenotype model"
MODEL_VERSION = 1
RANDOM_FOREST_ALGORITHM = "YDF Random Forest"
RANDOM_FOREST_RANDOM_SEED = 42
RANDOM_FOREST_ESTIMATORS = 300
DEFAULT_POSITIVE_PROBABILITY_THRESHOLD = 0.50


def _prepare_features(
    measurements: pd.DataFrame, feature_medians: pd.Series
) -> dict[str, np.ndarray]:
    numeric_data = measurements[list(feature_medians.index)].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(feature_medians)
    # Stable internal names support arbitrary measurement-column names and
    # prevent collisions with YDF's label and example-weight columns.
    return {
        f"feature_{index}": numeric_data[column].to_numpy(dtype=np.float32)
        for index, column in enumerate(feature_medians.index)
    }


class YDFRandomForestClassifier:
    """Keep ORBIT's prediction and joblib interfaces around a native YDF forest."""

    def __init__(self, model: ydf.RandomForestModel, feature_medians: pd.Series):
        self._model = model
        self.feature_medians = feature_medians.copy()

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([0, 1], dtype=np.uint8)

    def predict_proba(self, measurements: pd.DataFrame) -> np.ndarray:
        features = _prepare_features(measurements, self.feature_medians)
        if len(measurements) == 0:
            return np.empty((0, 2), dtype=float)
        probability = np.asarray(self._model.predict(features), dtype=float)
        # YDF returns the probability of its second label class, whose order
        # can depend on class frequency rather than the numeric label value.
        if self._model.label_classes()[1] != "1":
            probability = 1.0 - probability
        probability = np.clip(probability, 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))

    def predict(self, measurements: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(measurements), axis=1)]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = self._model.serialize()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._model = ydf.deserialize_model(state["_model"])


def fit_random_forest(
    training_data: pd.DataFrame,
    targets: Sequence[int] | np.ndarray,
    feature_columns: Sequence[str],
    random_seed: int = RANDOM_FOREST_RANDOM_SEED,
) -> tuple[YDFRandomForestClassifier, list[str]]:
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

    feature_medians = numeric_data[usable_features].median()
    # Prepare training and inference data with the same fitted medians.
    dataset = _prepare_features(numeric_data, feature_medians)
    feature_definitions = [
        (name, ydf.Semantic.NUMERICAL) for name in dataset
    ]
    dataset["label"] = target_array.astype(str)
    class_counts = np.bincount(target_array, minlength=2)
    # YDF 0.8 supports example weights. These reproduce sklearn's global
    # class_weight="balanced": n_samples / (n_classes * class_count).
    dataset["weight"] = (
        len(target_array) / (2.0 * class_counts[target_array])
    ).astype(np.float32)
    model = ydf.RandomForestLearner(
        label="label",
        weights="weight",
        task=ydf.Task.CLASSIFICATION,
        features=feature_definitions,
        num_trees=RANDOM_FOREST_ESTIMATORS,
        random_seed=int(random_seed),
        min_examples=1,
        max_depth=-1,
        winner_take_all=False,
        compute_oob_performances=False,
    ).train(dataset, verbose=0)
    return YDFRandomForestClassifier(model, feature_medians), usable_features


def model_calls_and_positive_probabilities(
    pipeline: YDFRandomForestClassifier,
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
