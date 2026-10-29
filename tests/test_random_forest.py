from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import ydf

from orbit.batch import apply_model_batch
from orbit.models.random_forest import (
    MODEL_FORMAT,
    MODEL_VERSION,
    RANDOM_FOREST_ALGORITHM,
    YDFRandomForestClassifier,
    fit_random_forest,
    model_calls_and_positive_probabilities,
)


class RandomForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = pd.DataFrame({
            "CD8: Membrane Mean": np.arange(60, dtype=float),
            "constant": 2.0,
            "empty": np.nan,
            "text": "not numeric",
        })
        cls.targets = (cls.training["CD8: Membrane Mean"] >= 20).astype(int)
        cls.pipeline, cls.features = fit_random_forest(
            cls.training, cls.targets, list(cls.training.columns)
        )

    def bundle(self):
        return {
            "format": MODEL_FORMAT,
            "version": MODEL_VERSION,
            "algorithm": RANDOM_FOREST_ALGORITHM,
            "pipeline": self.pipeline,
            "feature_columns": self.features,
            "phenotype_name": "CD8",
            "decision_threshold": 0.6,
        }

    def test_native_ydf_training_retains_feature_selection_and_positive_calls(self):
        self.assertIsInstance(self.pipeline, YDFRandomForestClassifier)
        self.assertIsInstance(self.pipeline._model, ydf.RandomForestModel)
        self.assertEqual(self.pipeline._model.num_trees(), 300)
        self.assertEqual(self.features, ["CD8: Membrane Mean"])
        samples = pd.DataFrame({self.features[0]: [0.0, 59.0]})
        calls, probabilities = model_calls_and_positive_probabilities(
            self.pipeline, samples
        )
        np.testing.assert_array_equal(calls, [False, True])
        self.assertLess(probabilities[0], 0.1)
        self.assertGreater(probabilities[1], 0.9)
        np.testing.assert_array_equal(self.pipeline.predict(samples), [0, 1])
        np.testing.assert_allclose(self.pipeline.predict_proba(samples).sum(axis=1), 1)

    def test_missing_values_use_training_medians_after_export_and_import(self):
        data = pd.DataFrame({"label": [0, "2", None, "invalid", 10, 12]})
        pipeline, features = fit_random_forest(data, [0, 0, 0, 1, 1, 1], ["label"])
        stream = io.BytesIO()
        joblib.dump(pipeline, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        measurements = pd.DataFrame({"label": [None, "invalid", 6.0]})
        probabilities = restored.predict_proba(measurements)
        np.testing.assert_array_equal(probabilities[0], probabilities[2])
        np.testing.assert_array_equal(probabilities[1], probabilities[2])
        np.testing.assert_array_equal(probabilities, pipeline.predict_proba(measurements))
        self.assertEqual(features, ["label"])

    def test_probability_threshold_is_inclusive_and_validated(self):
        samples = self.training[self.features]
        _, probabilities = model_calls_and_positive_probabilities(self.pipeline, samples)
        threshold = float(probabilities[20])
        calls, adjusted = model_calls_and_positive_probabilities(
            self.pipeline, samples, threshold
        )
        np.testing.assert_array_equal(calls, probabilities >= threshold)
        np.testing.assert_array_equal(adjusted, probabilities)
        self.assertTrue(calls[20])
        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                model_calls_and_positive_probabilities(self.pipeline, samples, threshold)

    def test_training_is_reproducible(self):
        other, features = fit_random_forest(
            self.training, self.targets, list(self.training.columns)
        )
        self.assertEqual(features, self.features)
        np.testing.assert_array_equal(
            self.pipeline.predict_proba(self.training), other.predict_proba(self.training)
        )

    def test_imbalanced_labels_receive_balanced_weight(self):
        data = pd.DataFrame({"signal": np.repeat([0.0, 1.0], 100)})
        targets = np.tile(np.repeat([0, 1], [90, 10]), 2)
        pipeline, _ = fit_random_forest(data, targets, ["signal"])
        probabilities = pipeline.predict_proba(pd.DataFrame({"signal": [0.0, 1.0]}))[:, 1]
        # Each feature value has the same 9:1 imbalance. Balancing the labels
        # should give both classes similar probability at either value.
        self.assertTrue(np.all((probabilities > 0.4) & (probabilities < 0.6)))

    def test_windowed_windows_runtime_can_train_predict_and_reload(self):
        script = '''
import sys
sys.stdout = sys.stderr = None
try:
    import io
    import joblib
    import pandas as pd
    from orbit.models.random_forest import fit_random_forest
    data = pd.DataFrame({"signal": range(20)})
    model, _ = fit_random_forest(data, [0] * 10 + [1] * 10, ["signal"])
    stream = io.BytesIO()
    joblib.dump(model, stream)
    stream.seek(0)
    restored = joblib.load(stream)
    assert restored.predict_proba(data).shape == (20, 2)
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_training_cells_and_empty_predictions_are_supported(self):
        data = pd.DataFrame({"weight": [0.0, 10.0]})
        pipeline, _ = fit_random_forest(data, [0, 1], ["weight"])
        self.assertEqual(pipeline.predict_proba(data).shape, (2, 2))
        calls, probabilities = model_calls_and_positive_probabilities(pipeline, data.iloc[:0])
        self.assertEqual(calls.shape, (0,))
        self.assertEqual(probabilities.shape, (0,))

    def test_invalid_training_still_fails(self):
        for targets in ([0], [[0, 1]], [0, 0], [1, 1], [0, 2]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                fit_random_forest(pd.DataFrame({"x": [0, 1]}), targets, ["x"])
        with self.assertRaisesRegex(ValueError, "do not vary"):
            fit_random_forest(pd.DataFrame({"x": [1, 1]}), [0, 1], ["x"])

    def test_legacy_sklearn_exports_still_predict(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        legacy = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]).fit(self.training[self.features], self.targets)
        stream = io.BytesIO()
        joblib.dump({**self.bundle(), "pipeline": legacy}, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        calls, probabilities = model_calls_and_positive_probabilities(
            restored["pipeline"], self.training[self.features], 0.6
        )
        np.testing.assert_array_equal(probabilities, legacy.predict_proba(self.training[self.features])[:, 1])
        np.testing.assert_array_equal(calls, probabilities >= 0.6)

    def test_exported_model_applies_to_multiple_batch_tables_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "phenotype.orbitmodel"
            joblib.dump(self.bundle(), model_path)
            images = []
            for index, values in enumerate(([0.0, 59.0], [None, 10.0, 50.0])):
                table = pd.DataFrame({self.features[0]: values, "Cell ID": np.arange(len(values))})
                table.to_csv(root / f"cells{index}.csv", index=False)
                images.append({"paths": {"image": f"image{index}.tif", "cell_data": f"cells{index}.csv"}})
            project_path = root / "project.orbit.json"
            project_path.write_text(json.dumps({
                "format": "ORBIT phenotype training session", "images": images
            }), encoding="utf-8")
            for override in (None, 0.0, 1.0):
                output = apply_model_batch(model_path, project_path, root / "calls.tsv", override)
                result = pd.read_csv(output, sep="\t")
                self.assertEqual(len(result), 5)
                self.assertIn("Cell ID", result.columns)
                self.assertEqual(result["Image Name"].nunique(), 2)
                threshold = 0.6 if override is None else override
                expected = self.pipeline.predict_proba(result[self.features])[:, 1]
                np.testing.assert_allclose(result["CD8 Positive Probability"], expected)
                np.testing.assert_array_equal(result["CD8 Label"] == "Positive", expected >= threshold)
                self.assertTrue((result["ORBIT Positive Probability Threshold"] == threshold).all())
                provenance = json.loads((root / "calls.provenance.json").read_text(encoding="utf-8"))
                self.assertEqual(provenance["algorithm"], RANDOM_FOREST_ALGORITHM)
                self.assertEqual(provenance["positive_probability_threshold"], threshold)


if __name__ == "__main__":
    unittest.main()
