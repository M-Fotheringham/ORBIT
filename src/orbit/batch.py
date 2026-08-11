"""Headless batch application of exported ORBIT phenotype models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from orbit.models.random_forest import (
    DEFAULT_POSITIVE_PROBABILITY_THRESHOLD,
    MODEL_FORMAT,
    model_calls_and_positive_probabilities,
)
from orbit.project import (
    export_provenance,
    load_project_document,
    resolve_reference,
    write_provenance,
)


def _project_cell_tables(project_path: Path) -> list[tuple[str, Path, dict]]:
    document = load_project_document(project_path)
    tables = []
    for entry in document.get("images", []):
        paths = entry.get("paths", {})
        image_path = resolve_reference(paths.get("image"), project_path)
        cell_path = resolve_reference(paths.get("cell_data"), project_path)
        if not image_path or not cell_path:
            raise ValueError("Every batch project image needs a cell-data path.")
        tables.append((Path(image_path).name, Path(cell_path), entry))
    if not tables:
        raise ValueError("The project contains no images.")
    return tables


def apply_model_batch(
    model_path: str | Path,
    project_path: str | Path,
    output_path: str | Path,
    decision_threshold: float | None = None,
) -> Path:
    """Apply one ORBIT model to every cell table in a portable project."""
    model_path, project_path, output_path = map(
        lambda value: Path(value).expanduser().resolve(),
        (model_path, project_path, output_path),
    )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        raise ValueError("The model file is not an ORBIT phenotype model.")
    features = list(bundle.get("feature_columns", ()))
    if not features:
        raise ValueError("The model does not contain feature-column metadata.")
    threshold = float(
        bundle.get("decision_threshold", DEFAULT_POSITIVE_PROBABILITY_THRESHOLD)
        if decision_threshold is None else decision_threshold
    )

    blocks = []
    source_records = []
    for image_name, cell_path, entry in _project_cell_tables(project_path):
        if not cell_path.is_file():
            raise FileNotFoundError(f"Cell data not found: {cell_path}")
        separator = "," if cell_path.suffix.lower() == ".csv" else "\t"
        cells = pd.read_csv(cell_path, sep=separator)
        missing = [column for column in features if column not in cells.columns]
        if missing:
            raise ValueError(
                f"{cell_path.name} is missing model feature(s): "
                + ", ".join(missing[:10])
            )
        calls, probabilities = model_calls_and_positive_probabilities(
            bundle["pipeline"],
            cells[features].apply(pd.to_numeric, errors="coerce"),
            positive_probability_threshold=threshold,
        )
        block = cells.copy()
        block.insert(0, "Image Name", image_name)
        phenotype = bundle.get("phenotype_name") or "Phenotype"
        block[f"{phenotype} Label"] = np.where(calls, "Positive", "Negative")
        block[f"{phenotype} Positive Probability"] = probabilities
        block["ORBIT Positive Probability Threshold"] = threshold
        block["ORBIT Label Source"] = "Headless Model"
        blocks.append(block)
        source_records.append({
            "image_name": image_name,
            "image_path": resolve_reference(entry.get("paths", {}).get("image"), project_path),
            "cell_data_path": str(cell_path),
            "segmentation_mask_path": resolve_reference(
                entry.get("paths", {}).get("segmentation_mask"), project_path
            ),
            "cell_count": len(cells),
            "segmentation": entry.get("cellpose_metadata"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "," if output_path.suffix.lower() == ".csv" else "\t"
    pd.concat(blocks, ignore_index=True, sort=False).to_csv(
        output_path, sep=separator, index=False
    )
    provenance = export_provenance(
        method="headless_random_forest",
        phenotype_name=bundle.get("phenotype_name") or "Phenotype",
        decision_threshold=threshold,
        feature_columns=features,
        images=source_records,
        model_bundle=bundle,
        settings={"project_path": str(project_path), "model_path": str(model_path)},
    )
    write_provenance(output_path, provenance)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit-batch",
        description="Apply an exported ORBIT phenotype model without opening the GUI.",
    )
    parser.add_argument("--model", required=True, help="Exported .orbitmodel file")
    parser.add_argument("--project", required=True, help="Portable .orbit.json project")
    parser.add_argument("--output", required=True, help="Output .tsv or .csv file")
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=None,
        help="Positive-call probability from 0 to 1 (model setting by default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    apply_model_batch(
        arguments.model,
        arguments.project,
        arguments.output,
        arguments.probability_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
