"""Portable ORBIT project paths and reproducible export provenance."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def orbit_version() -> str:
    try:
        return version("orbit")
    except PackageNotFoundError:
        return "unknown"


def portable_reference(path: str | Path | None, project_path: str | Path) -> str | None:
    """Represent a project asset relative to the project whenever possible."""
    if not path:
        return None
    asset = Path(path).expanduser().resolve()
    base = Path(project_path).expanduser().resolve().parent
    try:
        return os.path.relpath(asset, base)
    except ValueError:  # Different Windows drives cannot be made relative.
        return str(asset)


def resolve_reference(path: str | Path | None, project_path: str | Path) -> str | None:
    """Resolve an absolute or project-relative asset reference."""
    if not path:
        return None
    reference = Path(path).expanduser()
    if not reference.is_absolute():
        reference = Path(project_path).expanduser().resolve().parent / reference
    return str(reference.resolve())


def load_project_document(project_path: str | Path) -> dict[str, Any]:
    with Path(project_path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("format") != "ORBIT phenotype training session":
        raise ValueError("The selected file is not an ORBIT training session.")
    return document


def export_provenance(
    *,
    method: str,
    phenotype_name: str,
    decision_threshold: float | None,
    feature_columns: list[str] | tuple[str, ...],
    images: list[dict[str, Any]],
    model_bundle: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable audit record for a phenotype export."""
    model = model_bundle or {}
    return {
        "format": "ORBIT phenotype export provenance",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "orbit_version": orbit_version(),
        "method": str(method),
        "phenotype_name": str(phenotype_name),
        "positive_probability_threshold": decision_threshold,
        "feature_columns": list(feature_columns),
        "algorithm": model.get("algorithm"),
        "model_format": model.get("format"),
        "model_version": model.get("version"),
        "training_samples": model.get("training_samples"),
        "automated": model.get("automated"),
        "settings": dict(settings or {}),
        "images": images,
    }


def write_provenance(data_path: str | Path, provenance: dict[str, Any]) -> Path:
    """Write ``<export>.provenance.json`` next to a cell-table export."""
    data_path = Path(data_path)
    output = data_path.with_name(f"{data_path.stem}.provenance.json")
    temporary = output.with_name(f"{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    temporary.replace(output)
    return output


__all__ = [
    "export_provenance",
    "load_project_document",
    "orbit_version",
    "portable_reference",
    "resolve_reference",
    "write_provenance",
]
