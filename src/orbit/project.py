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


def relocated_reference(
    missing_path: str | Path,
    relocation_roots: dict[Path, Path],
) -> str | None:
    """Resolve a missing asset through previously selected directory moves."""
    missing = Path(missing_path).expanduser().resolve()
    roots = sorted(
        (
            (
                Path(old_root).expanduser().resolve(),
                Path(new_root).expanduser().resolve(),
            )
            for old_root, new_root in relocation_roots.items()
        ),
        key=lambda pair: len(pair[0].parts),
        reverse=True,
    )
    for old_root, new_root in roots:
        try:
            relative = missing.relative_to(old_root)
        except ValueError:
            continue
        candidate = new_root / relative
        if candidate.exists():
            return str(candidate.resolve())
    return None


def remember_relocation(
    missing_path: str | Path,
    selected_path: str | Path,
    relocation_roots: dict[Path, Path],
) -> None:
    """Remember compatible parent moves so sibling assets can be found.

    The exact containing directory is always recorded. Matching trailing
    directory names are then walked upward, allowing a selected path such as
    ``new/study/images/slide.tif`` to also relocate assets stored under
    ``old/study/segmentations`` without recursively searching the filesystem.
    """
    old_parent = Path(missing_path).expanduser().resolve().parent
    new_parent = Path(selected_path).expanduser().resolve().parent
    relocation_roots[old_parent] = new_parent

    while (
        old_parent.name
        and new_parent.name
        and old_parent.name.casefold() == new_parent.name.casefold()
    ):
        relocation_roots[old_parent] = new_parent
        next_old = old_parent.parent
        next_new = new_parent.parent
        if next_old == old_parent or next_new == new_parent:
            break
        old_parent, new_parent = next_old, next_new


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
    "relocated_reference",
    "remember_relocation",
    "resolve_reference",
    "write_provenance",
]
