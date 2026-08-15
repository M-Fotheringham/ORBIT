"""Rules for selecting cell measurements used by phenotype models."""

from __future__ import annotations

import re


_PIXEL_UNIT_SUFFIX = re.compile(
    r"(?:^|\s)(?:px|pixels?)(?:\s*(?:2|\^2|²))?$",
    flags=re.IGNORECASE,
)


def is_pixel_unit_measurement(column: object) -> bool:
    """Return whether a column is a pixel-unit morphology measurement.

    ORBIT exports pixel measurements for compatibility and micron measurements
    for physically meaningful analysis.  Pixel values are identified only when
    the unit occurs at the end of the column name, so fluorescence features such
    as ``Positive pixel %`` are not removed accidentally.
    """
    name = str(column).strip().replace("_", " ").replace("-", " ")
    name = re.sub(r"[()\[\]{}]", " ", name)
    name = " ".join(name.split())
    return _PIXEL_UNIT_SUFFIX.search(name) is not None


__all__ = ["is_pixel_unit_measurement"]
