"""Compile explicit authored Block hardness/resistance into Java registration.

The Fabric block registry template exposes a host-owned block_properties anchor.
Only approved finite numeric settings may be inserted; never infer behavior or
emit fake passes for missing gameplay features.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def authored_block_strength(config: Mapping[str, object]) -> tuple[float, float] | None:
    hardness = config.get("hardness")
    resistance = config.get("resistance")
    if hardness is None and resistance is None:
        return None
    if hardness is None or resistance is None:
        raise ValueError("BLOCK_PROPERTIES_INCOMPLETE: both hardness and resistance required")
    for name, value in (("hardness", hardness), ("resistance", resistance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("BLOCK_PROPERTY_NOT_NUMERIC:" + name)
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1000000:
            raise ValueError("BLOCK_PROPERTY_OUT_OF_RANGE:" + name)
    return float(hardness), float(resistance)


def lower_block_strength(rendered: str, registry_path: str, values: tuple[float, float]) -> str:
    marker = "/* MMM:block_properties:" + registry_path + " */"
    if not registry_path or rendered.count(marker) != 1:
        raise ValueError("BLOCK_PROPERTIES_ANCHOR_MISMATCH:" + registry_path)
    hardness, resistance = values
    if not all(math.isfinite(x) and 0 <= x <= 1000000 for x in values):
        raise ValueError("BLOCK_PROPERTY_OUT_OF_RANGE")
    return rendered.replace(marker, f".strength({hardness:g}F, {resistance:g}F)", 1)
