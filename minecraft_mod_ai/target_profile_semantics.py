from __future__ import annotations

"""Canonical target-version semantics shared by all Fabric execution boundaries.

This module owns only version-derived facts. Provider coordinates still come from the
validated target receipt; callers must not infer Fabric/Loader/API versions here.
"""

import re
from typing import Any

_NATIVE_NAME_MIN_VERSION = (26, 1)
_JAVA_21_MIN_VERSION = (1, 20, 5)


def minecraft_version_tuple(value: Any) -> tuple[int, ...]:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][A-Za-z0-9._-]+)?", text)
    if not match:
        raise ValueError(
            f"TARGET_MINECRAFT_VERSION: unsupported or unparseable Minecraft version {text!r}."
        )
    parts = [int(match.group(1)), int(match.group(2))]
    if match.group(3) is not None:
        parts.append(int(match.group(3)))
    return tuple(parts)


def uses_native_names(value: Any) -> bool:
    version = minecraft_version_tuple(value)
    return version[:2] >= _NATIVE_NAME_MIN_VERSION


def mappings_applicable(value: Any) -> bool:
    return not uses_native_names(value)


def minimum_java_major(value: Any) -> int | None:
    version = minecraft_version_tuple(value)
    if version[:2] >= _NATIVE_NAME_MIN_VERSION:
        return 25
    if version and version[0] == 1:
        padded = version + (0,) * (3 - len(version))
        if padded[:3] >= _JAVA_21_MIN_VERSION:
            return 21
    return None


__all__ = [
    "mappings_applicable",
    "minecraft_version_tuple",
    "minimum_java_major",
    "uses_native_names",
]
