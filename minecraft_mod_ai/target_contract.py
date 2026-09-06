from __future__ import annotations

"""Single source of truth for Minecraft target coordinate semantics.

Provider discovery owns *which* exact Fabric/Loader/API coordinates exist.  This module
owns the cross-stage contract for the target identity itself: Minecraft version, loader,
naming regime, mappings applicability/value, and minimum Java semantics.  Planning,
generation, and coder handoff must call this module instead of rebuilding those rules.
"""

import re
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Any

_NATIVE_NAME_MIN_VERSION = (26, 1)
_JAVA_21_MIN_VERSION = (1, 20, 5)


class TargetContractError(ValueError):
    """Raised when target coordinates contradict the canonical target contract."""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def minecraft_version_tuple(value: Any) -> tuple[int, ...]:
    text = _text(value)
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:[-+][A-Za-z0-9._-]+)?", text)
    if not match:
        raise TargetContractError(
            f"TARGET_MINECRAFT_VERSION: unsupported or unparseable Minecraft version {text!r}."
        )
    parts = [int(match.group(1)), int(match.group(2))]
    if match.group(3) is not None:
        parts.append(int(match.group(3)))
    return tuple(parts)


def uses_native_names(value: Any) -> bool:
    return minecraft_version_tuple(value)[:2] >= _NATIVE_NAME_MIN_VERSION


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


def mapping_value(target: Mapping[str, Any]) -> str:
    """Read the canonical mapping coordinate from a target-shaped mapping.

    ``mappings`` may be either the canonical scalar used at generation boundaries or the
    provider receipt object ``{"kind": ..., "version": ...}``.  Legacy aliases are read
    only to ingest existing provider receipts; callers must not invent a mapping from an
    unrelated field such as ``source_api_family``.
    """

    raw = target.get("mappings")
    if isinstance(raw, Mapping):
        return _text(raw.get("version"))
    if raw is not None:
        return _text(raw)
    return _text(target.get("mappings_version") or target.get("yarn_mappings"))


@dataclass(frozen=True)
class TargetCoordinates:
    minecraft_version: str
    loader: str
    mappings: str
    mappings_applicable: bool
    naming_regime: str
    minimum_java_major: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_target_coordinates(
    minecraft_version: Any,
    loader: Any,
    mappings: Any = "",
    *,
    declared_mappings_applicable: Any | None = None,
) -> TargetCoordinates:
    """Validate the target identity used by every planning/generation boundary.

    Mappings are required iff the Minecraft target uses mapped/obfuscated names.  Native
    targets require an empty mapping coordinate.  A caller-provided applicability flag is
    accepted only when it agrees with the version-derived canonical fact.
    """

    version = _text(minecraft_version)
    loader_id = _text(loader).casefold()
    mapping = _text(mappings)
    if not version:
        raise TargetContractError("TARGET_MINECRAFT_VERSION: Minecraft version is required.")
    if not loader_id:
        raise TargetContractError("TARGET_LOADER: loader is required.")

    applicable = mappings_applicable(version)
    if declared_mappings_applicable is not None:
        if type(declared_mappings_applicable) is not bool:
            raise TargetContractError(
                "TARGET_MAPPINGS_APPLICABLE: declared applicability must be boolean."
            )
        if declared_mappings_applicable is not applicable:
            raise TargetContractError(
                "TARGET_MAPPINGS_APPLICABLE: declared applicability contradicts the "
                "Minecraft-version naming regime."
            )

    if applicable and not mapping:
        raise TargetContractError(
            "TARGET_MAPPINGS_REQUIRED: mappings are required for this mapped/obfuscated target."
        )
    if not applicable and mapping:
        raise TargetContractError(
            "TARGET_MAPPINGS_INAPPLICABLE: native/unobfuscated targets must not carry a "
            "legacy mapping coordinate."
        )

    return TargetCoordinates(
        minecraft_version=version,
        loader=loader_id,
        mappings=mapping,
        mappings_applicable=applicable,
        naming_regime="mapped_obfuscated" if applicable else "native_unobfuscated",
        minimum_java_major=minimum_java_major(version),
    )


def target_coordinates_from_mapping(target: Mapping[str, Any]) -> TargetCoordinates:
    regime = target.get("naming_regime")
    declared: Any | None = None
    if isinstance(regime, Mapping) and "mappings_applicable" in regime:
        declared = regime.get("mappings_applicable")
    elif "mappings_applicable" in target:
        declared = target.get("mappings_applicable")
    return validate_target_coordinates(
        target.get("minecraft_version"),
        target.get("loader"),
        mapping_value(target),
        declared_mappings_applicable=declared,
    )


__all__ = [
    "TargetContractError",
    "TargetCoordinates",
    "mapping_value",
    "mappings_applicable",
    "minecraft_version_tuple",
    "minimum_java_major",
    "target_coordinates_from_mapping",
    "uses_native_names",
    "validate_target_coordinates",
]
