from __future__ import annotations

"""Deterministic Minecraft artifact detection from structural feature evidence.

Artifact selection deliberately ignores feature names, ids, descriptions, purposes and
behavior prose. Only explicit structural surfaces may select an artifact.
"""

from collections.abc import Iterable, Mapping
from typing import Any

CANONICAL_ARTIFACT_KINDS: tuple[str, ...] = (
    "item", "block", "block_entity", "entity", "mob", "attribute", "effect",
    "component", "inventory", "menu", "screen", "hud", "command", "event",
    "keybind", "network_payload", "saved_data", "recipe", "loot", "tag",
    "advancement", "worldgen", "structure", "biome", "dimension", "particle",
    "sound", "model", "texture", "animation", "language", "datagen",
)
_CANONICAL = frozenset(CANONICAL_ARTIFACT_KINDS)
_STRUCTURAL_KEYS = (
    "trigger", "state", "persistence", "networking", "ui", "resources", "assets",
    "external_interactions", "implementation_surfaces", "required_surfaces",
)
_TRIGGER_ARTIFACT = {"command": "command", "event": "event", "keybind": "keybind"}
_ARTIFACT_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "block_entity": ("block",),
    "mob": ("entity",),
    "screen": ("menu",),
    "structure": ("worldgen",),
    "biome": ("worldgen",),
    "dimension": ("worldgen",),
}


def _present(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        text = value.strip().casefold()
        return bool(text) and text not in {"none", "unknown", "false", "no"}
    if isinstance(value, Mapping):
        if "required" in value:
            return bool(value.get("required"))
        return any(_present(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_present(v) for v in value)
    return bool(value)


def _tokens(value: Any) -> tuple[str, ...]:
    """Extract explicit structural enum/surface tokens without parsing free-form prose."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip().casefold()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, nested in value.items():
            key_text = str(key).strip().casefold()
            if _present(nested):
                out.append(key_text)
            if isinstance(nested, (list, tuple, set, frozenset, Mapping)):
                out.extend(_tokens(nested))
        return tuple(dict.fromkeys(out))
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_tokens(item))
        return tuple(dict.fromkeys(out))
    return ()


def _canonical_surface_tokens(value: Any) -> tuple[str, ...]:
    return tuple(token for token in _tokens(value) if token in _CANONICAL)


def validate_artifact_kinds(artifacts: Iterable[Any]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(item).strip().casefold() for item in artifacts if str(item).strip()))
    invalid = tuple(item for item in normalized if item not in _CANONICAL)
    if invalid:
        raise ValueError(f"non-canonical Minecraft artifact kind(s): {', '.join(invalid)}")
    return tuple(item for item in CANONICAL_ARTIFACT_KINDS if item in normalized)


def structural_artifact_evidence(feature: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete and only routing input accepted by the detector."""
    return {key: feature.get(key) for key in _STRUCTURAL_KEYS}


def detect_artifacts(feature: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = structural_artifact_evidence(feature)
    selected: set[str] = set()
    trigger_tokens = _tokens(evidence["trigger"])
    for token, artifact in _TRIGGER_ARTIFACT.items():
        if token in trigger_tokens:
            selected.add(artifact)
    if _present(evidence["persistence"]):
        selected.add("saved_data")
    if _present(evidence["networking"]):
        selected.add("network_payload")
    selected.update(_canonical_surface_tokens(evidence["ui"]))
    selected.update(_canonical_surface_tokens(evidence["resources"]))
    selected.update(_canonical_surface_tokens(evidence["assets"]))
    selected.update(_canonical_surface_tokens(evidence["implementation_surfaces"]))
    selected.update(_canonical_surface_tokens(evidence["required_surfaces"]))
    selected.update(_canonical_surface_tokens(evidence["external_interactions"]))
    return validate_artifact_kinds(selected)


def expand_artifact_dependencies(artifacts: Iterable[Any]) -> tuple[str, ...]:
    selected = set(validate_artifact_kinds(artifacts))
    changed = True
    while changed:
        changed = False
        for artifact in tuple(selected):
            for dependency in _ARTIFACT_DEPENDENCIES.get(artifact, ()):
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return validate_artifact_kinds(selected)


def resolve_artifacts(feature: Mapping[str, Any]) -> tuple[str, ...]:
    return expand_artifact_dependencies(detect_artifacts(feature))


__all__ = [
    "CANONICAL_ARTIFACT_KINDS", "detect_artifacts", "expand_artifact_dependencies",
    "resolve_artifacts", "structural_artifact_evidence", "validate_artifact_kinds",
]
