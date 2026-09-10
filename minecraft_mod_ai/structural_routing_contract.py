from __future__ import annotations

"""Structural artifact routing contract.

This module intentionally contains no game-name, feature-name, domain, or semantic
classification. Artifact selection consumes only explicit canonical artifact kinds and
typed structural obligations.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CANONICAL_ARTIFACT_KINDS = (
    "item", "block", "block_entity", "entity", "mob", "attribute", "effect", "component",
    "inventory", "menu", "screen", "hud", "command", "event", "keybind", "network_payload",
    "saved_data", "recipe", "loot", "tag", "advancement", "worldgen", "structure", "biome",
    "dimension", "particle", "sound", "model", "texture", "animation", "language", "datagen",
)
_ARTIFACT_ORDER = {kind: index for index, kind in enumerate(CANONICAL_ARTIFACT_KINDS)}
_ARTIFACT_SET = frozenset(CANONICAL_ARTIFACT_KINDS)
ARTIFACT_DEPENDENCIES = {
    "block_entity": ("block",), "mob": ("entity",), "menu": ("inventory",),
    "screen": ("menu",), "structure": ("worldgen",), "biome": ("worldgen",),
    "dimension": ("worldgen",),
}
FLAG_TO_ARTIFACT = {
    "persistent_state": "saved_data", "networking": "network_payload", "inventory": "inventory",
    "menu": "menu", "screen": "screen", "hud": "hud", "recipe": "recipe", "loot": "loot",
    "worldgen": "worldgen", "structure": "structure", "biome": "biome", "dimension": "dimension",
    "particle": "particle", "sound": "sound", "model": "model", "texture": "texture",
    "animation": "animation", "language": "language", "datagen": "datagen",
}


@dataclass(frozen=True)
class StructuralArtifactRequirements:
    explicit_artifacts: tuple[str, ...] = ()
    persistent_state: bool = False
    networking: bool = False
    inventory: bool = False
    menu: bool = False
    screen: bool = False
    hud: bool = False
    recipe: bool = False
    loot: bool = False
    worldgen: bool = False
    structure: bool = False
    biome: bool = False
    dimension: bool = False
    particle: bool = False
    sound: bool = False
    model: bool = False
    texture: bool = False
    animation: bool = False
    language: bool = False
    datagen: bool = False


@dataclass(frozen=True)
class StructuralArtifactPlan:
    requested: tuple[str, ...]
    detected: tuple[str, ...]
    expanded: tuple[str, ...]
    proof: tuple[str, ...]

    @property
    def artifact_kinds(self) -> tuple[str, ...]:
        return self.expanded


def validate_artifact_kinds(kinds: Iterable[str]) -> tuple[str, ...]:
    normalized = set()
    for raw in kinds:
        if not isinstance(raw, str):
            raise TypeError("TEMPLATE_ARTIFACT: artifact kind must be a string")
        kind = raw.strip()
        if kind not in _ARTIFACT_SET:
            raise ValueError(f"TEMPLATE_ARTIFACT: unknown canonical artifact {kind!r}")
        normalized.add(kind)
    return tuple(sorted(normalized, key=_ARTIFACT_ORDER.__getitem__))


def requirements_from_record(record: Mapping[str, Any]) -> StructuralArtifactRequirements:
    if not isinstance(record, Mapping):
        raise TypeError("TEMPLATE_STRUCTURE: record must be a mapping")
    raw = record.get("explicit_artifacts", ())
    if raw is None:
        raw = ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError("TEMPLATE_STRUCTURE: explicit_artifacts must be a sequence")
    values = {"explicit_artifacts": validate_artifact_kinds(raw)}
    for key in FLAG_TO_ARTIFACT:
        value = record.get(key, False)
        if value is None:
            value = False
        if not isinstance(value, bool):
            raise TypeError(f"TEMPLATE_STRUCTURE: {key} must be boolean")
        values[key] = value
    return StructuralArtifactRequirements(**values)


def detect_artifacts(requirements: StructuralArtifactRequirements) -> tuple[str, ...]:
    kinds = list(requirements.explicit_artifacts)
    kinds.extend(artifact for field, artifact in FLAG_TO_ARTIFACT.items() if getattr(requirements, field))
    return validate_artifact_kinds(kinds)


def expand_artifact_dependencies(kinds: Iterable[str]) -> tuple[str, ...]:
    expanded = set(validate_artifact_kinds(kinds))
    pending = list(expanded)
    while pending:
        kind = pending.pop()
        for dependency in ARTIFACT_DEPENDENCIES.get(kind, ()):
            if dependency not in expanded:
                expanded.add(dependency)
                pending.append(dependency)
    return validate_artifact_kinds(expanded)


def build_artifact_plan(requirements: StructuralArtifactRequirements | Mapping[str, Any]) -> StructuralArtifactPlan:
    if isinstance(requirements, Mapping):
        requirements = requirements_from_record(requirements)
    if not isinstance(requirements, StructuralArtifactRequirements):
        raise TypeError("TEMPLATE_STRUCTURE: invalid requirements")
    requested = validate_artifact_kinds(requirements.explicit_artifacts)
    detected = detect_artifacts(requirements)
    expanded = expand_artifact_dependencies(detected)
    proof = tuple([f"explicit:{kind}" for kind in requested] + [
        f"structural:{field}->{artifact}" for field, artifact in FLAG_TO_ARTIFACT.items()
        if getattr(requirements, field)
    ] + [
        f"dependency:{kind}->{dependency}" for kind in detected
        for dependency in ARTIFACT_DEPENDENCIES.get(kind, ()) if dependency in expanded
    ])
    return StructuralArtifactPlan(requested, detected, expanded, proof)
