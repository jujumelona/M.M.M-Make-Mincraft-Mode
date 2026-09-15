from __future__ import annotations

"""Deterministic Minecraft artifact selection from explicit structural requirements."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CANONICAL_ARTIFACT_KINDS: tuple[str, ...] = (
    "item", "block", "block_entity", "entity", "mob", "attribute", "effect",
    "component", "inventory", "menu", "screen", "hud", "command", "event",
    "keybind", "network_payload", "saved_data", "recipe", "loot", "tag",
    "advancement", "worldgen", "structure", "biome", "dimension", "particle",
    "sound", "model", "texture", "animation", "language", "datagen",
)
_CANONICAL = frozenset(CANONICAL_ARTIFACT_KINDS)

_ARTIFACT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "item_model": ("item", "model"),
    "block_model": ("block", "model"),
    "entity_model": ("entity", "model"),
    "blockstate": ("block", "model"),
    "loot_table": ("loot",),
    "lang": ("language",),
    "dimension_data": ("dimension",),
    "worldgen_data": ("worldgen",),
    "network": ("network_payload",),
    "packet": ("network_payload",),
    "persistence": ("saved_data",),
}

_ARTIFACT_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "block_entity": ("block",),
    "mob": ("entity",),
    "structure": ("worldgen",),
    "biome": ("worldgen",),
    "dimension": ("worldgen",),
    "recipe": ("datagen",),
    "loot": ("datagen",),
    "tag": ("datagen",),
    "advancement": ("datagen",),
    "language": ("datagen",),
}

_BRANCH_BY_ARTIFACT: Mapping[str, tuple[str, ...]] = {
    "item": ("needs_registry",),
    "block": ("needs_registry",),
    "block_entity": ("needs_registry",),
    "entity": ("needs_registry",),
    "mob": ("needs_registry",),
    "attribute": ("needs_registry",),
    "effect": ("needs_registry",),
    "component": ("needs_registry",),
    "menu": ("needs_registry",),
    "particle": ("needs_registry",),
    "sound": ("needs_registry",),
    "network_payload": ("needs_network",),
    "saved_data": ("needs_persistence",),
    "recipe": ("needs_datagen",),
    "loot": ("needs_datagen",),
    "tag": ("needs_datagen",),
    "advancement": ("needs_datagen",),
    "worldgen": ("needs_worldgen", "needs_datagen"),
    "structure": ("needs_worldgen", "needs_datagen"),
    "biome": ("needs_worldgen", "needs_datagen"),
    "dimension": ("needs_worldgen", "needs_datagen"),
    "model": ("needs_client_render",),
    "texture": ("needs_client_render",),
    "animation": ("needs_client_render",),
    "language": ("needs_client_render", "needs_datagen"),
    "screen": ("needs_client_render",),
    "hud": ("needs_client_render",),
    "keybind": ("needs_client_render",),
}


@dataclass(frozen=True)
class StructuralArtifactPlan:
    artifact_kinds: tuple[str, ...]
    branch_features: frozenset[str]
    unresolved_inputs: tuple[str, ...]


def _text(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _iter_declared_kinds(requirement: Mapping[str, Any]) -> Iterable[str]:
    obligations = requirement.get("artifact_obligations")
    if isinstance(obligations, Sequence) and not isinstance(obligations, (str, bytes, bytearray)):
        for row in obligations:
            kind = _text(row.get("kind")) if isinstance(row, Mapping) else _text(row)
            if kind:
                yield kind

    surfaces = requirement.get("implementation_surfaces")
    if isinstance(surfaces, Sequence) and not isinstance(surfaces, (str, bytes, bytearray)):
        for value in surfaces:
            kind = _text(value)
            if kind:
                yield kind

    structural = requirement.get("minecraft_structure")
    if isinstance(structural, Mapping):
        artifacts = structural.get("required_artifacts")
        if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes, bytearray)):
            for value in artifacts:
                kind = _text(value)
                if kind:
                    yield kind


def normalize_artifact_kind(kind: Any) -> tuple[str, ...]:
    normalized = _text(kind)
    if normalized in _CANONICAL:
        return (normalized,)
    return _ARTIFACT_ALIASES.get(normalized, ())


def validate_artifact_kinds(kinds: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for value in kinds:
        normalized = _text(value)
        if normalized not in _CANONICAL:
            raise ValueError(f"STRUCTURAL_ARTIFACT_KIND: unsupported canonical artifact {value!r}")
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def expand_artifact_dependencies(kinds: Iterable[Any]) -> tuple[str, ...]:
    roots = list(validate_artifact_kinds(kinds))
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(kind: str) -> None:
        if kind in resolved:
            return
        if kind in visiting:
            raise ValueError(f"STRUCTURAL_ARTIFACT_DEPENDENCY: cycle at {kind}")
        visiting.add(kind)
        for dependency in _ARTIFACT_DEPENDENCIES.get(kind, ()):
            visit(dependency)
        visiting.remove(kind)
        resolved.append(kind)

    for kind in roots:
        visit(kind)
    return tuple(resolved)


def branch_features_for_artifacts(kinds: Iterable[Any]) -> frozenset[str]:
    features: set[str] = set()
    for kind in validate_artifact_kinds(kinds):
        features.update(_BRANCH_BY_ARTIFACT.get(kind, ()))
    return frozenset(features)


def detect_structural_artifacts(requirement: Mapping[str, Any]) -> StructuralArtifactPlan:
    """Return a name-invariant artifact plan.

    Capability IDs, requirement IDs, statements and prompt text are intentionally
    not read by this function.
    """
    direct: list[str] = []
    unresolved: list[str] = []
    for raw_kind in _iter_declared_kinds(requirement):
        mapped = normalize_artifact_kind(raw_kind)
        if not mapped:
            if raw_kind not in unresolved:
                unresolved.append(raw_kind)
            continue
        for kind in mapped:
            if kind not in direct:
                direct.append(kind)

    artifacts = expand_artifact_dependencies(direct)
    return StructuralArtifactPlan(
        artifact_kinds=artifacts,
        branch_features=branch_features_for_artifacts(artifacts),
        unresolved_inputs=tuple(unresolved),
    )


__all__ = [
    "CANONICAL_ARTIFACT_KINDS",
    "StructuralArtifactPlan",
    "branch_features_for_artifacts",
    "detect_structural_artifacts",
    "expand_artifact_dependencies",
    "normalize_artifact_kind",
    "validate_artifact_kinds",
]
