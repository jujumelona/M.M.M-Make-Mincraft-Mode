from __future__ import annotations

"""Deterministic execution of the fixed translation template sequence."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .structural_artifact_mapping import (
    branch_features_for_artifacts,
    expand_artifact_dependencies,
    normalize_artifact_kind,
    validate_artifact_kinds,
)
from .task_template_catalog import load_template

TRANSLATION_SEQUENCE: tuple[str, ...] = (
    "translation/minecraft_capability_mapping",
    "translation/artifact_detection",
    "translation/artifact_dependency_graph",
    "translation/client_server_split",
    "translation/persistence_mapping",
    "translation/networking_mapping",
    "translation/ui_mapping",
    "translation/resource_mapping",
)

_CLIENT_ARTIFACTS = frozenset({"screen", "hud", "keybind", "model", "texture", "animation", "language"})
_SERVER_ARTIFACTS = frozenset({"saved_data", "worldgen", "structure", "biome", "dimension", "recipe", "loot", "advancement"})
_UI_ARTIFACTS = frozenset({"menu", "screen", "hud", "keybind", "language"})
_RESOURCE_ARTIFACTS = frozenset({"recipe", "loot", "tag", "advancement", "worldgen", "structure", "biome", "dimension", "model", "texture", "animation", "sound", "language", "datagen"})


@dataclass(frozen=True)
class TranslationPlan:
    artifact_kinds: tuple[str, ...]
    branch_features: frozenset[str]
    unresolved_inputs: tuple[str, ...]
    server_artifacts: tuple[str, ...]
    client_artifacts: tuple[str, ...]
    shared_artifacts: tuple[str, ...]
    persistence_artifacts: tuple[str, ...]
    network_artifacts: tuple[str, ...]
    ui_artifacts: tuple[str, ...]
    resource_artifacts: tuple[str, ...]
    receipts: tuple[dict[str, Any], ...]


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _raw_structural_kinds(requirement: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for row in _sequence(requirement.get("artifact_obligations")):
        raw = row.get("kind") if isinstance(row, Mapping) else row
        text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if text and text not in values:
            values.append(text)
    for raw in _sequence(requirement.get("implementation_surfaces")):
        text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if text and text not in values:
            values.append(text)
    structure = requirement.get("minecraft_structure")
    if isinstance(structure, Mapping):
        for raw in _sequence(structure.get("required_artifacts")):
            text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
            if text and text not in values:
                values.append(text)
    for raw in _sequence(requirement.get("structural_artifacts")):
        text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if text and text not in values:
            values.append(text)
    for raw in _sequence(requirement.get("artifacts")):
        text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _receipt(template_id: str, output: Mapping[str, Any], *, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "template_id": template_id,
        "status": "PASS" if passed else "BLOCKED",
        "output": dict(output),
        "proof": {"passed": bool(passed), "reason": reason},
    }


def _validate_template_sequence() -> None:
    for identifier in TRANSLATION_SEQUENCE:
        template = load_template(identifier)
        if template.get("id") != identifier:
            raise ValueError(f"TRANSLATION_TEMPLATE: invalid template identity {identifier}")
        if not isinstance(template.get("input"), Mapping) or not isinstance(template.get("output"), Mapping):
            raise ValueError(f"TRANSLATION_TEMPLATE: {identifier} must declare input and output")
        proof = template.get("proof")
        if not isinstance(proof, Mapping) or not str(proof.get("predicate") or "").strip():
            raise ValueError(f"TRANSLATION_TEMPLATE: {identifier} must declare a proof predicate")


def translate_requirement(requirement: Mapping[str, Any]) -> TranslationPlan:
    """Execute every translation template in fixed order using structural inputs only."""

    _validate_template_sequence()
    raw_kinds = _raw_structural_kinds(requirement)
    receipts: list[dict[str, Any]] = []

    direct: list[str] = []
    unresolved: list[str] = []
    for raw in raw_kinds:
        normalized = normalize_artifact_kind(raw)
        if not normalized:
            unresolved.append(raw)
            continue
        for kind in normalized:
            if kind not in direct:
                direct.append(kind)
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[0],
            {"implementation_surfaces": list(direct), "evidence_links": ["explicit_structural_requirement"] if raw_kinds else []},
            passed=not unresolved,
            reason="surfaces derive only from explicit structural fields" if not unresolved else "unsupported structural surface is unresolved",
        )
    )

    detected = validate_artifact_kinds(direct)
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[1],
            {"artifact_kinds": list(detected), "unresolved_inputs": list(unresolved)},
            passed=not unresolved,
            reason="all emitted artifact kinds are canonical" if not unresolved else "unsupported input was preserved as unresolved",
        )
    )

    artifacts = expand_artifact_dependencies(detected)
    dependency_closed = expand_artifact_dependencies(artifacts) == artifacts
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[2],
            {"artifact_kinds": list(artifacts)},
            passed=dependency_closed,
            reason="artifact set is canonical, duplicate-free, and dependency-closed",
        )
    )

    client = tuple(kind for kind in artifacts if kind in _CLIENT_ARTIFACTS)
    server = tuple(kind for kind in artifacts if kind in _SERVER_ARTIFACTS)
    shared = tuple(kind for kind in artifacts if kind not in _CLIENT_ARTIFACTS and kind not in _SERVER_ARTIFACTS)
    side_partition = len(set(client) | set(server) | set(shared)) == len(artifacts)
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[3],
            {"server_artifacts": list(server), "client_artifacts": list(client), "shared_artifacts": list(shared), "unresolved_boundaries": []},
            passed=side_partition,
            reason="every canonical artifact has exactly one host side classification",
        )
    )

    persistence = tuple(kind for kind in artifacts if kind == "saved_data")
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[4],
            {"persistence_artifacts": list(persistence), "state_owner": "explicit_requirement", "lifecycle_requirements": [], "unresolved_inputs": []},
            passed=True,
            reason="persistence is emitted only when saved_data is structurally required",
        )
    )

    network = tuple(kind for kind in artifacts if kind == "network_payload")
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[5],
            {"payload_artifacts": list(network), "directions": [], "validation_requirements": [], "unresolved_inputs": []},
            passed=True,
            reason="network work is emitted only when network_payload is structurally required",
        )
    )

    ui = tuple(kind for kind in artifacts if kind in _UI_ARTIFACTS)
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[6],
            {"ui_artifacts": list(ui), "state_projection": "explicit_requirement", "interaction_boundary": "explicit_requirement", "unresolved_inputs": []},
            passed=True,
            reason="UI work is selected only from canonical UI artifacts",
        )
    )

    resources = tuple(kind for kind in artifacts if kind in _RESOURCE_ARTIFACTS)
    receipts.append(
        _receipt(
            TRANSLATION_SEQUENCE[7],
            {"resource_artifacts": list(resources), "asset_artifacts": [kind for kind in resources if kind in {"model", "texture", "animation", "sound"}], "unresolved_inputs": []},
            passed=True,
            reason="resource work is selected only from canonical resource artifacts",
        )
    )

    if tuple(receipt["template_id"] for receipt in receipts) != TRANSLATION_SEQUENCE:
        raise AssertionError("TRANSLATION_SEQUENCE: runtime order drifted")

    return TranslationPlan(
        artifact_kinds=artifacts,
        branch_features=branch_features_for_artifacts(artifacts),
        unresolved_inputs=tuple(unresolved),
        server_artifacts=server,
        client_artifacts=client,
        shared_artifacts=shared,
        persistence_artifacts=persistence,
        network_artifacts=network,
        ui_artifacts=ui,
        resource_artifacts=resources,
        receipts=tuple(receipts),
    )


__all__ = ["TRANSLATION_SEQUENCE", "TranslationPlan", "translate_requirement"]
