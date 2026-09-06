from __future__ import annotations

"""Fail-closed target grounding and unambiguous project-module identity.

Provider-specific completeness (Fabric/Gradle/pack receipts) is checked here. Minecraft
version/loader/mapping applicability and naming-regime semantics are owned only by
``target_contract`` and are consumed, never redefined, by this boundary.
"""

import re
from collections.abc import Mapping
from functools import wraps
from typing import Any

from . import evidence_first_planning as _planning
from .target_contract import (
    TargetContractError,
    required_target_fields,
    validate_complete_target,
)

_INSTALLED = False
def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _required_target_fields(coordinates: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        return required_target_fields(coordinates)
    except TargetContractError as exc:
        raise _planning.EvidencePlanError(str(exc)) from exc


def _validate_complete_target(coordinates: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_complete_target(coordinates)
    except TargetContractError as exc:
        raise _planning.EvidencePlanError(str(exc)) from exc


def _logical_module_id(raw_path: str, item: Mapping[str, Any]) -> str:
    explicit = _text(item.get("logical_module_id") or item.get("artifact_id") or item.get("name"))
    if explicit:
        source = explicit
    elif raw_path == ":":
        source = "root"
    elif raw_path.startswith(":"):
        source = raw_path.strip(":").replace(":", "_")
    else:
        source = raw_path
    value = re.sub(r"[^a-z0-9_]+", "_", source.casefold()).strip("_")
    value = re.sub(r"_+", "_", value)
    if not value:
        value = "root"
    if not value[0].isalpha():
        value = "module_" + value
    return value[:64]


def _project_topology(game_design: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    inventory = game_design.get("_existing_project_inventory") or game_design.get("_existing_snapshot")
    inventory = dict(inventory) if isinstance(inventory, Mapping) else {}
    raw_modules = inventory.get("modules")
    raw_modules = raw_modules if isinstance(raw_modules, list) else []

    modules: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_modules:
        if not isinstance(raw, Mapping):
            continue
        raw_identity = _text(raw.get("module_id") or raw.get("gradle_project_path"))
        if not raw_identity:
            continue
        source_sets = list(_planning._strings(raw.get("source_sets")))
        if len(raw_modules) > 1 and raw_identity == ":" and not source_sets:
            continue
        gradle_path = _text(raw.get("gradle_project_path"))
        if not gradle_path and (raw_identity == ":" or raw_identity.startswith(":")):
            gradle_path = raw_identity
        logical_id = _logical_module_id(raw_identity, raw)
        base = logical_id
        suffix = 2
        while logical_id in seen_ids:
            logical_id = f"{base}_{suffix}"[:64]
            suffix += 1
        seen_ids.add(logical_id)
        if gradle_path:
            seen_paths.add(gradle_path)
        modules.append(
            {
                "module_id": logical_id,
                "gradle_project_path": gradle_path,
                "source_sets": source_sets,
            }
        )

    if not modules:
        for value in current.get("module_ids", []) if isinstance(current.get("module_ids"), list) else []:
            raw_identity = _text(value)
            if not raw_identity:
                continue
            logical_id = _logical_module_id(raw_identity, {})
            if logical_id in seen_ids:
                continue
            seen_ids.add(logical_id)
            gradle_path = raw_identity if raw_identity == ":" or raw_identity.startswith(":") else ""
            if gradle_path:
                seen_paths.add(gradle_path)
            modules.append(
                {"module_id": logical_id, "gradle_project_path": gradle_path, "source_sets": []}
            )

    loaders = list(_planning._strings(current.get("loaders")))
    source_sets = sorted(
        {
            source_set
            for module in modules
            for source_set in module.get("source_sets", [])
            if source_set
        }
    )
    return {
        "modules": modules,
        "module_ids": [module["module_id"] for module in modules],
        "gradle_project_paths": sorted(seen_paths),
        "loaders": loaders,
        "source_sets": source_sets,
        "identity_contract": (
            "module_id is a logical production identity; gradle_project_path is the Gradle path. "
            "The root ':' value is never a module_id."
        ),
    }


def _harden_target_decision(original: Any, game_design: Mapping[str, Any], target_decision: Any = None):
    result = dict(original(game_design, target_decision))
    coordinates = result.get("coordinates")
    coordinates = dict(coordinates) if isinstance(coordinates, Mapping) else {}
    version = _text(coordinates.get("minecraft_version"))
    loader = _text(coordinates.get("loader"))
    materially_selected = (
        version and version.casefold() != "unresolved" and loader and loader.casefold() != "unresolved"
    )
    required_fields = list(_required_target_fields(coordinates))
    if materially_selected:
        coordinates = _validate_complete_target(coordinates)
        result["coordinates"] = coordinates
        result["hard_gate_status"] = "passed"
        result["target_grounding"] = {
            "schema_version": "mmm/target-grounding-v3",
            "status": "COMPLETE",
            "required_fields": required_fields,
            "release_metadata_url": coordinates["release_metadata_url"],
            "naming_regime": dict(coordinates["naming_regime"]),
        }
    else:
        result["hard_gate_status"] = "deferred"
        result["target_grounding"] = {
            "schema_version": "mmm/target-grounding-v3",
            "status": "UNRESOLVED",
            "required_fields": required_fields,
        }

    current_topology = result.get("project_topology")
    current_topology = dict(current_topology) if isinstance(current_topology, Mapping) else {}
    result["project_topology"] = _project_topology(game_design, current_topology)
    result["decision_sha256"] = ""
    result["decision_sha256"] = _planning._hash_without(result, "decision_sha256")
    return result


def install_target_grounding_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    current = _planning._target_decision
    if not getattr(current, "_mmm_complete_target_grounding_v3", False):
        @wraps(current)
        def target_decision(game_design: Mapping[str, Any], target_decision: Any = None):
            return _harden_target_decision(current, game_design, target_decision)

        target_decision._mmm_complete_target_grounding_v3 = True
        _planning._target_decision = target_decision
    _INSTALLED = True


__all__ = ["install_target_grounding_contract"]
