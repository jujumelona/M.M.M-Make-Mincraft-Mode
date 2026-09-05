from __future__ import annotations

"""Fail-closed target grounding and unambiguous project-module identity.

Planning may not pass the target barrier with only a Minecraft version and loader. The
selected executable provider receipt must be semantically complete for the selected target.
Target-version semantics come exclusively from ``target_profile_semantics``.
"""

import re
from collections.abc import Mapping
from functools import wraps
from typing import Any

from . import evidence_first_planning as _planning
from .target_profile_semantics import uses_native_names

_INSTALLED = False
_BASE_REQUIRED_TARGET_FIELDS = (
    "minecraft_version",
    "loader",
    "java_version",
    "fabric_loader",
    "fabric_api",
    "fabric_loom",
    "gradle",
    "gradle_sha256",
    "data_pack_version",
    "resource_pack_version",
    "resource_pack_format",
    "release_metadata_url",
)
_LEGACY_MAPPING_FIELDS = ("mappings_kind", "mappings_version")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_unresolved(value: Any) -> bool:
    return not _text(value) or _text(value).casefold() == "unresolved"


def _required_target_fields(coordinates: Mapping[str, Any]) -> tuple[str, ...]:
    version = coordinates.get("minecraft_version")
    if _is_unresolved(version) or uses_native_names(version):
        return _BASE_REQUIRED_TARGET_FIELDS
    return _BASE_REQUIRED_TARGET_FIELDS + _LEGACY_MAPPING_FIELDS


def _legacy_mapping_claims(coordinates: Mapping[str, Any]) -> dict[str, str]:
    claims: dict[str, str] = {}
    for field in (*_LEGACY_MAPPING_FIELDS, "yarn_mappings"):
        value = _text(coordinates.get(field))
        if value and value.casefold() != "unresolved":
            claims[field] = value
    return claims


def _validate_complete_target(coordinates: Mapping[str, Any]) -> dict[str, Any]:
    required_fields = _required_target_fields(coordinates)
    missing = [field for field in required_fields if _is_unresolved(coordinates.get(field))]
    if missing:
        raise _planning.EvidencePlanError(
            "TARGET_GROUNDING_INCOMPLETE: executable provider target is missing "
            + ", ".join(missing)
        )

    minecraft_version = _text(coordinates.get("minecraft_version"))
    try:
        native_names = uses_native_names(minecraft_version)
    except ValueError as exc:
        raise _planning.EvidencePlanError(str(exc)) from exc

    mappings_receipt: dict[str, str] | None = None
    if native_names:
        legacy_claims = _legacy_mapping_claims(coordinates)
        if legacy_claims:
            raise _planning.EvidencePlanError(
                "TARGET_MAPPINGS_INAPPLICABLE: Minecraft 26.1+ uses the native/unobfuscated "
                "naming regime; legacy mapping coordinates must not be fabricated or accepted "
                f"for this target ({', '.join(sorted(legacy_claims))})."
            )
        naming_regime = {
            "kind": "native_unobfuscated",
            "mappings_applicable": False,
            "minecraft_version": minecraft_version,
        }
    else:
        mappings_kind = _text(coordinates.get("mappings_kind")).casefold()
        mappings_version = _text(coordinates.get("mappings_version"))
        if mappings_kind not in {"mojang", "yarn"}:
            raise _planning.EvidencePlanError(
                f"TARGET_MAPPINGS_KIND: unsupported mappings kind {mappings_kind!r}."
            )
        legacy_mapping = _text(coordinates.get("yarn_mappings"))
        if (
            legacy_mapping
            and legacy_mapping.casefold() != "unresolved"
            and legacy_mapping != mappings_version
        ):
            raise _planning.EvidencePlanError(
                "TARGET_MAPPINGS_ALIAS: legacy yarn_mappings disagrees with mappings_version."
            )
        mappings_receipt = {"kind": mappings_kind, "version": mappings_version}
        naming_regime = {
            "kind": "mapped_obfuscated",
            "mappings_applicable": True,
            "minecraft_version": minecraft_version,
        }

    gradle_sha = _text(coordinates.get("gradle_sha256")).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", gradle_sha):
        raise _planning.EvidencePlanError(
            "TARGET_GRADLE_RECEIPT: target Gradle SHA-256 is missing or invalid."
        )

    data_pack = _text(coordinates.get("data_pack_version"))
    resource_pack = _text(coordinates.get("resource_pack_version"))
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", data_pack):
        raise _planning.EvidencePlanError(
            f"TARGET_DATA_PACK_VERSION: invalid data pack version {data_pack!r}."
        )
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", resource_pack):
        raise _planning.EvidencePlanError(
            f"TARGET_RESOURCE_PACK_VERSION: invalid resource pack version {resource_pack!r}."
        )
    resource_format = coordinates.get("resource_pack_format")
    if type(resource_format) is not int or resource_format <= 0:
        raise _planning.EvidencePlanError(
            "TARGET_RESOURCE_PACK_FORMAT: provider-derived format must be a positive integer."
        )
    if resource_format != int(resource_pack.split(".", 1)[0]):
        raise _planning.EvidencePlanError(
            "TARGET_RESOURCE_PACK_FORMAT: format major disagrees with exact resource pack version."
        )
    release_url = _text(coordinates.get("release_metadata_url"))
    if not release_url.startswith(
        (
            "https://www.minecraft.net/",
            "https://feedback.minecraft.net/",
            "https://piston-meta.mojang.com/",
            "https://launcher.mojang.com/",
        )
    ):
        raise _planning.EvidencePlanError(
            "TARGET_PACK_PROVENANCE: pack metadata is not grounded in an official Minecraft/Mojang metadata URL."
        )

    result = dict(coordinates)
    if mappings_receipt is None:
        for field in (*_LEGACY_MAPPING_FIELDS, "yarn_mappings"):
            result.pop(field, None)
    else:
        result["mappings"] = mappings_receipt
    result["naming_regime"] = naming_regime
    result["pack_versions"] = {
        "data": data_pack,
        "resource": resource_pack,
        "resource_major": resource_format,
    }
    result["target_schema_version"] = "3"
    return result


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
    try:
        required_fields = list(_required_target_fields(coordinates))
    except ValueError as exc:
        raise _planning.EvidencePlanError(str(exc)) from exc
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
