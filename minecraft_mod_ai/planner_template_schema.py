from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .implementation_template_contract import (
    build_implementation_template,
    sanitize_hole_fills,
)

MODULE_KEYS = frozenset(
    {"module_id", "kind", "config", "depends_on", "required_gates"}
)
ASSET_KEYS = frozenset(
    {"asset_id", "kind", "prompt", "target_path", "width", "height"}
)
TOP_LEVEL_KEYS = frozenset(
    {
        "modules",
        "assets",
        "acceptance_tests",
        "completed_deliverables",
        "complete",
        "next_cursor",
    }
)

# Evidence-task pages expose only this bounded detail surface to the planner model.
# The semantic contract itself remains byte-for-byte host-owned in ``evidence_task``.
MODEL_TASK_DETAIL_KEYS = frozenset(
    {
        "implementation_notes",
        "api_usage",
        "validation_notes",
        "asset_notes",
        "hole_fills",
    }
)

MODULE_KINDS = frozenset(
    {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "fluid",
        "machine",
        "recipe",
        "effect",
        "enchantment",
        "entity",
        "boss",
        "npc",
        "quest",
        "class",
        "skill",
        "economy",
        "shop",
        "gui",
        "networking",
        "party",
        "guild",
        "command",
        "structure",
        "biome",
        "dimension",
        "world_event",
        "advancement",
        "loot",
        "integration",
        "custom_java",
    }
)
ASSET_KINDS = frozenset({"item", "block", "entity", "gui", "environment", "icon"})

PRODUCTION_PAGE_TEMPLATE: dict[str, Any] = {
    "modules": [
        {
            "module_id": "example_module",
            "kind": "custom_java",
            "config": {},
            "depends_on": [],
            "required_gates": [],
        }
    ],
    "assets": [],
    "acceptance_tests": ["test_example_registers"],
    "completed_deliverables": ["example_deliverable"],
    "complete": True,
    "next_cursor": "",
}


def _normalize_id(value: Any, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text or not text[0].isalpha():
        text = f"{fallback}_{text}".rstrip("_")
    return text[:63]


def _unique_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(
        dict.fromkeys(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    )


def _safe_asset_path(value: Any) -> str:
    path = str(value or "").replace("\\", "/").strip()
    if not path or path.startswith("/") or ".." in path.split("/"):
        return ""
    return path


def _positive_int(value: Any, default: int) -> int:
    return value if type(value) is int and value > 0 else default


def _contract_task(contract: Mapping[str, Any]) -> dict[str, Any]:
    nested = contract.get("evidence_task")
    return dict(nested) if isinstance(nested, Mapping) else dict(contract)


def _contract_values(
    contracts: Mapping[str, Mapping[str, Any]],
    *keys: str,
) -> list[str]:
    values: list[str] = []
    for contract in contracts.values():
        if not isinstance(contract, Mapping):
            continue
        task = _contract_task(contract)
        for key in keys:
            for item in _unique_strings(task.get(key)):
                if item not in values:
                    values.append(item)
    return values


def build_batch_skeleton(
    batch_id: str,
    scope: str,
    deliverables: Sequence[str],
    exports: Sequence[str],
    depends_on_batches: Sequence[str] = (),
    known_module_ids: Sequence[str] = (),
    host_module_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    acceptance_tests: Sequence[str] = (),
) -> dict[str, Any]:
    """Create the complete page shape and module identities on the host.

    Evidence-owned pages derive acceptance and deliverables from the actual task contract.
    They never manufacture registration tests or feature names merely to make a page non-empty.
    """
    batch = _normalize_id(batch_id, "batch")
    module_ids = list(
        dict.fromkeys(_normalize_id(item, "module") for item in exports)
    ) or [batch]
    known = set(known_module_ids)
    dependencies = [
        item for item in _unique_strings(depends_on_batches) if item in known
    ]
    contracts = host_module_contracts or {}
    modules: list[dict[str, Any]] = []
    for module_id in module_ids:
        config: dict[str, Any] = {
            "summary": f"Implementation for {module_id}",
            "batch_id": batch,
            "scope": str(scope or "").strip(),
        }
        contract = contracts.get(module_id)
        if isinstance(contract, Mapping):
            evidence_task = _contract_task(contract)
            implementation_template = (
                build_implementation_template(evidence_task)
                if str(evidence_task.get("task_id") or "").strip()
                else None
            )
            config.update(
                {
                    "evidence_plan_sha256": str(
                        contract.get("evidence_plan_sha256") or ""
                    ),
                    "evidence_task": evidence_task,
                    "requirement_refs": _unique_strings(
                        contract.get("requirement_refs")
                    ),
                    "gap_refs": _unique_strings(contract.get("gap_refs")),
                    "reuse_refs": _unique_strings(contract.get("reuse_refs")),
                    "owned_anchors": list(contract.get("owned_anchors") or []),
                    "consumes": _unique_strings(contract.get("consumes")),
                    "provides": _unique_strings(contract.get("provides")),
                    "acceptance": _unique_strings(contract.get("acceptance")),
                    "impact_probes": _unique_strings(contract.get("impact_probes")),
                    "model_fill": {},
                }
            )
            if implementation_template is not None:
                config["implementation_template"] = implementation_template
        modules.append(
            {
                "module_id": module_id,
                "kind": "custom_java",
                "config": config,
                "depends_on": dependencies,
                "required_gates": (
                    _unique_strings(contract.get("required_gates"))
                    if isinstance(contract, Mapping)
                    else []
                ),
            }
        )

    explicit_acceptance = _unique_strings(acceptance_tests)
    explicit_deliverables = _unique_strings(deliverables)
    if contracts:
        derived_acceptance = _contract_values(
            contracts,
            "public_acceptance",
            "runtime_acceptance",
            "acceptance",
            "internal_invariants",
        )
        derived_deliverables = _contract_values(contracts, "provides")
        page_acceptance = explicit_acceptance or derived_acceptance
        page_deliverables = explicit_deliverables or derived_deliverables
    else:
        page_acceptance = explicit_acceptance or [
            f"test_{module_id}_registers" for module_id in module_ids
        ]
        page_deliverables = explicit_deliverables or [f"{batch}_feature"]

    return {
        "modules": modules,
        "assets": [],
        "acceptance_tests": page_acceptance,
        "completed_deliverables": page_deliverables,
        "complete": True,
        "next_cursor": "",
    }


def _merge_modules(
    skeleton: Mapping[str, Any],
    model_output: Mapping[str, Any],
    valid_module_catalog: set[str],
) -> list[dict[str, Any]]:
    skeleton_modules = [
        dict(item)
        for item in skeleton.get("modules", [])
        if isinstance(item, dict) and item.get("module_id")
    ]
    raw_modules = model_output.get("modules")
    if not isinstance(raw_modules, list):
        return skeleton_modules

    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_modules:
        if not isinstance(raw, Mapping):
            continue
        module_id = _normalize_id(raw.get("module_id"), "module")
        if module_id not in by_id:
            by_id[module_id] = raw

    merged: list[dict[str, Any]] = []
    for host_item in skeleton_modules:
        module_id = str(host_item["module_id"])
        raw = by_id.get(module_id)
        if raw is None:
            merged.append(host_item)
            continue

        item = {key: raw[key] for key in MODULE_KEYS if key in raw}
        kind = str(item.get("kind") or host_item.get("kind") or "custom_java")
        if kind not in MODULE_KINDS:
            kind = "custom_java"
        host_config = dict(host_item.get("config") or {})
        raw_config = item.get("config")
        evidence_owned = isinstance(host_config.get("evidence_task"), Mapping)
        if evidence_owned:
            details = {
                key: raw_config[key]
                for key in MODEL_TASK_DETAIL_KEYS
                if isinstance(raw_config, Mapping) and key in raw_config
            }
            implementation_template = host_config.get("implementation_template")
            if isinstance(implementation_template, Mapping):
                details["hole_fills"] = sanitize_hole_fills(
                    implementation_template,
                    details.get("hole_fills"),
                )
            else:
                details.pop("hole_fills", None)
            config = {**host_config, "model_fill": details}
            # Task kind, dependency edges, target, artifacts, holes, and gates are host-owned.
            kind = str(host_item.get("kind") or "custom_java")
        elif isinstance(raw_config, dict):
            config = dict(raw_config)
        else:
            config = host_config
        dependencies = [
            dependency
            for dependency in _unique_strings(host_item.get("depends_on"))
            if dependency in valid_module_catalog and dependency != module_id
        ]
        gates = (
            _unique_strings(host_item.get("required_gates"))
            if evidence_owned
            else _unique_strings(item.get("required_gates"))
        )
        merged.append(
            {
                "module_id": module_id,
                "kind": kind,
                "config": config,
                "depends_on": dependencies,
                "required_gates": gates,
            }
        )
    return merged


def _merge_assets(model_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_assets = model_output.get("assets")
    if not isinstance(raw_assets, list):
        return []
    assets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in raw_assets:
        if not isinstance(raw, Mapping):
            continue
        item = {key: raw[key] for key in ASSET_KEYS if key in raw}
        asset_id = _normalize_id(item.get("asset_id"), "asset")
        kind = str(item.get("kind") or "item")
        prompt = str(item.get("prompt") or "").strip()
        target_path = _safe_asset_path(item.get("target_path"))
        if (
            asset_id in seen_ids
            or target_path in seen_paths
            or kind not in ASSET_KINDS
            or not prompt
            or not target_path
        ):
            continue
        assets.append(
            {
                "asset_id": asset_id,
                "kind": kind,
                "prompt": prompt,
                "target_path": target_path,
                "width": _positive_int(item.get("width"), 16),
                "height": _positive_int(item.get("height"), 16),
            }
        )
        seen_ids.add(asset_id)
        seen_paths.add(target_path)
    return assets


def merge_model_output_into_skeleton(
    skeleton: Mapping[str, Any],
    model_output: Mapping[str, Any],
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    """Merge only whitelisted values into the closed host-owned page."""
    allowed_output = {
        key: model_output[key] for key in TOP_LEVEL_KEYS if key in model_output
    }
    evidence_owned = any(
        isinstance(item, Mapping)
        and isinstance(_mapping_config(item).get("evidence_task"), Mapping)
        for item in skeleton.get("modules", ())
    )
    return {
        "modules": _merge_modules(skeleton, allowed_output, valid_module_catalog),
        "assets": [] if evidence_owned else _merge_assets(allowed_output),
        "acceptance_tests": (
            _unique_strings(skeleton.get("acceptance_tests"))
            if evidence_owned
            else _unique_strings(allowed_output.get("acceptance_tests"))
            or _unique_strings(skeleton.get("acceptance_tests"))
        ),
        "completed_deliverables": (
            _unique_strings(skeleton.get("completed_deliverables"))
            if evidence_owned
            else _unique_strings(allowed_output.get("completed_deliverables"))
            or _unique_strings(skeleton.get("completed_deliverables"))
        ),
        "complete": True,
        "next_cursor": "",
    }


def _mapping_config(value: Mapping[str, Any]) -> dict[str, Any]:
    config = value.get("config")
    return dict(config) if isinstance(config, Mapping) else {}
