from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

MODULE_KEYS = frozenset(
    {
        "module_id",
        "kind",
        "config",
        "depends_on",
        "required_gates",
    }
)
ASSET_KEYS = frozenset(
    {
        "asset_id",
        "kind",
        "prompt",
        "target_path",
        "width",
        "height",
    }
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
ASSET_KINDS = frozenset(
    {
        "item",
        "block",
        "entity",
        "gui",
        "environment",
        "icon",
    }
)

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
    text = re.sub(
        r"[^a-z0-9_]+",
        "_",
        str(value or "").strip().lower(),
    ).strip("_")
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


def build_batch_skeleton(
    batch_id: str,
    scope: str,
    deliverables: Sequence[str],
    exports: Sequence[str],
    depends_on_batches: Sequence[str] = (),
    known_module_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the host-owned production page before model output is merged.

    The model never creates the page shape. It can only fill values that the
    host subsequently accepts through ``merge_model_output_into_skeleton``.
    """
    batch = _normalize_id(batch_id, "batch")
    module_ids = [_normalize_id(item, "module") for item in exports] or [batch]
    known = set(known_module_ids)
    dependencies = [
        item
        for item in _unique_strings(depends_on_batches)
        if item in known
    ]

    modules = [
        {
            "module_id": module_id,
            "kind": "custom_java",
            "config": {
                "summary": f"Implementation for {module_id}",
                "batch_id": batch,
                "scope": str(scope or "").strip(),
            },
            "depends_on": dependencies,
            "required_gates": [],
        }
        for module_id in module_ids
    ]

    return {
        "modules": modules,
        "assets": [],
        "acceptance_tests": [
            f"test_{module_id}_registers" for module_id in module_ids
        ],
        "completed_deliverables": (
            _unique_strings(deliverables) or [f"{batch}_feature"]
        ),
        "complete": True,
        "next_cursor": "",
    }


def _merge_modules(
    skeleton: Mapping[str, Any],
    model_output: Mapping[str, Any],
    valid_module_catalog: set[str],
) -> list[dict[str, Any]]:
    raw_modules = model_output.get("modules")
    if not isinstance(raw_modules, list):
        return [
            dict(item)
            for item in skeleton.get("modules", [])
            if isinstance(item, dict)
        ]

    modules: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_modules:
        if not isinstance(raw, dict):
            continue

        item = {key: raw[key] for key in MODULE_KEYS if key in raw}
        module_id = _normalize_id(item.get("module_id"), "module")
        if module_id in seen_ids:
            continue

        kind = str(item.get("kind") or "custom_java")
        if kind not in MODULE_KINDS:
            kind = "custom_java"

        config = item.get("config")
        if not isinstance(config, dict):
            config = {}

        dependencies = [
            dependency
            for dependency in _unique_strings(item.get("depends_on"))
            if dependency in valid_module_catalog and dependency != module_id
        ]

        modules.append(
            {
                "module_id": module_id,
                "kind": kind,
                "config": config,
                "depends_on": dependencies,
                "required_gates": _unique_strings(item.get("required_gates")),
            }
        )
        seen_ids.add(module_id)

    if modules:
        return modules

    return [
        dict(item)
        for item in skeleton.get("modules", [])
        if isinstance(item, dict)
    ]


def _merge_assets(model_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_assets = model_output.get("assets")
    if not isinstance(raw_assets, list):
        return []

    assets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_assets:
        if not isinstance(raw, dict):
            continue

        item = {key: raw[key] for key in ASSET_KEYS if key in raw}
        asset_id = _normalize_id(item.get("asset_id"), "asset")
        if asset_id in seen_ids:
            continue

        kind = str(item.get("kind") or "item")
        prompt = str(item.get("prompt") or "").strip()
        target_path = _safe_asset_path(item.get("target_path"))
        if kind not in ASSET_KINDS or not prompt or not target_path:
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

    return assets


def merge_model_output_into_skeleton(
    skeleton: Mapping[str, Any],
    model_output: Mapping[str, Any],
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    """Merge model values into a closed host-owned production page.

    Unknown top-level fields and unknown module/asset fields are discarded.
    This keeps planner recovery deterministic instead of accumulating runtime
    compatibility patches for every malformed model response.
    """
    allowed_output = {
        key: model_output[key]
        for key in TOP_LEVEL_KEYS
        if key in model_output
    }

    return {
        "modules": _merge_modules(
            skeleton,
            allowed_output,
            valid_module_catalog,
        ),
        "assets": _merge_assets(allowed_output),
        "acceptance_tests": (
            _unique_strings(allowed_output.get("acceptance_tests"))
            or _unique_strings(skeleton.get("acceptance_tests"))
        ),
        "completed_deliverables": (
            _unique_strings(allowed_output.get("completed_deliverables"))
            or _unique_strings(skeleton.get("completed_deliverables"))
        ),
        "complete": bool(allowed_output.get("complete", True)),
        "next_cursor": str(allowed_output.get("next_cursor") or ""),
    }
