from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

MODULE_KEYS = frozenset({"module_id", "kind", "config", "depends_on", "required_gates"})
ASSET_KEYS = frozenset({"asset_id", "kind", "prompt", "target_path", "width", "height"})
MODULE_KINDS = frozenset({"item", "block", "tool", "weapon", "armor", "food", "crop", "fluid", "machine", "recipe", "effect", "enchantment", "entity", "boss", "npc", "quest", "class", "skill", "economy", "shop", "gui", "networking", "party", "guild", "command", "structure", "biome", "dimension", "world_event", "advancement", "loot", "integration", "custom_java"})
ASSET_KINDS = frozenset({"item", "block", "entity", "gui", "environment", "icon"})
PRODUCTION_PAGE_TEMPLATE = {"modules": [{"module_id": "example_module", "kind": "custom_java", "config": {}, "depends_on": [], "required_gates": []}], "assets": [], "acceptance_tests": ["test_example_registers"], "completed_deliverables": ["example_deliverable"], "complete": True, "next_cursor": ""}

def _id(value: Any, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not text or not text[0].isalpha():
        text = f"{fallback}_{text}".rstrip("_")
    return text[:63]

def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(x.strip() for x in value if isinstance(x, str) and x.strip()))

def build_batch_skeleton(batch_id: str, scope: str, deliverables: Sequence[str], exports: Sequence[str], depends_on_batches: Sequence[str] = (), known_module_ids: Sequence[str] = ()) -> dict[str, Any]:
    batch = _id(batch_id, "batch")
    module_ids = [_id(x, "module") for x in exports] or [batch]
    known = set(known_module_ids)
    deps = [x for x in _strings(depends_on_batches) if x in known]
    return {"modules": [{"module_id": mid, "kind": "custom_java", "config": {"summary": f"Implementation for {mid}", "batch_id": batch}, "depends_on": deps, "required_gates": []} for mid in module_ids], "assets": [], "acceptance_tests": [f"test_{mid}_registers" for mid in module_ids], "completed_deliverables": _strings(deliverables) or [f"{batch}_feature"], "complete": True, "next_cursor": ""}

def merge_model_output_into_skeleton(skeleton: Mapping[str, Any], model_output: Mapping[str, Any], valid_module_catalog: set[str]) -> dict[str, Any]:
    modules = []
    raw_modules = model_output.get("modules")
    if isinstance(raw_modules, list):
        for raw in raw_modules:
            if not isinstance(raw, dict):
                continue
            item = {k: raw[k] for k in MODULE_KEYS if k in raw}
            mid = _id(item.get("module_id"), "module")
            kind = str(item.get("kind") or "custom_java")
            if kind not in MODULE_KINDS:
                kind = "custom_java"
            config = item.get("config") if isinstance(item.get("config"), dict) else {}
            deps = [x for x in _strings(item.get("depends_on")) if x in valid_module_catalog and x != mid]
            modules.append({"module_id": mid, "kind": kind, "config": config, "depends_on": deps, "required_gates": _strings(item.get("required_gates"))})
    if not modules:
        modules = [dict(x) for x in skeleton.get("modules", []) if isinstance(x, dict)]
    assets = []
    raw_assets = model_output.get("assets")
    if isinstance(raw_assets, list):
        for raw in raw_assets:
            if not isinstance(raw, dict):
                continue
            item = {k: raw[k] for k in ASSET_KEYS if k in raw}
            aid = _id(item.get("asset_id"), "asset")
            kind = str(item.get("kind") or "item")
            target = str(item.get("target_path") or "").replace("\\", "/")
            prompt = str(item.get("prompt") or "").strip()
            if kind not in ASSET_KINDS or not target or target.startswith("/") or ".." in target.split("/") or not prompt:
                continue
            width = item.get("width") if type(item.get("width")) is int and item.get("width") > 0 else 16
            height = item.get("height") if type(item.get("height")) is int and item.get("height") > 0 else 16
            assets.append({"asset_id": aid, "kind": kind, "prompt": prompt, "target_path": target, "width": width, "height": height})
    return {"modules": modules, "assets": assets, "acceptance_tests": _strings(model_output.get("acceptance_tests")) or _strings(skeleton.get("acceptance_tests")), "completed_deliverables": _strings(model_output.get("completed_deliverables")) or _strings(skeleton.get("completed_deliverables")), "complete": bool(model_output.get("complete", True)), "next_cursor": str(model_output.get("next_cursor") or "")}
