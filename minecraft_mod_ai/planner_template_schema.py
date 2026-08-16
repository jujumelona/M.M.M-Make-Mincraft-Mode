"""Canonical JSON template and filling utilities for Minecraft Mod production planning."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


PRODUCTION_PAGE_TEMPLATE: dict[str, Any] = {
    "modules": [
        {
            "module_id": "example_module",
            "kind": "custom_java",
            "config": {"summary": "Description of feature implementation"},
            "depends_on": [],
            "required_gates": [],
            "implements_deliverables": ["example_deliverable"],
        }
    ],
    "assets": [],
    "audio": [],
    "acceptance_tests": ["test_example_registers"],
    "completed_deliverables": ["example_deliverable"],
    "complete": True,
    "next_cursor": "",
}


def build_batch_skeleton(
    batch_id: str,
    scope: str,
    deliverables: Sequence[str],
    exports: Sequence[str],
    depends_on_batches: Sequence[str] = (),
    known_module_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic pre-filled template skeleton for a single batch."""
    deliv_list = [str(d).strip() for d in deliverables if str(d).strip()] or [f"{batch_id}_feature"]
    export_list = [str(e).strip() for e in exports if str(e).strip()] or [batch_id]

    modules = [
        {
            "module_id": export_id,
            "kind": "custom_java",
            "config": {
                # Scope is already carried by the enclosing batch request. Repeating it
                # here can duplicate an entire authoritative request page on the hot path.
                "summary": f"Implementation for {export_id}",
                "batch_id": batch_id,
            },
            "depends_on": [dep for dep in depends_on_batches if dep in known_module_ids],
            "required_gates": [],
            "implements_deliverables": deliv_list,
        }
        for export_id in export_list
    ]

    tests = [f"test_{export_id}_registers" for export_id in export_list]

    return {
        "modules": modules,
        "assets": [],
        "audio": [],
        "acceptance_tests": tests,
        "completed_deliverables": deliv_list,
        "complete": True,
        "next_cursor": "",
    }


def merge_model_output_into_skeleton(
    skeleton: Mapping[str, Any],
    model_output: Mapping[str, Any],
    valid_module_catalog: set[str],
) -> dict[str, Any]:
    """Merge model-generated details into the canonical skeleton, preserving structural integrity."""
    raw_modules = model_output.get("modules")
    raw_assets = model_output.get("assets")
    raw_audio = model_output.get("audio")
    raw_tests = model_output.get("acceptance_tests")
    raw_completed = model_output.get("completed_deliverables")

    modules: list[dict[str, Any]] = []
    if isinstance(raw_modules, list) and raw_modules:
        for item in raw_modules:
            if not isinstance(item, dict):
                continue
            mod_id = str(item.get("module_id") or "").strip()
            if not mod_id:
                continue
            kind = str(item.get("kind") or "custom_java").strip() or "custom_java"
            cfg = item.get("config") if isinstance(item.get("config"), dict) else {"summary": str(item.get("config") or "")}
            raw_deps = item.get("depends_on") if isinstance(item.get("depends_on"), list) else []
            # Keep only valid module IDs
            deps = [str(d).strip() for d in raw_deps if str(d).strip() in valid_module_catalog and str(d).strip() != mod_id]
            gates = [str(g).strip() for g in item.get("required_gates", []) if str(g).strip()] if isinstance(item.get("required_gates"), list) else []
            claims = [str(c).strip() for c in item.get("implements_deliverables", []) if str(c).strip()] if isinstance(item.get("implements_deliverables"), list) else []
            modules.append({
                "module_id": mod_id,
                "kind": kind,
                "config": cfg,
                "depends_on": deps,
                "required_gates": gates,
                "implements_deliverables": claims or list(skeleton.get("completed_deliverables", [])),
            })

    if not modules:
        modules = list(skeleton.get("modules", []))

    assets: list[dict[str, Any]] = []
    if isinstance(raw_assets, list):
        for item in raw_assets:
            if isinstance(item, dict) and item.get("asset_id"):
                assets.append(item)

    audio: list[dict[str, Any]] = []
    if isinstance(raw_audio, list):
        for item in raw_audio:
            if isinstance(item, dict) and item.get("sound_id"):
                audio.append(item)

    tests: list[str] = []
    if isinstance(raw_tests, list) and raw_tests:
        tests = [str(t).strip() for t in raw_tests if str(t).strip()]
    else:
        tests = list(skeleton.get("acceptance_tests", []))

    completed: list[str] = []
    if isinstance(raw_completed, list) and raw_completed:
        completed = [str(c).strip() for c in raw_completed if str(c).strip()]
    else:
        completed = list(skeleton.get("completed_deliverables", []))

    return {
        "modules": modules,
        "assets": assets,
        "audio": audio,
        "acceptance_tests": tests,
        "completed_deliverables": completed,
        "complete": True,
        "next_cursor": "",
    }
