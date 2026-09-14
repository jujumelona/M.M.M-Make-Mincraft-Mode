from __future__ import annotations

"""Atomic design slot pipeline for small models.

The model path is an enrichment path, not a terminal gate. If graph/slot generation is
malformed, interrupted, or unavailable, the host deterministically projects the frozen
request catalog into the exact game-design shape consumed downstream.
"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

ALL_DESIGN_SLOTS: tuple[str, ...] = (
    "design/audio_identity",
    "design/combat_role",
    "design/content_scale",
    "design/core_action",
    "design/core_loop",
    "design/core_loop_step",
    "design/crafting_role",
    "design/economy_sink",
    "design/economy_source",
    "design/enemy_role",
    "design/exploration_target",
    "design/failure_condition",
    "design/first_goal",
    "design/goal_prerequisite",
    "design/machine_role",
    "design/network_requirement",
    "design/npc_role",
    "design/persistence_requirement",
    "design/player_fantasy",
    "design/progression_condition",
    "design/progression_edge",
    "design/progression_node",
    "design/resource_sink",
    "design/resource_source",
    "design/reward",
    "design/risk",
    "design/texture_requirement",
    "design/theme",
    "design/ui_requirement",
    "design/unlock",
    "design/visual_identity",
    "design/world_interaction",
)
DESIGN_SLOTS = ALL_DESIGN_SLOTS


def _sanitize_stem(name: str) -> str:
    cleaned = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in name.lower())
    parts = [p for p in cleaned.split("_") if p]
    stem = "_".join(parts)
    if not stem:
        stem = f"mmm_{hashlib.sha256(name.encode('utf-8')).hexdigest()[:10]}"
    if not stem[0].isalpha() or not stem[0].isascii():
        stem = f"mod_{stem}"
    return stem[:30]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _host_design(prompt: str, request_catalog: Mapping[str, Any] | None) -> dict[str, Any]:
    requirements = (
        request_catalog.get("requirements", [])
        if isinstance(request_catalog, Mapping)
        else []
    )
    rows = [row for row in requirements if isinstance(row, Mapping)]
    if not rows:
        rows = [
            {
                "requirement_id": "req_001",
                "capability": _text(prompt).casefold() or "authored_request",
                "statement": _text(prompt) or "Authored Minecraft behavior",
                "acceptance": [_text(prompt) or "Authored Minecraft behavior is observable"],
            }
        ]

    statements = [
        _text(row.get("semantic_statement"))
        or _text(row.get("statement"))
        or _text(row.get("capability"))
        for row in rows
    ]
    statements = list(dict.fromkeys(value for value in statements if value))
    acceptance_tests: list[str] = []
    modules: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        requirement_id = _text(row.get("requirement_id")) or f"req_{index:03d}"
        statement = (
            _text(row.get("semantic_statement"))
            or _text(row.get("statement"))
            or _text(row.get("capability"))
            or requirement_id
        )
        raw_acceptance = row.get("acceptance")
        checks = (
            [_text(item) for item in raw_acceptance if _text(item)]
            if isinstance(raw_acceptance, Sequence) and not isinstance(raw_acceptance, (str, bytes, bytearray))
            else []
        )
        if not checks:
            checks = [statement]
        acceptance_tests.extend(checks)
        obligations = row.get("implementation_obligations")
        if not isinstance(obligations, list) or not obligations:
            obligations = [f"Implement the authored requirement: {statement}"]
        modules.append(
            {
                "plugin_id": f"design_{requirement_id}",
                "status": "custom_required",
                "capability": _text(row.get("capability")) or statement,
                "reason": statement,
                "requirement_refs": [requirement_id],
                "implementation_obligations": [
                    _text(item) for item in obligations if _text(item)
                ] or [f"Implement the authored requirement: {statement}"],
            }
        )

    return {
        "title": "Requested Minecraft Mod",
        "pitch": "Implement the authored Minecraft behaviors without expanding their scope.",
        "core_loop": statements or [_text(prompt)],
        "progression": statements or [_text(prompt)],
        "combat": {},
        "mod_context": {"host_projection": True},
        "modules": modules,
        "assets": [],
        "acceptance_tests": list(dict.fromkeys(acceptance_tests)),
        "_design_slots": {},
        "_implementation_facts": [],
        "_content_entities": [],
        "_content_relations": [],
        "_research_facts": [],
    }


def compile_atomic_design(
    prompt: str,
    router: Any = None,
    *,
    research: Mapping[str, Any] | None = None,
    request_catalog: Mapping[str, Any] | None = None,
    progress=None,
    checkpoint=None,
) -> dict[str, Any]:
    """Compile design slots; technical generation defects fall back to host projection."""
    prompt_text = str(prompt).strip()
    if not prompt_text:
        raise ValueError("ATOMIC_DESIGN: prompt must not be empty")

    from .content_design_graph import compile_content_graph
    from .minecraft_generation_design import complete_generation_fields
    from .root_cause_trace import emit_root_cause

    try:
        graph = compile_content_graph(
            prompt_text,
            router,
            request_catalog=request_catalog,
            research=research,
            progress=progress,
            checkpoint=checkpoint,
        )
        return complete_generation_fields(
            graph,
            router,
            prompt=prompt_text,
            progress=progress,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        design = _host_design(prompt_text, request_catalog)
        emit_root_cause(
            "atomic_design_host_projection",
            stage="planning",
            operation="compile_atomic_design",
            result="CONTINUE",
            reason=f"{type(exc).__name__}: {exc}",
            details={
                "requirements": len(design.get("modules", [])),
                "acceptance_tests": len(design.get("acceptance_tests", [])),
            },
        )
        return design


__all__ = ["ALL_DESIGN_SLOTS", "DESIGN_SLOTS", "compile_atomic_design"]
