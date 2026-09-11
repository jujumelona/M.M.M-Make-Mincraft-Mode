from __future__ import annotations

"""Atomic design slot pipeline for small models.

Compiles user requests and grounded research into discrete, bounded design
slots strictly complying with physical atomicity bounds (MAX_MODEL_FIELDS <= 3,
MAX_MODEL_STRING_CHARS <= 256, MAX_SCHEMA_DEPTH <= 2).
"""

import hashlib
from collections.abc import Mapping
from typing import Any

# Full ontology of 32 atomic design slots from high-level fantasy down to mechanics, systems, progression, and assets
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


def compile_atomic_design(
    prompt: str,
    router: Any = None,
    *,
    research: Mapping[str, Any] | None = None,
    request_catalog: Mapping[str, Any] | None = None,
    progress=None,
    checkpoint=None,
) -> dict[str, Any]:
    """Compile atomic design and finish only the fields deterministic generation consumes."""
    prompt_text = str(prompt).strip()
    if not prompt_text:
        raise ValueError("ATOMIC_DESIGN: prompt must not be empty")

    from .content_design_graph import compile_content_graph
    from .minecraft_generation_design import complete_generation_fields

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


__all__ = ["ALL_DESIGN_SLOTS", "DESIGN_SLOTS", "compile_atomic_design"]
