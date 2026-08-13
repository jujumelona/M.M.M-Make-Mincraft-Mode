from __future__ import annotations

from typing import Any

from .agentic_research_game_design import bind_game_design_planner


def install(complete_planner_module: Any) -> None:
    """Keep planner prompts target-neutral and bind research-first game design.

    The exact platform coordinates remain host-owned. Game-design generation is bound
    here because this planner prompt layer is installed after target-neutral planning
    semantics and before the outer live-platform selection wrapper. The binder is a
    normal planner helper, not another contract composer.
    """

    from . import game_design as game_design_module

    bind_game_design_planner(game_design_module)

    replacements = (
        (
            "Minecraft Java 1.20.1 Fabric",
            "the host-selected Minecraft Java Fabric target",
        ),
        (
            "Minecraft 1.20.1 Fabric",
            "the host-selected Minecraft Fabric target",
        ),
    )
    for name, value in tuple(vars(complete_planner_module).items()):
        if not name.startswith("_") or not isinstance(value, str):
            continue
        updated = value
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != value:
            setattr(complete_planner_module, name, updated)
