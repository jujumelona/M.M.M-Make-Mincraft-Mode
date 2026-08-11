from __future__ import annotations

from typing import Any


def install(complete_planner_module: Any) -> None:
    """Remove stale global 1.20.1 instructions after host target selection was added.

    The actual selected coordinates are injected by the implementation/planning
    contracts. Global planner prompts must describe the product, not override that
    host-owned target with an obsolete fixed version.
    """

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
