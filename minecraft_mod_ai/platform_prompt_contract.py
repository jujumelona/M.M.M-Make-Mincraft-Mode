from __future__ import annotations

"""Target-neutral planner prompt cleanup only."""

from typing import Any


def install(complete_planner_module: Any) -> None:
    replacements = (
        ("Minecraft Java 1.20.1 Fabric", "the host-selected Minecraft Java target"),
        ("Minecraft 1.20.1 Fabric", "the host-selected Minecraft target"),
        ("the host-selected Minecraft Java Fabric target", "the host-selected Minecraft Java target"),
        ("the host-selected Minecraft Fabric target", "the host-selected Minecraft target"),
    )
    for name, value in tuple(vars(complete_planner_module).items()):
        if not name.startswith("_") or not isinstance(value, str):
            continue
        updated = value
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != value:
            setattr(complete_planner_module, name, updated)


__all__ = ["install"]
