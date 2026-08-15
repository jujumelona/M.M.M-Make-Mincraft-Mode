from __future__ import annotations

"""Public presentation compatibility for platform metadata.

Session target constraints are implemented directly in ``api.py``.  This contract no
longer wraps constructors or injects a placeholder Minecraft/Fabric target.
"""

from functools import wraps
from typing import Any


def install(api_module: Any, plan_render_module: Any) -> None:
    def supported_versions(*, loader: str | None = None) -> tuple[str, ...]:
        from .platform_catalog import supported_minecraft_versions

        return supported_minecraft_versions(loader=loader)

    api_module.supported_minecraft_versions = supported_versions
    _install_plan_render(plan_render_module)


def _install_plan_render(module: Any) -> None:
    original = module.render_complete_plan
    if getattr(original, "_mmm_platform_visible", False):
        return

    @wraps(original)
    def render_complete_plan(*, requested_prompt, game_design, modules, acceptance_tests):
        rendered = original(
            requested_prompt=requested_prompt,
            game_design=game_design,
            modules=modules,
            acceptance_tests=acceptance_tests,
        )
        selection = game_design.get("_platform_selection") if isinstance(game_design, dict) else None
        if not isinstance(selection, dict):
            return rendered
        target = selection.get("target")
        if not isinstance(target, dict):
            return rendered
        korean = any("가" <= char <= "힣" for char in requested_prompt)
        mappings_kind = target.get("mappings_kind") or (
            "mojang" if target.get("mappings") == "mojang" else "named"
        )
        if korean:
            block = (
                "플랫폼 타깃\n"
                f"- Minecraft Java {target.get('minecraft_version')} / {target.get('loader')}\n"
                f"- Java {target.get('java_version')} / mappings {mappings_kind}:{target.get('mappings')}\n"
                f"- 선택 방식: {selection.get('source', '')}\n"
                f"- 선택 이유: {selection.get('reason', '')}"
            )
            marker = "\n\n플레이 흐름"
        else:
            block = (
                "Platform target\n"
                f"- Minecraft Java {target.get('minecraft_version')} / {target.get('loader')}\n"
                f"- Java {target.get('java_version')} / mappings {mappings_kind}:{target.get('mappings')}\n"
                f"- Selection: {selection.get('source', '')}\n"
                f"- Reason: {selection.get('reason', '')}"
            )
            marker = "\n\nPlayer loop"
        if marker in rendered:
            return rendered.replace(marker, "\n\n" + block + marker, 1)
        return block + "\n\n" + rendered

    render_complete_plan._mmm_platform_visible = True
    module.render_complete_plan = render_complete_plan


__all__ = ["install"]
