from __future__ import annotations

from functools import wraps
from typing import Any

from .platform_catalog import (
    PLATFORM_ADAPTERS,
    adapter_for_target,
    supported_minecraft_versions as discover_supported_minecraft_versions,
)


def install(api_module: Any, plan_render_module: Any) -> None:
    """Install only the Python/API platform surface."""
    api_module.SUPPORTED_MINECRAFT_VERSIONS = tuple(
        item.minecraft_version for item in PLATFORM_ADAPTERS
    )

    def supported_versions() -> tuple[str, ...]:
        return discover_supported_minecraft_versions(loader="fabric")

    api_module.supported_minecraft_versions = supported_versions
    _install_complete_session(api_module)
    _install_legacy_session(api_module)
    _install_plan_render(plan_render_module)


def _install_complete_session(api_module: Any) -> None:
    cls = api_module.CompleteModAISession
    original = cls.__init__
    if getattr(original, "_mmm_dynamic_platform_session", False):
        return

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        explicit_version = kwargs.pop("minecraft_version", None)
        if explicit_version is not None:
            explicit_version = str(explicit_version).strip()
            if explicit_version.casefold() in {"", "auto", "automatic"}:
                explicit_version = None
            else:
                try:
                    adapter_for_target(explicit_version, "fabric")
                except ValueError as exc:
                    raise api_module.SpecValidationError(str(exc)) from exc
        # Historical constructor compatibility only; planner target selection is
        # authoritative after initialization and does not inherit this placeholder.
        kwargs["minecraft_version"] = "1.20.1"
        original(self, *args, **kwargs)

        if explicit_version:
            self.router._mmm_requested_minecraft_version = explicit_version
            self.router._mmm_requested_loader = "fabric"

        if self.existing_input is not None:
            from .importer import inspect_existing_project_archive

            report = inspect_existing_project_archive(self.existing_input)
            if report.minecraft_version:
                self.router._mmm_existing_minecraft_version = report.minecraft_version
            if report.loader:
                self.router._mmm_existing_loader = report.loader
            self._mmm_existing_platform_report = {
                "minecraft_version": report.minecraft_version,
                "minecraft_versions": list(report.minecraft_versions),
                "loader": report.loader,
                "source": str(self.existing_input),
            }

    init._mmm_dynamic_platform_session = True
    cls.__init__ = init


def _install_legacy_session(api_module: Any) -> None:
    cls = api_module.ModAISession
    original = cls.__init__
    if getattr(original, "_mmm_dynamic_platform_session", False):
        return

    @wraps(original)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        explicit = kwargs.get("minecraft_version")
        if explicit is not None:
            explicit = str(explicit).strip()
            if explicit and explicit.casefold() not in {"auto", "automatic"}:
                try:
                    adapter_for_target(explicit, "fabric")
                except ValueError as exc:
                    raise api_module.SpecValidationError(str(exc)) from exc
            else:
                kwargs["minecraft_version"] = "1.20.1"
        original(self, *args, **kwargs)

    init._mmm_dynamic_platform_session = True
    cls.__init__ = init


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
            "mojang" if target.get("mappings") == "mojang" else "yarn"
        )
        if korean:
            block = (
                "플랫폼 타깃\n"
                f"- Minecraft Java {target.get('minecraft_version')} / {target.get('loader')}\n"
                f"- Java {target.get('java_version')} / mappings {mappings_kind}:{target.get('mappings')}\n"
                f"- Fabric Loader {target.get('fabric_loader')} / Fabric API {target.get('fabric_api')}\n"
                f"- 선택 방식: {selection.get('source', '')}\n"
                f"- 선택 이유: {selection.get('reason', '')}"
            )
            marker = "\n\n플레이 흐름"
        else:
            block = (
                "Platform target\n"
                f"- Minecraft Java {target.get('minecraft_version')} / {target.get('loader')}\n"
                f"- Java {target.get('java_version')} / mappings {mappings_kind}:{target.get('mappings')}\n"
                f"- Fabric Loader {target.get('fabric_loader')} / Fabric API {target.get('fabric_api')}\n"
                f"- Selection: {selection.get('source', '')}\n"
                f"- Reason: {selection.get('reason', '')}"
            )
            marker = "\n\nPlayer loop"
        if marker in rendered:
            return rendered.replace(marker, "\n\n" + block + marker, 1)
        return block + "\n\n" + rendered

    render_complete_plan._mmm_platform_visible = True
    module.render_complete_plan = render_complete_plan
