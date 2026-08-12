from __future__ import annotations

from functools import wraps
from typing import Any

from .platform_catalog import (
    PLATFORM_ADAPTERS,
    adapter_for_target,
    supported_minecraft_versions as discover_supported_minecraft_versions,
)


def install(api_module: Any, plan_render_module: Any) -> None:
    api_module.SUPPORTED_MINECRAFT_VERSIONS = tuple(
        item.minecraft_version for item in PLATFORM_ADAPTERS
    )

    def supported_versions() -> tuple[str, ...]:
        return discover_supported_minecraft_versions(loader="fabric")

    api_module.supported_minecraft_versions = supported_versions
    _install_complete_session(api_module)
    _install_legacy_session(api_module)
    _install_plan_render(plan_render_module)

    from . import complete_orchestrator as orchestrator_module
    from .platform_live_execution_contract import install as install_live_execution

    install_live_execution(orchestrator_module)

    from . import runtime_manager as runtime_manager_module
    from .minecraft_mcp_runtime_helper_contract import install as install_runtime_helpers

    install_runtime_helpers(runtime_manager_module)

    # Route-scoped validation is installed before any external MCP call: search and
    # migration responses may mention many historical versions, while reviewed
    # runtime status tools are allowed to prove the actual running target.
    from . import external_mcp_router as external_mcp_router_module
    from .external_mcp_target_validation_contract import (
        install as install_mcp_target_validation,
    )

    install_mcp_target_validation(external_mcp_router_module)

    from .minecraft_mcp_runtime_contract import install as install_mcp_runtime

    install_mcp_runtime(orchestrator_module)

    from . import complete_planner as complete_planner_module
    from . import custom_module_generator as custom_module_generator_module
    from . import mcp_tools as mcp_tools_module
    from . import repair_engine as repair_engine_module
    from .minecraft_mcp_federation_contract import install as install_mcp_federation

    install_mcp_federation(
        complete_planner_module=complete_planner_module,
        custom_module_generator_module=custom_module_generator_module,
        repair_engine_module=repair_engine_module,
        mcp_tools_module=mcp_tools_module,
    )

    from . import minecraft_mcp_repair_batch_contract as repair_batch_module
    from .minecraft_mcp_repair_batch_contract import install as install_mcp_repair_batch
    from .mcp_repair_diagnostic_shape_contract import (
        install as install_mcp_repair_diagnostic_shape,
    )

    install_mcp_repair_batch(repair_engine_module)
    install_mcp_repair_diagnostic_shape(repair_batch_module)

    from . import skill_catalog as skill_catalog_module
    from .platform_skill_policy_contract import install as install_skill_policy

    install_skill_policy(skill_catalog_module)


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
