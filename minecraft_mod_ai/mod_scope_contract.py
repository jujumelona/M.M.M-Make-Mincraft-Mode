from __future__ import annotations

from typing import Any, Callable

from .mod_development_methods import resolve_mod_development_methods
from .spec import SpecValidationError

_WORLDGEN_MODULE_KINDS = frozenset({"structure", "biome", "dimension", "world_event"})
_INSTALL_MARKER = "_mmm_mod_scope_contract_installed"


def install(complete_spec_module: Any, complete_planner_module: Any) -> None:
    """Install mod-only scope without composing platform contracts."""
    if getattr(complete_spec_module, _INSTALL_MARKER, False):
        return

    original_prompt: Callable[..., str] = complete_planner_module._implementation_prompt
    original_builder: Callable[..., Any] = complete_spec_module.complete_proposal_from_parts

    def scoped_implementation_prompt(prompt: str, game_design: dict[str, Any]) -> str:
        method_plan = resolve_mod_development_methods(prompt)
        enriched_design = {**game_design, "_mod_development_methods": method_plan}
        target = game_design.get("_platform_selection")
        target_text = ""
        if isinstance(target, dict) and isinstance(target.get("target"), dict):
            selected = target["target"]
            target_text = (
                " The exact host-verified target for this plan is "
                f"Minecraft {selected.get('minecraft_version')} / {selected.get('loader')} / "
                f"Java {selected.get('java_version')} / {selected.get('mappings')}. "
                "Do not substitute APIs from another version. For a live-discovered target, "
                "use the supplied official target evidence and compiler feedback rather than a "
                "hard-coded historical source template."
            )
        return (
            original_prompt(prompt, enriched_design)
            + "\n\nThe resolved mod-development method plan is authoritative for implementation scope. "
            "Generate only a mod project for the selected platform target. Never emit a standalone "
            "map, world save, world ZIP, schematic, Litematica file, BuildSpec, NPZ block delta, or "
            "external Builder handoff. Native worldgen modules are mod-owned code and datapack "
            "resources only when fabric_worldgen was selected from the request."
            + target_text
        )

    def scoped_complete_proposal_from_parts(
        *,
        requested_prompt: str,
        base_proposal: Any,
        game_design: dict[str, Any],
        modules: tuple[Any, ...],
        assets: tuple[Any, ...] = (),
        audio: tuple[Any, ...] = (),
        acceptance_tests: tuple[str, ...],
        existing_input_sha256: str = "",
    ) -> Any:
        method_plan = resolve_mod_development_methods(
            requested_prompt,
            existing_project=bool(existing_input_sha256),
        )
        if method_plan["standalone_map_requested"]:
            raise SpecValidationError(
                "Standalone map, world-save, schematic and Litematica outputs are outside M.M.M's mod project scope."
            )
        method_ids = frozenset(method_plan["method_ids"])
        worldgen_selected = "fabric_worldgen" in method_ids
        worldgen_modules = tuple(
            module.module_id
            for module in modules
            if module.kind in _WORLDGEN_MODULE_KINDS
            or (
                module.kind == "custom_java"
                and module.config.get("requested_kind") in _WORLDGEN_MODULE_KINDS
            )
        )
        if not worldgen_selected and worldgen_modules:
            raise SpecValidationError(
                "The planner attempted world or structure generation although the request did not select fabric_worldgen."
            )
        scoped_design = {
            **game_design,
            "_mod_development_methods": method_plan,
            "_product_scope": {
                "kind": "minecraft_mod_project",
                "standalone_map_generation": False,
                "worldgen_selected": worldgen_selected,
                "platform": game_design.get("_platform_selection", {}),
            },
        }
        return original_builder(
            requested_prompt=requested_prompt,
            base_proposal=base_proposal,
            game_design=scoped_design,
            modules=modules,
            assets=assets,
            audio=audio,
            acceptance_tests=acceptance_tests,
            existing_input_sha256=existing_input_sha256,
        )

    complete_planner_module._implementation_prompt = scoped_implementation_prompt
    complete_spec_module.complete_proposal_from_parts = scoped_complete_proposal_from_parts
    complete_planner_module.complete_proposal_from_parts = scoped_complete_proposal_from_parts
    setattr(complete_spec_module, _INSTALL_MARKER, True)
