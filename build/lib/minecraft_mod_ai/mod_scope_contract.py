from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .mod_development_methods import resolve_mod_development_methods
from .spec import SpecValidationError

_WORLDGEN_MODULE_KINDS = frozenset({"structure", "biome", "dimension", "world_event"})
_INSTALL_MARKER = "_mmm_mod_scope_contract_installed"


def _effective_assets(
    complete_spec_module: Any,
    modules: tuple[Any, ...],
    assets: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Resolve the exact asset scope before a v2 production contract is bound."""

    from .resource_contracts import derive_module_asset_specs

    resolved = list(assets)
    resolved.extend(
        complete_spec_module.AssetRequest(**row)
        for row in derive_module_asset_specs(
            modules,
            existing_asset_ids=[asset.asset_id for asset in resolved],
        )
    )
    return tuple(resolved)


def _rebind_contract_to_effective_scope(
    *,
    requested_prompt: str,
    game_design: dict[str, Any],
    modules: tuple[Any, ...],
    assets: tuple[Any, ...],
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    """Recompile only an existing v2 contract against the finalized module/asset scope."""

    contract = game_design.get("_production_contract")
    if not isinstance(contract, Mapping):
        return game_design, None

    acceptance_catalog = contract.get("acceptance_catalog")
    input_acceptance = ()
    if isinstance(acceptance_catalog, list):
        input_acceptance = tuple(
            str(item["statement"])
            for item in acceptance_catalog
            if isinstance(item, Mapping)
            and item.get("origin") == "input"
            and isinstance(item.get("statement"), str)
            and item["statement"].strip()
        )

    research_brief = game_design.get("_research_brief")
    evidence_plan = game_design.get("_evidence_first_plan")

    from .production_contract import compile_production_contract

    compiled = compile_production_contract(
        requested_prompt=requested_prompt,
        game_design=game_design,
        research_brief=research_brief,
        modules=modules,
        assets=assets,
        acceptance_tests=input_acceptance,
        evidence_plan=evidence_plan if isinstance(evidence_plan, Mapping) else None,
    )
    return {**game_design, "_production_contract": compiled.contract}, compiled.acceptance_tests


def install(complete_spec_module: Any, complete_planner_module: Any) -> None:
    """Install mod-only scope on the live proposal construction boundary."""
    if getattr(complete_spec_module, _INSTALL_MARKER, False):
        return

    original_builder: Callable[..., Any] = complete_spec_module.complete_proposal_from_parts

    def scoped_complete_proposal_from_parts(
        *,
        requested_prompt: str,
        base_proposal: Any,
        game_design: dict[str, Any],
        modules: tuple[Any, ...],
        assets: tuple[Any, ...] = (),
        acceptance_tests: tuple[str, ...],
        existing_input_sha256: str = "",
    ) -> Any:
        method_plan = resolve_mod_development_methods(
            requested_prompt,
            existing_project=bool(existing_input_sha256),
        )
        if method_plan["standalone_map_requested"]:
            raise SpecValidationError(
                "Standalone map, world-save, schematic and Litematica outputs are outside "
                "M.M.M's mod project scope."
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
                "The planner attempted world or structure generation although the request "
                "did not select fabric_worldgen."
            )

        effective_assets = _effective_assets(complete_spec_module, modules, assets)
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
        scoped_design, rebound_acceptance = _rebind_contract_to_effective_scope(
            requested_prompt=requested_prompt,
            game_design=scoped_design,
            modules=modules,
            assets=effective_assets,
        )
        if rebound_acceptance is not None:
            acceptance_tests = rebound_acceptance

        return original_builder(
            requested_prompt=requested_prompt,
            base_proposal=base_proposal,
            game_design=scoped_design,
            modules=modules,
            assets=effective_assets,
            acceptance_tests=acceptance_tests,
            existing_input_sha256=existing_input_sha256,
        )

    complete_spec_module.complete_proposal_from_parts = scoped_complete_proposal_from_parts
    complete_planner_module.complete_proposal_from_parts = scoped_complete_proposal_from_parts
    setattr(complete_spec_module, _INSTALL_MARKER, True)
