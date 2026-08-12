from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

from .complete_spec import MODULE_KINDS, CompleteProposal, ProductionModule
from .platform_resolver import resolve_platform, retarget_proposal


_LIVE_NON_SOURCE_KINDS = frozenset({"audio", "integration"})


def install(*, game_design_module: Any, complete_planner_module: Any) -> None:
    """Make live platform selection/model generation the outermost planning contract."""

    _make_old_live_kind_gate_non_blocking()
    _install_central_target_choice(game_design_module)
    _install_live_module_lowering(complete_planner_module)


def _make_old_live_kind_gate_non_blocking() -> None:
    # platform_planning_contract predates live AI targets and contains a final
    # deterministic-kind gate. Keep that gate for the two legacy optimized source
    # families, but expose all semantic module kinds to that one compatibility check
    # for live targets. Actual live execution is lowered to custom_java below.
    from . import platform_planning_contract as legacy_contract

    current = legacy_contract.adapter_for_target
    if getattr(current, "_mmm_live_ai_kind_gate", False):
        return

    @wraps(current)
    def adapter_for_target(version: str, loader: str = "fabric"):
        adapter = current(version, loader)
        if adapter.source_api_family == "fabric_live_ai":
            return replace(
                adapter,
                deterministic_module_kinds=frozenset(MODULE_KINDS),
            )
        return adapter

    adapter_for_target._mmm_live_ai_kind_gate = True
    legacy_contract.adapter_for_target = adapter_for_target


def _install_central_target_choice(module: Any) -> None:
    cls = module.GameDesignPlanner
    original = cls.plan
    if getattr(original, "_mmm_central_live_platform_choice", False):
        return

    @wraps(original)
    def plan(self: Any, prompt: str, *, media_paths=()):
        design, proposal = original(self, prompt, media_paths=media_paths)
        router = self.router
        existing_version = getattr(router, "_mmm_existing_minecraft_version", None)
        existing_loader = getattr(router, "_mmm_existing_loader", None)
        requested_version = getattr(router, "_mmm_requested_minecraft_version", None)
        requested_loader = getattr(router, "_mmm_requested_loader", None)
        effective_prompt = str(prompt)
        if requested_version and str(requested_version) not in effective_prompt:
            effective_prompt += (
                f"\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]"
            )
        if requested_loader and str(requested_loader).casefold() not in effective_prompt.casefold():
            effective_prompt += (
                f"\n[HOST_LOADER_CONSTRAINT {requested_loader}]"
            )

        selection = resolve_platform(
            effective_prompt,
            design=design,
            existing_version=existing_version,
            existing_loader=existing_loader,
            router=router,
        )
        proposal = retarget_proposal(proposal, selection)
        selection_dict = selection.to_dict()
        research_brief = design.get("_research_brief")
        if isinstance(research_brief, dict):
            research_brief = {
                **research_brief,
                "_mmm_platform_target": dict(selection_dict["target"]),
            }
        design = {
            **design,
            "_platform_selection": selection_dict,
            **(
                {"_research_brief": research_brief}
                if isinstance(research_brief, dict)
                else {}
            ),
        }
        return design, proposal

    plan._mmm_central_live_platform_choice = True
    cls.plan = plan


def _install_live_module_lowering(module: Any) -> None:
    cls = module.CompleteGameDesignPlanner
    original = cls._plan_in_session
    if getattr(original, "_mmm_live_ai_module_lowering", False):
        return

    @wraps(original)
    def plan_in_session(
        self: Any,
        prompt: str,
        *,
        media_paths=(),
        existing_input_sha256="",
    ):
        result = original(
            self,
            prompt,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
        )
        target = result.game_design.get("_platform_selection", {}).get("target", {})
        if not isinstance(target, dict) or target.get("source_api_family") != "fabric_live_ai":
            return result

        lowered: list[ProductionModule] = []
        changed = False
        for item in result.modules:
            if item.kind in _LIVE_NON_SOURCE_KINDS or item.kind == "custom_java":
                lowered.append(item)
                continue
            config = {
                **item.config,
                "implementation": "custom",
                "requested_kind": item.kind,
                "platform_generation": "central_ai_live_target",
            }
            lowered.append(
                ProductionModule(
                    module_id=item.module_id,
                    kind="custom_java",
                    config=config,
                    depends_on=item.depends_on,
                    required_gates=item.required_gates,
                )
            )
            changed = True

        if not changed:
            return result

        game_design = {
            **result.game_design,
            "_platform_execution": {
                "mode": "central_ai_compile_repair",
                "source_api_family": "fabric_live_ai",
                "semantic_kinds_preserved_in": "module.config.requested_kind",
            },
        }
        updated: CompleteProposal = replace(
            result,
            game_design=game_design,
            modules=tuple(lowered),
            approval_hash="",
        )
        updated = updated.with_hash()
        updated.validate(policy=getattr(self, "policy", None))
        return updated

    plan_in_session._mmm_live_ai_module_lowering = True
    cls._plan_in_session = plan_in_session
