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


def _bootstrap_content_payload(result: CompleteProposal) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for content in result.base_proposal.spec.contents:
        payload.append(
            {
                "content_id": content.content_id,
                "kind": content.kind.value,
                "display_name_en": content.display_name_en,
                "display_name_ko": content.display_name_ko,
                "color": content.color,
                "recipe": content.recipe,
            }
        )
    return payload


def _bootstrap_boss_payload(result: CompleteProposal) -> dict[str, Any] | None:
    boss = result.base_proposal.spec.boss
    if boss is None:
        return None
    return {
        "entity_id": boss.entity_id,
        "display_name_en": boss.display_name_en,
        "display_name_ko": boss.display_name_ko,
        "max_health": boss.max_health,
        "attack_damage": boss.attack_damage,
        "movement_speed": boss.movement_speed,
        "scale": boss.scale,
        "primary_color": boss.primary_color,
        "secondary_color": boss.secondary_color,
        "model_kind": boss.model_kind,
    }


def _input_acceptance_tests(result: CompleteProposal) -> tuple[str, ...]:
    """Recover only planner-authored tests before the code-owned quality expansion."""

    contract = result.game_design.get("_production_contract")
    if isinstance(contract, dict):
        catalog = contract.get("acceptance_catalog")
        if isinstance(catalog, list):
            values = tuple(
                str(item.get("statement", "")).strip()
                for item in catalog
                if isinstance(item, dict)
                and item.get("origin") == "input"
                and str(item.get("statement", "")).strip()
            )
            if values:
                return values
    # Saved/legacy proposal fallback: recompiling from all tests is preferable to
    # leaving a stale module implementation catalog. Normal planning always has the
    # structured acceptance catalog above.
    return tuple(result.acceptance_tests)


def _recompile_live_contract(
    module: Any,
    result: CompleteProposal,
    *,
    game_design: dict[str, Any],
    lowered: tuple[ProductionModule, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    contract_design = {
        key: value
        for key, value in game_design.items()
        if not str(key).startswith("_")
    }
    research_brief = game_design.get("_research_brief")
    compiled = module.compile_production_contract(
        requested_prompt=result.requested_prompt,
        game_design=contract_design,
        research_brief=(research_brief if isinstance(research_brief, dict) else None),
        modules=lowered,
        assets=result.assets,
        audio=result.audio,
        acceptance_tests=_input_acceptance_tests(result),
    )
    rebound_design = {
        **game_design,
        "_production_contract": compiled.contract,
    }
    return rebound_design, tuple(compiled.acceptance_tests)


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

        bootstrap_contents = _bootstrap_content_payload(result)
        bootstrap_boss = _bootstrap_boss_payload(result)
        lowered: list[ProductionModule] = []
        changed = False
        bootstrap_bound = False

        for item in result.modules:
            uses_base_content = (
                item.kind == "integration"
                and isinstance(item.config.get("uses_base_content"), list)
            )
            if uses_base_content:
                config = {
                    **item.config,
                    "implementation": "custom",
                    "requested_kind": "bootstrap_content",
                    "platform_generation": "central_ai_live_target",
                    "bootstrap_contents": bootstrap_contents,
                    "bootstrap_boss": bootstrap_boss,
                    "require_exact_base_spec": True,
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
                bootstrap_bound = True
                changed = True
                continue

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

        # _remove_bootstrap_duplicates intentionally removes modules duplicated by
        # ModSpec.contents because the historical deterministic generator creates
        # those files. A live target starts from Fabric's official blank template,
        # so bind the exact base spec into an existing source-generating module if
        # the compatibility sentinel was not emitted.
        if (bootstrap_contents or bootstrap_boss) and not bootstrap_bound:
            target_index = next(
                (
                    index
                    for index, item in enumerate(lowered)
                    if item.kind == "custom_java"
                ),
                None,
            )
            if target_index is None:
                target_index = next(
                    (
                        index
                        for index, item in enumerate(lowered)
                        if item.kind != "audio"
                        and not (
                            item.kind == "integration"
                            and item.config.get("integration_type")
                            in {"mmm_research_shard", "mmm_local_ai_sidecar"}
                        )
                    ),
                    None,
                )
            if target_index is None:
                raise module.SpecValidationError(
                    "Live target has base ModSpec content but no production module "
                    "that can carry its implementation."
                )
            carrier = lowered[target_index]
            config = {
                **carrier.config,
                "implementation": "custom",
                "requested_kind": carrier.config.get("requested_kind", carrier.kind),
                "platform_generation": "central_ai_live_target",
                "bootstrap_contents": bootstrap_contents,
                "bootstrap_boss": bootstrap_boss,
                "require_exact_base_spec": True,
            }
            lowered[target_index] = ProductionModule(
                module_id=carrier.module_id,
                kind="custom_java",
                config=config,
                depends_on=carrier.depends_on,
                required_gates=carrier.required_gates,
            )
            changed = True

        if not changed:
            return result

        lowered_tuple = tuple(lowered)
        game_design = {
            **result.game_design,
            "_platform_execution": {
                "mode": "central_ai_compile_repair",
                "source_api_family": "fabric_live_ai",
                "semantic_kinds_preserved_in": "module.config.requested_kind",
                "base_modspec_bound_to_live_generation": bool(
                    bootstrap_contents or bootstrap_boss
                ),
                "production_contract_rebound_after_lowering": True,
            },
        }
        game_design, acceptance_tests = _recompile_live_contract(
            module,
            result,
            game_design=game_design,
            lowered=lowered_tuple,
        )
        updated: CompleteProposal = replace(
            result,
            game_design=game_design,
            modules=lowered_tuple,
            acceptance_tests=acceptance_tests,
            approval_hash="",
        )
        updated = updated.with_hash()
        updated.validate(policy=getattr(self, "policy", None))
        return updated

    plan_in_session._mmm_live_ai_module_lowering = True
    cls._plan_in_session = plan_in_session
