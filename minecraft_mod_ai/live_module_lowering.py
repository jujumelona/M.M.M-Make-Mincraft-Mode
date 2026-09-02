from __future__ import annotations

"""Explicit lowering of semantic production modules for validated live targets.

Unlike the removed platform contract, this module does not install wrappers or select a
platform. It is a pure post-planning transformation over an already validated proposal.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .complete_spec import CompleteProposal, ProductionModule
from .spec import SpecValidationError

_LIVE_NON_SOURCE_KINDS = frozenset({"integration"})


def lower_live_modules(planner: Any, result: CompleteProposal) -> CompleteProposal:
    selection = result.game_design.get("_platform_selection", {})
    target = selection.get("target", {}) if isinstance(selection, dict) else {}
    if not isinstance(target, dict) or target.get("source_api_family") != "fabric_live_ai":
        return result

    migration_requested = bool(
        isinstance(selection, dict) and selection.get("migration_requested")
    )
    migration_from = selection.get("migration_from") if isinstance(selection, dict) else None
    if not migration_requested and _validated_retain_only(result):
        return result

    bootstrap_contents = _bootstrap_content_payload(result)
    bootstrap_boss = _bootstrap_boss_payload(result)
    lowered: list[ProductionModule] = []
    changed = False
    bootstrap_bound = False

    for item in result.modules:
        uses_base_content = item.kind == "integration" and isinstance(
            item.config.get("uses_base_content"), list
        )
        if uses_base_content:
            lowered.append(
                ProductionModule(
                    module_id=item.module_id,
                    kind="custom_java",
                    config={
                        **item.config,
                        "implementation": "custom",
                        "requested_kind": "bootstrap_content",
                        "platform_generation": "canonical_live_target",
                        "bootstrap_contents": bootstrap_contents,
                        "bootstrap_boss": bootstrap_boss,
                        "require_exact_base_spec": True,
                    },
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
        lowered.append(_as_custom_carrier(item, extra_config={}))
        changed = True

    if (bootstrap_contents or bootstrap_boss) and not bootstrap_bound:
        target_index = _carrier_index(lowered)
        if target_index is None:
            raise SpecValidationError(
                "Live target has base ModSpec content but no production module that can carry it."
            )
        lowered[target_index] = _as_custom_carrier(
            lowered[target_index],
            extra_config={
                "bootstrap_contents": bootstrap_contents,
                "bootstrap_boss": bootstrap_boss,
                "require_exact_base_spec": True,
            },
        )
        changed = True

    if migration_requested:
        target_index = _carrier_index(lowered)
        if target_index is None:
            raise SpecValidationError(
                "Version migration requires at least one source-generation module."
            )
        if not isinstance(migration_from, dict):
            raise SpecValidationError(
                "Version migration requires an explicit validated source target receipt."
            )
        lowered[target_index] = _as_custom_carrier(
            lowered[target_index],
            extra_config={
                "platform_migration": {
                    "from": dict(migration_from),
                    "to": dict(target),
                    "requirements": [
                        "migrate build and loader metadata to the approved target",
                        "port API usage using target-scoped official evidence",
                        "preserve requested behavior and existing project content",
                        "finish only after language, build and game tests pass",
                    ],
                }
            },
        )
        changed = True

    if not changed:
        return result

    lowered_tuple = tuple(lowered)
    game_design = {
        **result.game_design,
        "_platform_execution": {
            "mode": "canonical_compile_repair",
            "source_api_family": "fabric_live_ai",
            "semantic_kinds_preserved_in": "module.config.requested_kind",
            "base_modspec_bound_to_live_generation": bool(
                bootstrap_contents or bootstrap_boss
            ),
            "migration_bound_to_live_generation": migration_requested,
            "production_contract_rebound_after_lowering": True,
        },
    }
    game_design, acceptance_tests = _recompile_live_contract(
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
    ).with_hash()
    updated.validate(policy=getattr(planner, "policy", None))
    return updated


def _bootstrap_content_payload(result: CompleteProposal) -> list[dict[str, Any]]:
    return [
        {
            "content_id": content.content_id,
            "kind": content.kind.value,
            "display_name_en": content.display_name_en,
            "display_name_ko": content.display_name_ko,
            "color": content.color,
            "recipe": content.recipe,
        }
        for content in result.base_proposal.spec.contents
    ]


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
    return tuple(result.acceptance_tests)


def _recompile_live_contract(
    result: CompleteProposal,
    *,
    game_design: dict[str, Any],
    lowered: tuple[ProductionModule, ...],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    from . import production_contract

    contract_design = {
        key: value for key, value in game_design.items() if not str(key).startswith("_")
    }
    research_brief = game_design.get("_research_brief")
    evidence_plan = game_design.get("_evidence_first_plan")
    compiled = production_contract.compile_production_contract(
        requested_prompt=result.requested_prompt,
        game_design=contract_design,
        research_brief=research_brief if isinstance(research_brief, dict) else None,
        modules=lowered,
        assets=result.assets,
        acceptance_tests=_input_acceptance_tests(result),
        evidence_plan=evidence_plan if isinstance(evidence_plan, Mapping) else None,
    )
    return (
        {**game_design, "_production_contract": compiled.contract},
        tuple(compiled.acceptance_tests),
    )


def _carrier_index(modules: list[ProductionModule]) -> int | None:
    return next(
        (index for index, item in enumerate(modules) if item.kind == "custom_java"),
        None,
    )


def _as_custom_carrier(
    item: ProductionModule,
    *,
    extra_config: dict[str, Any],
) -> ProductionModule:
    config = {
        **item.config,
        "implementation": "custom",
        "requested_kind": item.config.get("requested_kind", item.kind),
        "platform_generation": "canonical_live_target",
        **extra_config,
    }
    return ProductionModule(
        module_id=item.module_id,
        kind="custom_java",
        config=config,
        depends_on=item.depends_on,
        required_gates=item.required_gates,
    )


def _validated_retain_only(result: CompleteProposal) -> bool:
    if result.modules:
        return False
    plan = result.game_design.get("_evidence_first_plan")
    if not isinstance(plan, Mapping):
        return False
    from .evidence_first_planning import validate_evidence_first_plan

    validate_evidence_first_plan(plan, prompt=result.requested_prompt)
    return (
        bool(plan.get("verified_provides"))
        and not plan.get("gap_catalog")
        and not plan.get("tasks")
    )


__all__ = ["lower_live_modules"]
