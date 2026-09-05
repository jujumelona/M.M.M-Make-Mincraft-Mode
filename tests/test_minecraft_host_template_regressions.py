from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from minecraft_mod_ai import evidence_first_planning as planning
from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability
from minecraft_mod_ai.minecraft_template_steps import steps_for_profile
from minecraft_mod_ai.semantic_batching_contract import build_bounded_requirement_catalog


class _SemanticQueueRouter:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def generate_tool_decision(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if not self.outputs:
            raise AssertionError("semantic router was called more times than expected")
        return self.outputs.pop(0)


def _semantic_leaf(
    source_clause_index: int,
    capability: str,
    anchor: str,
    *,
    given: str = "the authored precondition holds",
    when: str = "the authored action occurs",
    then: str = "the authored outcome is observed",
) -> dict[str, Any]:
    return {
        "requirements": [
            {
                "source_clause_index": source_clause_index,
                "capability_id": capability,
                "source_anchor": anchor,
                "semantic_statement": anchor,
                "given": given,
                "when": when,
                "then": then,
                "semantic_type": "gameplay_mechanic",
            }
        ]
    }


def _step_names(capability: str) -> tuple[str, ...]:
    profile = profile_for_capability(capability)
    return tuple(step.name for step in steps_for_profile(profile))


def test_special_mineral_is_worldgen_resource_not_alien_combat() -> None:
    profile = profile_for_capability("planet.special_mineral")
    assert profile.template_id == "worldgen_resource"

    steps = steps_for_profile(profile)
    names = {step.name for step in steps}
    assert {
        "semantic_contract",
        "resource_registry",
        "configured_feature",
        "placed_feature",
        "biome_dimension_binding",
        "mining_loot_acquisition",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert "entity_attributes_spawn" not in names
    assert "ai_damage_death" not in names
    assert all("aggro" not in step.outcome.casefold() for step in steps)
    assert all("combat" not in step.outcome.casefold() for step in steps)


def test_space_travel_has_dedicated_multistage_template_not_generic_fallback() -> None:
    profile = profile_for_capability("space.travel")
    assert profile.template_id == "space_travel"

    names = set(_step_names("space.travel"))
    assert {
        "semantic_contract",
        "launch_unlock_policy",
        "fuel_destination_transaction",
        "world_transition",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert "semantic_implementation" not in names
    assert len(names) >= 6


def test_unknown_capability_still_compiles_to_multistep_host_template() -> None:
    capability = "custom.semantic_0123456789abcdef"
    profile = profile_for_capability(capability)
    assert profile.template_id == "custom_gameplay"

    names = set(_step_names(capability))
    assert {
        "semantic_contract",
        "authoritative_behavior",
        "integration_binding",
        "failure_contract",
        "runtime_scenario",
    } <= names
    assert len(names) >= 5
    assert "semantic_implementation" not in names


def test_production_semantic_catalog_binds_resource_and_economy_to_spacecraft() -> None:
    prompt = (
        "Gather farm resources.\n"
        "Earn credits.\n"
        "Trade with merchants.\n"
        "Construct spacecraft components."
    )
    router = _SemanticQueueRouter(
        [
            _semantic_leaf(0, "resource.farming", "Gather farm resources"),
            _semantic_leaf(1, "economy.currency", "Earn credits"),
            _semantic_leaf(2, "economy.trade", "Trade with merchants"),
            _semantic_leaf(
                3,
                "spacecraft.component_construction",
                "Construct spacecraft components",
            ),
        ]
    )

    catalog = build_bounded_requirement_catalog(prompt, router=router)
    assert router.calls == 4
    requirements = {
        str(item["capability"]): item
        for item in catalog["requirements"]
        if isinstance(item, Mapping)
    }
    construction = requirements["spacecraft.component_construction"]
    expected_capabilities = {
        "resource.farming",
        "economy.currency",
        "economy.trade",
    }
    assert expected_capabilities <= set(
        construction["unlock_policy"]["required_capabilities"]
    )
    expected_refs = {
        str(requirements[capability]["requirement_id"])
        for capability in expected_capabilities
    }
    assert expected_refs <= set(construction["depends_on"])
    assert expected_refs <= set(
        construction["unlock_policy"]["required_requirement_refs"]
    )
    assert construction["dependency_provenance"]["owner"] == (
        "host_minecraft_feature_model"
    )


def test_validation_recompiles_templates_with_tracing_disabled(monkeypatch) -> None:
    prompt = "Travel to another planet."
    router = _SemanticQueueRouter(
        [
            _semantic_leaf(
                0,
                "space.travel",
                "Travel to another planet",
                given="a valid spacecraft and destination exist",
                when="the player launches",
                then="the player arrives at the destination",
            )
        ]
    )
    catalog = build_bounded_requirement_catalog(prompt, router=router)
    game_design = {
        "mod_id": "template_regression",
        "_evidence_request_catalog": catalog,
    }
    target = {
        "target": {
            "minecraft_version": "1.21.8",
            "loader": "fabric",
            "java_version": 21,
            "source_api_family": "fabric",
        },
        "reason": "host regression fixture",
    }
    plan = planning.compile_evidence_first_plan(
        prompt,
        game_design,
        target_decision=target,
    )

    original_compile_tasks = planning._compile_tasks
    observed_emit_trace: list[bool] = []

    def recording_compile_tasks(*args: Any, **kwargs: Any):
        observed_emit_trace.append(bool(kwargs.get("emit_trace", True)))
        return original_compile_tasks(*args, **kwargs)

    monkeypatch.setattr(planning, "_compile_tasks", recording_compile_tasks)
    planning.validate_evidence_first_plan(plan, prompt=prompt)

    assert observed_emit_trace
    assert observed_emit_trace == [False]
