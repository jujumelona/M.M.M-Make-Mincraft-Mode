from __future__ import annotations

from minecraft_mod_ai import deep_design_execution_contract as deep
from minecraft_mod_ai import evidence_first_planning as evidence


def _branches() -> dict[str, dict[str, str]]:
    return {
        name: {"status": "NOT_APPLICABLE"}
        for name in (
            "needs_registry",
            "needs_datagen",
            "needs_persistence",
            "needs_network",
            "needs_client_render",
            "needs_worldgen",
            "needs_mixin",
            "needs_loader_leaf",
        )
    }


def test_design_modules_become_leaf_steps_plus_integration() -> None:
    context = (
        {
            "requirement_ref": "req_parent",
            "parent_capability": "alien_planet_interaction",
            "capability": "design.module.alien_encounter",
            "detail": "Spawn and resolve hostile alien encounters on visited planets.",
            "source": "game_design.modules[0]",
            "reuse_refs": [],
            "reuse_mode": "fresh",
            "proof_level": "",
        },
        {
            "requirement_ref": "req_parent",
            "parent_capability": "alien_planet_interaction",
            "capability": "design.module.colony_establishment",
            "detail": "Establish and persist a player colony after the planet is secured.",
            "source": "game_design.modules[1]",
            "reuse_refs": ["github:verified-colony-source"],
            "reuse_mode": "source_transplant",
            "proof_level": "PINNED",
        },
    )
    token = deep._ACTIVE_DESIGN_EXECUTION.set(context)
    try:
        steps = evidence._semantic_steps("alien_planet_interaction", _branches())
    finally:
        deep._ACTIVE_DESIGN_EXECUTION.reset(token)

    assert len(steps) == 3
    assert steps[0].name == "design_leaf_1"
    assert steps[1].name == "design_leaf_2"
    assert steps[2].name == "design_integration"
    assert steps[0].provides[0].startswith("design_leaf:")
    assert steps[1].provides[0].startswith("design_leaf:")
    assert set(steps[2].consumes) == {steps[0].provides[0], steps[1].provides[0]}
    assert steps[2].provides == ("alien_planet_interaction",)


def test_concrete_module_facets_replace_duplicate_narrative_coder_facets() -> None:
    design = {
        "_evidence_request_catalog": {
            "requirements": [
                {
                    "requirement_id": "req_parent",
                    "capability": "alien_planet_interaction",
                }
            ]
        },
        "_pre_retrieval_plan": {
            "design_retrieval_facets": [
                {
                    "capability": "design.module.alien_encounter",
                    "requirement_ref": "req_parent",
                    "source": "game_design.modules[0]",
                },
                {
                    "capability": "design.core_loop.visit_planet",
                    "requirement_ref": "req_parent",
                    "source": "game_design.core_loop[0]",
                },
            ]
        },
        "modules": [
            {
                "plugin_id": "alien_encounter",
                "status": "custom",
                "reason": "Spawn and resolve hostile alien encounters on visited planets.",
            }
        ],
        "core_loop": ["Visit a planet and resolve its encounter."],
        "progression": [],
        "combat": {},
        "mod_context": {},
    }
    reuse_plan = {
        "capabilities": [
            {
                "capability": "design.module.alien_encounter",
                "mode": "source_transplant",
                "source_id": "github:alien-encounter-source",
                "proof_level": "PINNED",
            }
        ]
    }

    context = deep._execution_context(design, reuse_plan)

    assert [item["capability"] for item in context] == [
        "design.module.alien_encounter"
    ]
    assert context[0]["parent_capability"] == "alien_planet_interaction"
    assert context[0]["reuse_refs"] == ["github:alien-encounter-source"]


def test_runtime_installs_research_first_design_generator_without_replacing_plan_owner() -> None:
    from minecraft_mod_ai import game_design

    assert getattr(
        game_design._generate_game_design_once,
        "__mmm_research_first_design_generator__",
        False,
    )
    assert getattr(game_design.GameDesignPlanner.plan, "_mmm_host_owned_template", False)
