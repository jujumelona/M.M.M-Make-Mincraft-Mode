from __future__ import annotations

from minecraft_mod_ai import deep_design_execution_contract as deep


def _design() -> dict:
    return {
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
                    "capability": "design.module.colony_establishment",
                    "requirement_ref": "req_parent",
                    "source": "game_design.modules[1]",
                },
            ]
        },
        "modules": [
            {
                "plugin_id": "alien_encounter",
                "status": "custom",
                "reason": "Spawn and resolve hostile alien encounters on visited planets.",
            },
            {
                "plugin_id": "colony_establishment",
                "status": "custom",
                "reason": "Establish and persist a player colony after the planet is secured.",
            },
        ],
        "core_loop": ["Visit a planet and resolve its encounter."],
        "progression": [],
        "combat": {},
        "mod_context": {},
    }


def test_design_modules_are_bounded_template_fill_evidence() -> None:
    context = deep._execution_context(
        _design(),
        {
            "capabilities": [
                {
                    "capability": "design.module.colony_establishment",
                    "mode": "source_transplant",
                    "source_id": "github:verified-colony-source",
                    "proof_level": "PINNED",
                }
            ]
        },
    )

    assert [item["design_leaf_capability"] for item in context] == [
        "design.module.alien_encounter",
        "design.module.colony_establishment",
    ]
    assert all(item["requirement_ref"] == "req_parent" for item in context)
    assert all(
        item["parent_capability"] == "alien_planet_interaction" for item in context
    )
    assert all(item["authority"] == "template_fill_evidence_only" for item in context)
    assert context[0]["reuse_mode"] == "fresh"
    assert context[0]["reuse_refs"] == []
    assert context[1]["reuse_mode"] == "source_transplant"
    assert context[1]["reuse_refs"] == ["github:verified-colony-source"]
    assert not any("task_id" in item or "depends_on" in item for item in context)


def test_explicit_narrative_retrieval_facet_is_preserved_without_becoming_authority() -> None:
    design = _design()
    design["_pre_retrieval_plan"]["design_retrieval_facets"] = [
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

    assert [item["design_leaf_capability"] for item in context] == [
        "design.module.alien_encounter",
        "design.core_loop.visit_planet",
    ]
    assert all(item["parent_capability"] == "alien_planet_interaction" for item in context)
    assert context[0]["reuse_refs"] == ["github:alien-encounter-source"]
    assert context[1]["reuse_mode"] == "fresh"
    assert all(item["authority"] == "template_fill_evidence_only" for item in context)
    assert not any("task_id" in item or "depends_on" in item for item in context)
