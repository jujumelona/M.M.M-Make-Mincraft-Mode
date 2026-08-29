from __future__ import annotations

from minecraft_mod_ai import semantic_requirement_authority as semantic
from minecraft_mod_ai.planner_graph_integrity_contract import (
    _design_facets,
    _facet_work_index,
    _production_depth_game_design_prompt,
)


class _SemanticRouter:
    def __init__(self) -> None:
        self.parameters = None
        self.messages = None

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description,
    ):
        del role, tool_name, description
        self.messages = messages
        self.parameters = parameters
        return {"requirements": []}


def test_semantic_leaf_planning_has_no_per_clause_item_cap() -> None:
    router = _SemanticRouter()
    clauses = [
        {
            "clause_index": 0,
            "char_start": 0,
            "char_end": 40,
            "text": "farm resources earn money trade build upgrade travel",
            "text_sha256": "sha256:" + "0" * 64,
        }
    ]

    semantic._call_semantic_model(router, clauses)

    requirements = router.parameters["properties"]["requirements"]
    assert "maxItems" not in requirements
    assert "many independent behaviors" in router.messages[0]["content"]
    assert "Never compress multiple verbs" in router.messages[0]["content"]
    assert "no fixed leaf target or per-clause leaf ceiling" in router.messages[0]["content"]


def test_game_design_prompt_requires_leaf_subsystems_before_reuse_search() -> None:
    prompt = _production_depth_game_design_prompt()

    assert "Complete the gameplay/mod design before choosing any third-party" in prompt
    assert "smallest meaningful subsystems" in prompt
    assert "there is no arbitrary module count" in prompt
    assert "Skills/MCP research" in prompt


def test_design_facets_expand_modules_and_design_sections() -> None:
    design = {
        "modules": [
            {
                "plugin_id": "currency_wallet",
                "status": "custom",
                "reason": "Earn and spend persistent currency from resource gathering.",
            },
            {
                "plugin_id": "ship_part_assembly",
                "status": "custom",
                "reason": "Purchase and assemble individual ship parts into a functional ship.",
            },
        ],
        "core_loop": [
            "Gather resources and convert them into spendable currency.",
            "Buy ship components and assemble the vessel.",
        ],
        "progression": [
            "Upgrade weapons, crew capacity, and propulsion before launch.",
        ],
        "combat": {
            "alien_encounters": [
                "Fight alien enemies on discovered planets.",
                "Receive combat rewards that feed progression.",
            ]
        },
        "mod_context": {
            "planet_content": [
                "Mine planet-specific special minerals.",
            ]
        },
    }

    facets = _design_facets(design)
    capabilities = {item["capability"] for item in facets}

    assert "design.module.currency_wallet" in capabilities
    assert "design.module.ship_part_assembly" in capabilities
    assert len(facets) >= 8


def test_facet_binding_prefers_related_authored_work() -> None:
    work = [
        {
            "work_id": "economy",
            "objective": "resource farming currency accumulation and trading",
            "capabilities": ["resource_farming_currency_accumulation"],
            "acceptance": ["gather resources then currency increases"],
        },
        {
            "work_id": "ship",
            "objective": "purchase ship parts and assemble a functional interstellar ship",
            "capabilities": ["interstellar_ship_construction"],
            "acceptance": ["purchased components can be assembled"],
        },
    ]
    facet = {
        "capability": "design.module.ship_part_assembly",
        "label": "ship part assembly",
        "detail": "purchase and assemble individual ship parts into a functional ship",
        "source": "game_design.modules[0]",
    }

    index = _facet_work_index(facet, work, 0, 1)

    assert index == 1


def test_facet_binding_uses_narrative_order_when_language_has_no_token_overlap() -> None:
    work = [
        {"work_id": "first", "objective": "첫 번째 요구", "capabilities": ["first"]},
        {"work_id": "second", "objective": "두 번째 요구", "capabilities": ["second"]},
    ]
    facet = {
        "capability": "design.module.flight",
        "label": "interstellar flight",
        "detail": "launch the completed ship into space",
        "source": "game_design.modules[0]",
    }

    assert _facet_work_index(facet, work, 1, 2) == 1
