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


def test_sharded_request_completes_all_research_before_any_design(monkeypatch) -> None:
    from minecraft_mod_ai import agentic_research_game_design as agentic
    from minecraft_mod_ai import game_design
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline

    events: list[str] = []
    pages = ("page zero", "page one", "page two")

    monkeypatch.setattr(agentic, "supports_agentic_research_router", lambda router: True)

    def collect(router, prompt, *, trace_metadata=None):
        del router
        page_index = int(dict(trace_metadata or {})["request_page_index"])
        events.append(f"research:{page_index}:{prompt}")
        return {
            "research_sha256": f"research-{page_index}",
            "model_view_sha256": f"view-{page_index}",
            "research_brief": {"summary": prompt},
        }

    monkeypatch.setattr(pipeline, "collect_design_research", collect)

    def generate(
        router,
        *,
        authoritative_prompt,
        media_paths,
        system_prompt,
        fallback_prompt=None,
        precollected_research=None,
    ):
        del router, authoritative_prompt, media_paths, system_prompt
        page = str(fallback_prompt or "")
        page_index = pages.index(page)
        assert precollected_research["research_sha256"] == f"research-{page_index}"
        events.append(f"design:{page_index}:{page}")
        return {
            "title": "merged",
            "pitch": "bounded sharded design",
            "core_loop": [f"loop {page_index}"],
            "progression": [f"progress {page_index}"],
            "combat": {},
            "mod_context": {},
            "modules": [],
            "assets": [],
            "acceptance_tests": [f"verify {page_index}"],
        }

    monkeypatch.setattr(game_design, "_generate_game_design_once", generate)
    planner = game_design.GameDesignPlanner(object())
    result = planner._plan_sharded_request(
        "".join(pages),
        request_pages=pages,
        media_paths=(),
        page_budget=4096,
    )

    assert events[:3] == [
        "research:0:page zero",
        "research:1:page one",
        "research:2:page two",
    ]
    assert events[3:] == [
        "design:0:page zero",
        "design:1:page one",
        "design:2:page two",
    ]
    ledger = result["_pre_design_research"]
    assert ledger["page_count"] == 3
    assert [page["research_sha256"] for page in ledger["pages"]] == [
        "research-0",
        "research-1",
        "research-2",
    ]


def test_runtime_uses_native_research_first_design_owner() -> None:
    from minecraft_mod_ai import game_design

    assert not hasattr(game_design._generate_game_design_once, "__wrapped__")
    assert not hasattr(game_design.GameDesignPlanner._plan_sharded_request, "__wrapped__")
    assert getattr(game_design.GameDesignPlanner.plan, "_mmm_host_owned_template", False)
