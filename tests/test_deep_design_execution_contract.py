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
