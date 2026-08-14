from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.research_adaptive_provider_routing as provider


def _brief(*, kinds, providers):
    return {
        "brief_sha256": "sha256:brief",
        "domains": [
            {
                "domain_id": "domain",
                "evidence_kinds": list(kinds),
                "providers": list(providers),
                "queries": ["query"],
            }
        ],
    }


def test_official_docs_owns_catalog_without_duplicate_project_rag():
    adaptive_rag = SimpleNamespace(
        _route=lambda _domain, code_index_available: {
            "catalog": True,
            "code": False,
            "providers": frozenset({"official_docs", "project_rag"}),
            "evidence_kinds": frozenset({"minecraft_api"}),
            "reason": "evidence_kind_route",
        }
    )
    agentic = SimpleNamespace(
        collect_technology_radar=lambda *_args, **_kwargs: {"called": True},
        collect_ecosystem_seed_bundle=lambda *_args, **_kwargs: {"called": True},
    )

    provider.harden(agentic, adaptive_rag)
    routed = adaptive_rag._route(
        _brief(kinds=["minecraft_api"], providers=["official_docs", "project_rag"])[
            "domains"
        ][0],
        code_index_available=True,
    )

    assert routed["catalog"] is False
    assert routed["code"] is False
    assert routed["evidence_kinds"] == frozenset()
    assert routed["reason"] == "official_docs_owns_catalog"


def test_official_docs_keeps_explicit_code_lane_but_not_catalog_lane():
    adaptive_rag = SimpleNamespace(
        _route=lambda _domain, code_index_available: {
            "catalog": True,
            "code": bool(code_index_available),
            "providers": frozenset({"official_docs", "project_rag"}),
            "evidence_kinds": frozenset({"minecraft_api", "source_code"}),
            "reason": "evidence_kind_route",
        }
    )
    agentic = SimpleNamespace(
        collect_technology_radar=lambda *_args, **_kwargs: {"called": True},
        collect_ecosystem_seed_bundle=lambda *_args, **_kwargs: {"called": True},
    )

    provider.harden(agentic, adaptive_rag)
    routed = adaptive_rag._route(
        _brief(
            kinds=["minecraft_api", "source_code"],
            providers=["official_docs", "project_rag"],
        )["domains"][0],
        code_index_available=True,
    )

    assert routed["catalog"] is False
    assert routed["code"] is True
    assert routed["evidence_kinds"] == frozenset({"source_code"})


def test_technology_radar_runs_only_for_technology_evidence():
    tech_calls = []

    def technology(prompt, research_brief=None, **kwargs):
        tech_calls.append((prompt, research_brief, kwargs))
        return {"status": "called"}

    agentic = SimpleNamespace(
        collect_technology_radar=technology,
        collect_ecosystem_seed_bundle=lambda *_args, **_kwargs: {"called": True},
    )
    adaptive_rag = SimpleNamespace(
        _route=lambda _domain, code_index_available: {
            "catalog": False,
            "code": False,
            "providers": frozenset(),
            "evidence_kinds": frozenset(),
            "reason": "none",
        }
    )
    provider.harden(agentic, adaptive_rag)

    skipped = agentic.collect_technology_radar(
        "ordinary mod",
        _brief(kinds=["minecraft_api"], providers=["official_docs"]),
        page_size=50,
    )
    called = agentic.collect_technology_radar(
        "AI mod",
        _brief(kinds=["ai_inference"], providers=["huggingface_models"]),
        page_size=50,
    )

    assert skipped["status"] == "not_required"
    assert skipped["requirements"] == []
    assert called["status"] == "called"
    assert len(tech_calls) == 1


def test_planning_ecosystem_is_deferred_but_specialist_call_is_preserved():
    ecosystem_calls = []

    def ecosystem(prompt, game_design, **kwargs):
        ecosystem_calls.append((prompt, game_design, kwargs))
        return {"status": "called"}

    agentic = SimpleNamespace(
        collect_technology_radar=lambda *_args, **_kwargs: {"called": True},
        collect_ecosystem_seed_bundle=ecosystem,
    )
    adaptive_rag = SimpleNamespace(
        _route=lambda _domain, code_index_available: {
            "catalog": False,
            "code": False,
            "providers": frozenset(),
            "evidence_kinds": frozenset(),
            "reason": "none",
        }
    )
    provider.harden(agentic, adaptive_rag)
    brief = _brief(kinds=["dependency"], providers=["github", "modrinth"])

    deferred = agentic.collect_ecosystem_seed_bundle(
        "mod request",
        {},
        research_brief=brief,
        planning_seed_only=True,
        route_limit=12,
    )
    specialist = agentic.collect_ecosystem_seed_bundle(
        "mod request",
        {},
        research_brief=brief,
        planning_seed_only=False,
        route_limit=12,
    )

    assert deferred["status"] == "deferred"
    assert deferred["route_count"] == 2
    assert deferred["candidate_count"] == 0
    assert specialist["status"] == "called"
    assert len(ecosystem_calls) == 1
