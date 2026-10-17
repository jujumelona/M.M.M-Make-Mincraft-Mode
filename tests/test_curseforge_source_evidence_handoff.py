"""Regression coverage for CurseForge catalogue -> source evidence handoff."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import grounded_source_reuse
from minecraft_mod_ai.source_reuse_evidence import build_source_reuse_evidence


def _design():
    return {
        "_pre_retrieval_plan": {
            "plan_sha256": "test-plan",
            "capability_graph": {
                "nodes": ["space"],
                "requirement_bindings": [
                    {"requirement_id": "req-space", "capability": "space"}
                ],
            },
        },
        "_platform_selection": {
            "target": {"minecraft_version": "26.2", "loader": "fabric"}
        },
    }


@pytest.mark.parametrize(
    ("discovery_status", "catalog_count", "evidence_status", "failure_code"),
    [
        ("catalog_only", 2, "unresolved", ""),
        ("search_empty", 0, "search_empty", ""),
        ("provider_error", 0, "provider_error", "CURSEFORGE_PROVIDER_ERROR"),
        ("not_configured", 0, "provider_error", "CURSEFORGE_NOT_CONFIGURED"),
    ],
)
def test_catalogue_status_can_compile_as_fresh_evidence(
    monkeypatch, discovery_status, catalog_count, evidence_status, failure_code
):
    monkeypatch.setattr(
        grounded_source_reuse,
        "adapter_for_target",
        lambda *_args: SimpleNamespace(validate=lambda: None),
    )
    monkeypatch.setattr(grounded_source_reuse, "curseforge_api_key", lambda: "fake")
    def fake_discovery(_capabilities, _client, *, diagnostics, **_kwargs):
        diagnostics.append({
            "provider": "curseforge",
            "status": discovery_status,
            "catalog_result_count": catalog_count,
        })
        return {"space": ()}
    monkeypatch.setattr(
        grounded_source_reuse, "discover_repositories_for_graph", fake_discovery
    )
    plan = grounded_source_reuse.build_repository_reuse_plan(_design())
    assert len(plan["capabilities"]) == 1
    decision = plan["capabilities"][0]
    assert decision["mode"] == "fresh"
    assert decision["search_status"] == evidence_status
    assert decision["discovery_status"] == discovery_status
    assert decision["failure_code"] == failure_code
    assert decision["source_id"] == ""

    evidence = build_source_reuse_evidence(
        [{"requirement_id": "req-space"}], plan, {}
    )
    assert len(evidence) == 1
    assert evidence[0].mode == "fresh"
    assert evidence[0].status == evidence_status
    assert evidence[0].failure_code == failure_code
    assert evidence[0].repository == ""
    assert evidence[0].files == ()


def test_actual_curseforge_source_link_survives_into_production(monkeypatch):
    from minecraft_mod_ai.research_reuse_candidates import (
        project_repository_candidates, ensure_authored_source_handoff,
    )
    from minecraft_mod_ai.pre_design_research_pipeline import _canonical_sha256

    grounded = {"queries": [{"query": "spaceship", "evidence_records": [{
        "source_id": "curseforge:42",
        "metadata": {"provider": "curseforge", "source_id": "curseforge:42",
                     "source_url": "https://github.com/example/space-donor",
                     "catalog_url": "https://www.curseforge.com/minecraft/mc-mods/space"},
        "summary": "Spaceship with movable parts",
    }]}]}
    links = project_repository_candidates({"domain_id": "request"}, grounded)
    assert len(links) == 1
    assert links[0]["repository"] == "example/space-donor"
    assert links[0]["reference_only"] is True
    assert links[0]["source_reuse_authority"] == "verification_required"
    prompt = "Build a spaceship"
    record = {"_pre_design_source_handoff": {"prompt_sha256": _canonical_sha256(prompt),
                                              "repository_candidates": links}}
    recovered, candidates = ensure_authored_source_handoff(
        record, router=object(), prompt=prompt,
    )
    assert candidates == links
    assert recovered["_pre_design_source_handoff"]["candidate_count"] == 1


def test_curseforge_catalogue_without_source_link_is_not_a_code_candidate():
    from minecraft_mod_ai.research_reuse_candidates import project_repository_candidates

    grounded = {"queries": [{"evidence_records": [{
        "source_id": "curseforge:42", "metadata": {"provider": "curseforge",
            "source_url": "", "source_id": "curseforge:42"}
    }]}]}
    assert project_repository_candidates({}, grounded) == []


@pytest.mark.parametrize(
    ("legacy_status", "expected_status", "expected_failure"),
    [
        ("catalog_only", "unresolved", ""),
        ("provider_error", "provider_error", "CATALOG_PROVIDER_ERROR_UNSPECIFIED"),
    ],
)
def test_existing_saved_source_plans_do_not_crash(
    legacy_status, expected_status, expected_failure
):
    plan = {
        "capability_graph": {
            "requirement_bindings": [
                {"requirement_id": "req-space", "capability": "space"}
            ]
        },
        "capabilities": [
            {"capability": "space", "mode": "fresh",
             "search_status": legacy_status, "source_id": "",
             "failure_code": ""}
        ],
    }
    evidence = build_source_reuse_evidence(
        [{"requirement_id": "req-space"}], plan, {}
    )
    assert len(evidence) == 1
    assert evidence[0].status == expected_status
    assert evidence[0].failure_code == expected_failure
    assert evidence[0].requirement_ids == ("req-space",)
