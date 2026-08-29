from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agentic_pre_design_rag as project_rag
from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai import pre_design_research_pipeline as pipeline
from minecraft_mod_ai.pre_design_research_pipeline import (
    PreDesignResearchFailure,
    _bounded_model_view,
    _design_request_fits,
    collect_design_research,
)


def test_pre_design_parallelizes_target_neutral_evidence_and_defers_target_radar(
    monkeypatch,
) -> None:
    brief = {
        "domains": [
            {
                "domain_id": "fabric_api",
                "queries": ["Fabric API target-neutral behavior"],
            }
        ]
    }
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(
        pipeline,
        "compile_minecraft_knowledge_plan",
        lambda _prompt: {
            "plan_sha256": "sha256:plan",
            "policy": {"target_frozen": False},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_route_coverage",
        lambda *_args, **_kwargs: {"status": "PASS", "blocking_requirement_refs": []},
    )

    barrier = threading.Barrier(2)

    def official(_brief):
        barrier.wait(timeout=2)
        return {"status": "available", "kind": "official"}

    def forced(_router, _brief):
        barrier.wait(timeout=2)
        return {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "domain_count": 1,
            "query_count": 1,
            "domains": [
                {
                    "domain_id": "fabric_api",
                    "queries": [{"query": "Fabric API target-neutral behavior"}],
                }
            ],
        }

    def radar_must_not_run(*_args, **_kwargs):
        raise AssertionError("target-specific technology radar ran before target freeze")

    monkeypatch.setattr(pipeline, "retrieve_domain_evidence", official)
    monkeypatch.setattr(pipeline, "collect_technology_radar", radar_must_not_run)
    monkeypatch.setattr(project_rag, "_forced_rag_bundle", forced)

    seen_deterministic = []

    def domain_agent(_router, **kwargs):
        deterministic = kwargs["deterministic"]
        seen_deterministic.append(set(deterministic))
        return {
            "domain_id": kwargs["domain"]["domain_id"],
            "claims": [
                {
                    "claim": "Target-neutral Fabric research can proceed before target freeze.",
                    "evidence_refs": ["sha256:plan"],
                }
            ],
            "gaps": [],
            "next_queries": [],
            "procedures": [],
            "sufficient": True,
        }

    monkeypatch.setattr(agentic, "_research_domain_with_agent", domain_agent)
    monkeypatch.setattr(pipeline, "attach_procedural_skillbank", lambda _r, _p, value: value)
    monkeypatch.setattr(pipeline, "compose_research_skillbank", lambda _r, _p, value: value)

    payload = collect_design_research(object(), "build a Fabric mechanic")

    expected = {"official_rag", "technology_radar", "forced_project_rag"}
    assert set(payload["deterministic"]) == expected
    assert payload["deterministic"]["technology_radar"]["status"] == "deferred_until_target_freeze"
    assert "ecosystem_discovery" not in payload["deterministic"]
    assert seen_deterministic == [expected]
    assert payload["domain_notes"][0]["domain_id"] == "fabric_api"
    assert "deferred" in payload["method"]["planning_search"]


def test_terminal_gap_prints_full_failure_and_stops_before_post_research_work(
    monkeypatch, capsys
) -> None:
    brief = {"domains": [{"domain_id": "request", "queries": ["request evidence"]}]}
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(pipeline, "retrieve_domain_evidence", lambda _brief: {"status": "available", "domains": []})
    monkeypatch.setattr(project_rag, "_forced_rag_bundle", lambda *_args, **_kwargs: {"status": "available", "domains": []})
    monkeypatch.setattr(
        agentic,
        "_research_domain_with_agent",
        lambda *_args, **_kwargs: {
            "domain_id": "request",
            "claims": [],
            "gaps": ["EXACT_SYNTHESIS_GAP"],
            "next_queries": [],
            "sufficient": False,
            "checkpoint": {"status": "terminal_gap", "request_sha256": "sha256:failure"},
            "research_failures": [{"unit": "synthesis:0:0", "error": "EXACT_VALIDATOR_FAILURE: missing claim evidence"}],
            "fixed_point": True,
        },
    )

    def must_not_continue(*_args, **_kwargs):
        raise AssertionError("post-research processing must not run after terminal_gap")

    monkeypatch.setattr(pipeline, "attach_procedural_skillbank", must_not_continue)
    monkeypatch.setattr(pipeline, "compose_research_skillbank", must_not_continue)

    with pytest.raises(PreDesignResearchFailure, match="terminal_gap"):
        collect_design_research(object(), "failing request")

    logged = capsys.readouterr().out
    assert "PRE-DESIGN RESEARCH DIAGNOSTIC:" in logged
    assert '"event": "domain_result"' in logged
    assert "terminal_gap" in logged
    assert "synthesis:0:0" in logged
    assert "EXACT_VALIDATOR_FAILURE: missing claim evidence" in logged
    assert "EXACT_SYNTHESIS_GAP" in logged


def test_domain_exception_prints_full_traceback_and_escapes(monkeypatch, capsys) -> None:
    brief = {"domains": [{"domain_id": "request", "queries": ["request evidence"]}]}
    monkeypatch.setattr(pipeline, "_pre_design_brief", lambda _prompt: brief)
    monkeypatch.setattr(pipeline, "retrieve_domain_evidence", lambda _brief: {})
    monkeypatch.setattr(project_rag, "_forced_rag_bundle", lambda *_args, **_kwargs: {})

    def explode(*_args, **_kwargs):
        raise ValueError("EXACT_DOMAIN_EXCEPTION")

    monkeypatch.setattr(agentic, "_research_domain_with_agent", explode)

    with pytest.raises(ValueError, match="EXACT_DOMAIN_EXCEPTION"):
        collect_design_research(object(), "failing request")

    logged = capsys.readouterr().out
    assert '"event": "domain_execution_exception"' in logged
    assert "EXACT_DOMAIN_EXCEPTION" in logged
    assert "ValueError" in logged
    assert "Traceback (most recent call last)" in logged


class _Registry:
    def role(self, profile: str, role: str):
        assert profile == "test"
        assert role == "planner"
        return SimpleNamespace(
            adapter="openai_compatible",
            max_context=8192,
            max_input_tokens=0,
            max_new_tokens=1024,
        )


class _BudgetedRouter:
    profile = "test"
    registry = _Registry()


def _oversized_research_payload() -> dict:
    domains = []
    notes = []
    for index in range(24):
        domain_id = f"domain_{index}"
        domains.append({"domain_id": domain_id, "objective": f"Design evidence for {domain_id}", "providers": ["official_docs", "project_rag"]})
        notes.append(
            {
                "domain_id": domain_id,
                "claims": [
                    {
                        "claim": f"{domain_id}:{claim_index}:" + ("evidence " * 700),
                        "evidence_refs": [f"receipt:{domain_id}:{claim_index}"],
                    }
                    for claim_index in range(3)
                ],
                "gaps": ["gap " * 500],
                "next_queries": ["query " * 500],
                "sufficient": True,
            }
        )
    payload = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": {"summary": "oversized regression fixture", "domains": domains, "unresolved_questions": []},
        "deterministic": {
            "official_rag": {"status": "available", "evidence_sha256": "official"},
            "technology_radar": {"status": "available", "radar_sha256": "radar"},
            "forced_project_rag": {"status": "available", "research_sha256": "project"},
        },
        "domain_notes": notes,
        "errors": [],
        "method": {},
    }
    payload["research_sha256"] = agentic._json_sha256(payload)
    return payload


def test_oversized_research_keeps_full_host_ledger_and_fits_every_design_section() -> None:
    router = _BudgetedRouter()
    prompt = "설계 전에 충분히 조사하고 상세한 우주선 모드를 설계한다."
    payload = _oversized_research_payload()

    assert not _design_request_fits(agentic, router, prompt, payload)
    view = _bounded_model_view(agentic, router, prompt, payload)

    assert _design_request_fits(agentic, router, prompt, view)
    assert view["research_sha256"] == payload["research_sha256"]
    assert view["host_research_ledger"]["research_brief"] == payload["research_brief"]
    assert view["host_research_ledger"]["domain_notes"] == payload["domain_notes"]
    assert view["research_brief"]["model_context_view"]["budget_authority"] == "model_context_budget.request_message_budget"
    assert view["research_brief"]["domain_count"] == len(payload["domain_notes"])
    assert "host_research_ledger" not in agentic._compact_research_for_design(view)


def test_small_research_payload_is_not_compacted() -> None:
    router = _BudgetedRouter()
    payload = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": {"summary": "small", "domains": [{"domain_id": "request"}], "unresolved_questions": []},
        "deterministic": {},
        "domain_notes": [
            {
                "domain_id": "request",
                "claims": [{"claim": "Fabric registration is available", "evidence_refs": []}],
                "gaps": [],
                "next_queries": [],
                "sufficient": True,
            }
        ],
        "errors": [],
        "method": {},
    }
    payload["research_sha256"] = agentic._json_sha256(payload)

    assert _bounded_model_view(agentic, router, "small request", payload) is payload
