from __future__ import annotations

import threading
from types import SimpleNamespace

from minecraft_mod_ai import agentic_pre_design_rag as project_rag
from minecraft_mod_ai import agentic_research_game_design as agentic
from minecraft_mod_ai import pre_design_research_pipeline as pipeline
from minecraft_mod_ai.pre_design_research_pipeline import (
    _bounded_model_view,
    _design_request_fits,
    collect_design_research,
)


def test_pre_design_research_parallelizes_all_evidence_and_defers_donor_search(
    monkeypatch,
) -> None:
    brief = {
        "domains": [
            {
                "domain_id": "fabric_api",
                "queries": ["Fabric API target behavior"],
            }
        ]
    }
    monkeypatch.setattr(pipeline, "normalize_research_brief", lambda *_args: brief)

    barrier = threading.Barrier(3)

    def official(_brief):
        barrier.wait(timeout=2)
        return {"status": "available", "kind": "official"}

    def radar(_prompt, _brief, **_kwargs):
        barrier.wait(timeout=2)
        return {"status": "available", "kind": "radar"}

    def forced(_router, _brief):
        barrier.wait(timeout=2)
        return {
            "schema_version": "mmm/forced-pre-design-rag-v2",
            "domain_count": 1,
            "query_count": 1,
            "domains": [
                {
                    "domain_id": "fabric_api",
                    "queries": [{"query": "Fabric API target behavior"}],
                }
            ],
        }

    monkeypatch.setattr(pipeline, "retrieve_domain_evidence", official)
    monkeypatch.setattr(pipeline, "collect_technology_radar", radar)
    monkeypatch.setattr(project_rag, "_forced_rag_bundle", forced)

    seen_deterministic = []

    def domain_agent(_router, **kwargs):
        deterministic = kwargs["deterministic"]
        seen_deterministic.append(set(deterministic))
        return {
            "domain_id": kwargs["domain"]["domain_id"],
            "claims": [],
            "gaps": [],
            "next_queries": [],
            "sufficient": True,
        }

    monkeypatch.setattr(agentic, "_research_domain_with_agent", domain_agent)

    payload = collect_design_research(object(), "build a Fabric mechanic")

    expected = {"official_rag", "technology_radar", "forced_project_rag"}
    assert set(payload["deterministic"]) == expected
    assert "ecosystem_discovery" not in payload["deterministic"]
    assert seen_deterministic == [expected]
    assert payload["domain_notes"][0]["domain_id"] == "fabric_api"
    assert "deferred" in payload["method"]["planning_search"]


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
        domains.append(
            {
                "domain_id": domain_id,
                "objective": f"Design evidence for {domain_id}",
                "providers": ["official_docs", "project_rag"],
            }
        )
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
        "research_brief": {
            "summary": "oversized regression fixture",
            "domains": domains,
            "unresolved_questions": [],
        },
        "deterministic": {
            "official_rag": {"status": "available", "evidence_sha256": "official"},
            "technology_radar": {"status": "available", "radar_sha256": "radar"},
            "forced_project_rag": {
                "status": "available",
                "research_sha256": "project",
            },
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
    assert (
        view["research_brief"]["model_context_view"]["budget_authority"]
        == "model_context_budget.request_message_budget"
    )
    assert view["research_brief"]["domain_count"] == len(payload["domain_notes"])
    assert "host_research_ledger" not in agentic._compact_research_for_design(view)


def test_small_research_payload_is_not_compacted() -> None:
    router = _BudgetedRouter()
    payload = {
        "schema_version": "mmm/agentic-pre-design-research-v1",
        "research_brief": {
            "summary": "small",
            "domains": [{"domain_id": "request"}],
            "unresolved_questions": [],
        },
        "deterministic": {},
        "domain_notes": [
            {
                "domain_id": "request",
                "claims": [
                    {
                        "claim": "Fabric registration is available",
                        "evidence_refs": [],
                    }
                ],
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
