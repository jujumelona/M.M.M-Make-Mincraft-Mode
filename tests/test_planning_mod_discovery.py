from copy import deepcopy

import pytest

from minecraft_mod_ai.catalog_first_grounded_rag import _domain_specs
from minecraft_mod_ai.planning_mod_discovery import catalog_queries, discovery_receipt
from minecraft_mod_ai.planning_state_implementation import _requirement_grounding
import json
from minecraft_mod_ai.planning_detail_template import WORKSHEET_SECTIONS
from minecraft_mod_ai.research_reuse_candidates import project_repository_candidates


def _state():
    return {
        "decisions": [{"decision_type": "requirement", "requirement_id": "req_001",
                       "semantic_capability": "space.travel", "statement": "Travel to space"}],
        "research_queue": [{"research_id": "r_001", "requirement_ref": "req_001",
                            "objective": "Find useful implementation/reuse options for space travel",
                            "information_needed": "Concrete implementation patterns and support artifacts",
                            "source_kinds": ["existing_mods", "minecraft_docs"],
                            "queries": [], "resolves": ["u_001"], "status": "pending"}],
        "unresolved": [{"unresolved_id": "u_001", "status": "open"}],
        "evidence": [], "resolved": [], "blockers": [], "state_sha256": "",
    }


def _grounded(*, status="available", hits=True, source_url=""):
    return {"queries": [{"query": "space travel", "query_sha256": "sha256:query",
        "provider_receipts": {"modrinth": {"status": status, "search_requests": 1,
                                            "result_count": int(hits)}},
        "evidence_records": [{"source_id": "modrinth:space", "title": "Space Example",
            "url": "https://modrinth.com/mod/space-example", "content": "Travel to space.",
            "metadata": {"versions": ["1.20.1"], "loaders": ["forge"],
                         "source_url": source_url}}] if hits else []}]}


def test_catalog_queries_use_approved_capability_and_do_not_send_api_instructions():
    state = _state()
    queries = catalog_queries(state, state["research_queue"][0])
    assert queries == ["space travel", "space", "travel"]
    specs = _domain_specs({"providers": ["modrinth", "official_docs"],
                           "catalog_queries": queries, "queries": ["Minecraft API teleport"]})
    assert ("space", ("modrinth",)) in specs
    assert ("Minecraft API teleport", ("official_docs",)) in specs
    assert not any("API" in query and "modrinth" in providers for query, providers in specs)


def test_catalog_mod_without_repository_survives_and_linked_repo_does_not_need_fetch():
    grounded = _grounded()
    receipt = discovery_receipt("r_001", grounded)
    assert receipt["complete"] and receipt["candidates"][0]["source_url"] == ""
    assert receipt["candidates"][0]["license"] is None
    assert receipt["candidates"][0]["compatibility"] == "not_verified"
    grounded = _grounded(source_url="https://github.com/example/space")
    candidates = project_repository_candidates(
        {"domain_id": "r_001", "evidence_kinds": ["source_code"]}, grounded)
    assert candidates[0]["repository"] == "example/space"
    assert candidates[0]["source_reuse_authority"] == "verification_required"


@pytest.mark.parametrize("status,expected", [
    ("available", "no_results"), ("error", "catalog_unavailable"),
    ("not_configured", "catalog_unavailable"),
])
def test_empty_search_is_distinguished_from_transport_failure(status, expected):
    receipt = discovery_receipt("r_001", _grounded(status=status, hits=False))
    assert receipt["status"] == expected
    assert receipt["complete"] == (status == "available")


def test_catalog_hits_lost_during_projection_are_not_success():
    grounded = _grounded(hits=False)
    grounded["queries"][0]["provider_receipts"]["modrinth"]["result_count"] = 2
    receipt = discovery_receipt("r_001", grounded)
    assert receipt["status"] == "candidate_projection_failed"
    assert not receipt["complete"]


@pytest.mark.parametrize("available", [True, False])
def test_research_to_criterion_preserves_catalog_or_blocks_unavailable_catalog(monkeypatch, available):
    from minecraft_mod_ai import planning_state_research as research
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    from minecraft_mod_ai import pre_design_grounded_rag as backend

    monkeypatch.setattr(research, "validate_planning_state", lambda *a, **k: None)
    monkeypatch.setattr(research, "emit_root_cause", lambda *a, **k: None)
    monkeypatch.setattr(research, "forced_rag_bundle", lambda *a: {})
    monkeypatch.setattr(pipeline, "_grounded_domain_evidence", lambda *a: _grounded(
        hits=available, status="available" if available else "error"))
    monkeypatch.setattr(backend, "_materialize_domain_evidence_document", lambda *a: {})
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *a, **k: None)
    # API evidence alone used to pass even when the catalog search had failed.
    monkeypatch.setattr(research, "research_document_domain", lambda *a, **k: {
        "domain_id": "r_001", "sufficient": True,
        "claims": [{"claim": "Server owns dimension transfer", "evidence_refs": ["api:transfer"]}],
    })
    result = research.collect_planning_state_research(object(), "space", deepcopy(_state()))
    assert (result["research_queue"][0]["status"] == "complete") == available
    if not available:
        assert "catalog_unavailable" in result["blockers"][0]["statement"]
        return
    evidence, refs = _requirement_grounding(result, "req_001")
    assert "modrinth:space" in refs
    messages = [{}, {"content": json.dumps(evidence)}]
    assert "Space Example" in messages[1]["content"]
    assert "https://modrinth.com/mod/space-example" in messages[1]["content"]
    assert "not_verified" in messages[1]["content"]
