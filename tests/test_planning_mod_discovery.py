import json
from copy import deepcopy

import pytest

from minecraft_mod_ai.catalog_first_grounded_rag import _domain_specs
from minecraft_mod_ai.planning_mod_discovery import catalog_queries, discovery_receipt
from minecraft_mod_ai.planning_state_implementation import _requirement_grounding
from minecraft_mod_ai.research_reuse_candidates import (
    merge_repository_candidates,
    project_repository_candidates,
)


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


def test_catalog_queries_preserve_task_context_and_do_not_send_api_instructions():
    state = _state()
    research = state["research_queue"][0]
    prompt = "Farm resources, trade for money, build and upgrade a spaceship, then explore planets"
    queries = catalog_queries(state, research, prompt=prompt)
    assert queries == [
        "space travel",
        "space",
        "travel",
        "Travel to space",
        prompt,
    ]
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
def test_empty_search_never_completes_candidate_discovery(status, expected):
    receipt = discovery_receipt("r_001", _grounded(status=status, hits=False))
    assert receipt["status"] == expected
    assert not receipt["complete"]
    assert receipt["corrective_retrieval_required"] == (status == "available")


def test_catalog_hits_lost_during_projection_are_not_success():
    grounded = _grounded(hits=False)
    grounded["queries"][0]["provider_receipts"]["modrinth"]["result_count"] = 2
    receipt = discovery_receipt("r_001", grounded)
    assert receipt["status"] == "candidate_projection_failed"
    assert not receipt["complete"]


def test_repository_candidate_merge_has_no_hidden_global_top_n_cut():
    candidates = []
    for index in range(12):
        candidates.append({
            "repository": f"example/repo-{index}",
            "candidate_score": 1.0 / (index + 1),
            "domain_ids": ["r_001"],
            "source_ids": [f"github:example/repo-{index}"],
            "source_urls": [f"https://github.com/example/repo-{index}"],
            "query_sha256": [f"sha256:{index}"],
            "evidence_text": "spacecraft gameplay",
            "evidence_tokens": ["spacecraft", "gameplay"],
            "origin": "host_grounded_retrieval",
            "reference_only": True,
            "source_reuse_authority": "verification_required",
        })
    merged = merge_repository_candidates([], candidates)
    assert len(merged) == 12


@pytest.mark.parametrize("available", [True, False])
def test_research_to_criterion_preserves_catalog_or_blocks_unavailable_catalog(monkeypatch, available):
    from minecraft_mod_ai import planning_state_research as research
    from minecraft_mod_ai import pre_design_grounded_rag as backend
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline

    monkeypatch.setattr(research, "validate_planning_state", lambda *a, **k: None)
    monkeypatch.setattr(research, "emit_root_cause", lambda *a, **k: None)
    monkeypatch.setattr(research, "forced_rag_bundle", lambda *a: {})
    monkeypatch.setattr(pipeline, "_grounded_domain_evidence", lambda *a: _grounded(
        hits=available, status="available" if available else "error"))
    monkeypatch.setattr(backend, "_materialize_domain_evidence_document", lambda *a: {})
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *a, **k: None)
    # API evidence alone must not pass when candidate discovery did not produce a candidate.
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
