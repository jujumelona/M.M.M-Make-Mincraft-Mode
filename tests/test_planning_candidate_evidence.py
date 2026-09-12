from copy import deepcopy
from urllib.parse import parse_qs, urlparse

import pytest

from minecraft_mod_ai.planning_candidate_evidence import (
    global_grounded_pool,
    requirement_candidate_trace,
)


def requirement(identifier="req_001", capability="spacecraft.upgrade"):
    return {"decision_type": "requirement", "requirement_id": identifier,
            "semantic_capability": capability, "statement": "Trade weapons and crew for spacecraft upgrade"}


def grounded(content, source_id="modrinth:donor"):
    return {"queries": [{"query": "space", "provider_receipts": {
        "modrinth": {"status": "available", "search_requests": 1, "result_count": int(bool(content))}},
        "evidence_records": [{"source_id": source_id, "content": content,
                              "url": "https://example.com/donor"}] if content else []}]}


def test_sibling_candidate_reassessed_and_late_evidence_preserved():
    body = "Unrelated preface. " * 200 + "Build spacecraft. Purchase an upgrade."
    pool = global_grounded_pool({"r_002": grounded(body)})
    trace = requirement_candidate_trace(requirement(), pool)
    assert trace["coverage_complete"]
    assert trace["candidates"][0]["origin_domains"] == ["r_002"]
    assert trace["candidates"][0]["requirement_ref"] == "req_001"
    assert any("upgrade" in item["exact_excerpt"] for item in trace["candidates"][0]["evidence"])
    assert trace["semantic_implementation_proof"] is False


def test_shared_pool_never_transfers_sibling_relevance():
    pool = global_grounded_pool({"r_002": grounded("Space exploration and launch.")})
    trace = requirement_candidate_trace(requirement(), pool)
    assert not trace["coverage_complete"]
    assert trace["candidates"][0]["status"] == "unresolved_relevance"
    assert len(trace["candidates"]) == 1  # keep unresolved candidates, do not discard recall


def test_compound_capability_requires_each_facet_not_one_word():
    pool = global_grounded_pool({"r_001": grounded("spacecraft construction")})
    trace = requirement_candidate_trace(requirement(capability="spacecraft_upgrade"), pool)
    assert not trace["coverage_complete"]
    assert ["upgrade"] in trace["missing_facets"]


def test_task_and_sibling_queries_reach_actual_brief():
    from minecraft_mod_ai.planning_state_research import _research_brief
    state = {"original_prompt": "Mine minerals and trade to build spacecraft",
             "decisions": [requirement(), requirement("req_002", "space.exploration")],
             "research_queue": [{"research_id": "r_001", "requirement_ref": "req_001",
                                  "queries": ["original authored query"], "status": "pending",
                                  "source_kinds": ["existing_mods"]}]}
    brief, _, _ = _research_brief(state["original_prompt"], state)
    domain = brief["domains"][0]
    assert state["original_prompt"] in domain["catalog_queries"]
    assert "space" in domain["catalog_queries"]
    assert domain["task_query_context"]["sibling_requirements"][0]["requirement_id"] == "req_002"


def test_modrinth_paginates_past_keyword_saturation(monkeypatch):
    from minecraft_mod_ai import pre_design_grounded_rag as backend
    offsets = []
    def fetch(url):
        params = parse_qs(urlparse(url).query)
        if "/projects?" in url:
            return []
        offset = int(params["offset"][0])
        offsets.append(offset)
        assert int(params["limit"][0]) == 100
        return {"hits": [{"project_id": str(offset + index), "slug": str(offset + index),
                          "description": "spacecraft upgrade"} for index in range(100 if offset == 0 else 1)],
                "total_hits": 101, "offset": offset}
    monkeypatch.setattr(backend, "_json", fetch)
    monkeypatch.setattr(backend, "_provider_result_limit", lambda: None)
    monkeypatch.setattr(backend, "_provider_page_limit", lambda: None)
    records, receipt = backend._search_modrinth("spacecraft upgrade")
    assert offsets == [0, 100]
    assert len(records) == 101
    assert receipt["search_requests"] == 2


def test_later_provider_failure_preserves_already_retrieved_candidates(monkeypatch):
    from minecraft_mod_ai import pre_design_grounded_rag as backend
    def fetch(url):
        if "/projects?" in url:
            return []
        if int(parse_qs(urlparse(url).query)["offset"][0]):
            raise OSError("provider temporarily unavailable")
        return {"hits": [{"project_id": "first", "slug": "first", "description": "spacecraft"}],
                "total_hits": 2, "offset": 0}
    monkeypatch.setattr(backend, "_json", fetch)
    monkeypatch.setattr(backend, "_provider_result_limit", lambda: None)
    monkeypatch.setattr(backend, "_provider_page_limit", lambda: None)
    records, receipt = backend._search_modrinth("spacecraft")
    assert len(records) == 1
    assert not receipt["retrieval_complete"]
    assert "partial_search_page" in receipt["detail_errors"][0]


def _run_collection(monkeypatch, state, responses):
    from minecraft_mod_ai import planning_state_research as research
    from minecraft_mod_ai import pre_design_grounded_rag as backend
    from minecraft_mod_ai import pre_design_research_pipeline as pipeline
    calls, events = [], []
    monkeypatch.setattr(research, "validate_planning_state", lambda *a, **k: None)
    monkeypatch.setattr(research, "emit_root_cause", lambda *a, **k: events.append(k))
    def retrieve(_backend, _router, brief):
        calls.append(deepcopy(brief))
        return responses[min(len(calls) - 1, len(responses) - 1)]
    monkeypatch.setattr(research, "forced_rag_bundle", retrieve)
    monkeypatch.setattr(pipeline, "_grounded_domain_evidence", lambda domain, bundle: deepcopy(bundle[domain]))
    monkeypatch.setattr(backend, "_materialize_domain_evidence_document", lambda *a: {})
    monkeypatch.setattr(pipeline, "_validate_document_grounding", lambda *a, **k: None)
    monkeypatch.setattr(research, "research_document_domain", lambda *a, **k: {
        "domain_id": k["domain"]["domain_id"], "sufficient": True,
        "claims": [{"claim": "Exact APIs require verification", "evidence_refs": ["api:generic"]}]})
    result = research.collect_planning_state_research(object(), state["original_prompt"], state)
    return result, calls, events


def _state():
    return {"original_prompt": "Build spacecraft and trade weapons", "decisions": [requirement()],
            "research_queue": [{"research_id": "r_001", "requirement_ref": "req_001", "queries": ["api"],
                                "status": "pending", "source_kinds": ["existing_mods"], "resolves": ["u_001"]}],
            "unresolved": [{"unresolved_id": "u_001", "status": "open"}],
            "evidence": [], "resolved": [], "blockers": []}


def test_zero_candidates_and_generic_api_never_complete_or_repeat_exhausted_retry(monkeypatch):
    state = _state()
    empty = {"r_001": grounded("")}
    result, calls, events = _run_collection(monkeypatch, state, [empty])
    assert len(calls) == 2
    assert result["research_queue"][0]["research_state"] == "RESEARCH_BLOCKED"
    assert result["unresolved"][0]["status"] == "open"
    assert result["resolved"] == []
    assert result["blockers"]
    assert any(event.get("result") == "RETRY" for event in events)
    primary = set(calls[0]["domains"][0]["catalog_queries"])
    assert not primary.intersection(calls[1]["domains"][0]["catalog_queries"])
    from minecraft_mod_ai.planning_state_research import collect_planning_state_research
    collect_planning_state_research(object(), state["original_prompt"], result)
    assert len(calls) == 2


def test_corrective_search_and_cross_requirement_evidence_reach_state(monkeypatch):
    state = _state()
    result, calls, _ = _run_collection(monkeypatch, state, [
        {"r_001": grounded("")}, {"r_001": grounded("Build spacecraft and purchase an upgrade.")}])
    assert len(calls) == 2
    assert result["research_queue"][0]["research_state"] == "COMPLETE"
    assert result["candidate_requirement_trace"][0]["coverage_complete"]


def test_nonzero_unrelated_candidates_do_not_complete(monkeypatch):
    result, _, _ = _run_collection(monkeypatch, _state(), [{"r_001": grounded("Decoration furniture and chairs.")}])
    assert result["research_queue"][0]["research_state"] == "RESEARCH_BLOCKED"
    assert result["repository_candidates"] == []


def test_generic_instruction_terms_are_not_requirement_evidence():
    from minecraft_mod_ai.pre_design_domain_research import _domain_terms
    domain = {"requirement": requirement(), "objective": "Find implementation support artifacts",
              "queries": ["Exact test APIs source support artifacts"]}
    assert not _domain_terms(domain).intersection({"exact", "test", "apis", "support", "artifacts"})


def test_restored_false_complete_and_changed_requirement_cannot_enter_planner(monkeypatch):
    from minecraft_mod_ai.planning_candidate_evidence import (
        assert_candidate_research_complete,
    )
    legacy = _state()
    legacy["research_queue"][0].update(status="complete", mod_discovery={"complete": True, "candidates": []})
    with pytest.raises(ValueError, match="RESEARCH_BLOCKED"):
        assert_candidate_research_complete(legacy)
    result, _, _ = _run_collection(monkeypatch, _state(), [
        {"r_001": grounded("spacecraft upgrade")},
    ])
    assert_candidate_research_complete(result)
    result["decisions"][0]["statement"] = "Different authored requirement"
    with pytest.raises(ValueError, match="RESEARCH_BLOCKED"):
        assert_candidate_research_complete(result)


def test_repository_projection_retains_evidence_beyond_preview():
    from minecraft_mod_ai.research_reuse_candidates import project_repository_candidates
    body = "Preface. " * 200 + "spacecraft upgrade"
    rows = project_repository_candidates(
        {"domain_id": "r_002", "evidence_kinds": ["source_code"]}, grounded(body, "github:example/donor"))
    assert "spacecraft upgrade" in rows[0]["evidence_text"]


def test_collection_uses_candidate_discovered_by_other_requirement(monkeypatch):
    state = _state()
    state["decisions"].append(requirement("req_002", "space.exploration"))
    state["research_queue"].append({**deepcopy(state["research_queue"][0]),
                                     "research_id": "r_002", "requirement_ref": "req_002", "resolves": []})
    result, calls, _ = _run_collection(monkeypatch, state, [{
        "r_001": grounded(""),
        "r_002": grounded("space exploration with spacecraft upgrade"),
    }])
    assert len(calls) == 1
    assert result["research_queue"][0]["research_state"] == "COMPLETE"
    assert result["research_queue"][0]["candidate_trace"]["candidates"][0]["origin_domains"] == ["r_002"]
