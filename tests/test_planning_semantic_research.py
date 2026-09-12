import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import planning_semantic_research as semantic
from minecraft_mod_ai.planning_candidate_evidence import (
    expansion_queries,
    fingerprint,
    global_grounded_pool,
    requirement_candidate_trace,
)
from minecraft_mod_ai.planning_state_implementation import _requirement_grounding


def requirement():
    return {
        "decision_type": "requirement",
        "requirement_id": "req_1",
        "semantic_capability": "spacecraft upgrade",
        "statement": "Players trade to upgrade spacecraft.",
        "acceptance": ["Players can trade to upgrade spacecraft."],
    }


def pool_for(*bodies):
    return global_grounded_pool({
        "r_1": {
            "queries": [{
                "query": "spacecraft",
                "evidence_records": [
                    {
                        "source_id": f"modrinth:{index}",
                        "content": body,
                        "url": f"https://example.com/{index}",
                    }
                    for index, body in enumerate(bodies)
                ],
            }]
        }
    })


def observation(req, pool, *, excerpt="Players trade to upgrade spacecraft.", index=0):
    record = pool["queries"][0]["evidence_records"][0]
    return {
        "requirement_sha256": fingerprint(req),
        "obligation_index": index,
        "source_id": record["source_id"],
        "content_sha256": semantic._body_sha(record["content"]),
        "window_start": 0,
        "window_end": len(record["content"]),
        "supports": True,
        "entailment_verified": True,
        "excerpt": excerpt,
        "reason": "Trading changes the ship's performance.",
    }


def review_for(req, pool, observations):
    return {
        "schema_version": "mmm/semantic-research-review-v1",
        "requirement_sha256": fingerprint(req),
        "pool_sha256": fingerprint(pool),
        "observations": observations,
    }


def test_lexical_union_and_negated_capability_never_authorize_completion():
    req = requirement()
    pool = pool_for("A spacecraft cosmetic.", "Upgrade a furnace.")
    trace = requirement_candidate_trace(req, pool)
    assert not trace["lexical_coverage_complete"]
    assert not trace["coverage_complete"]
    negated = requirement_candidate_trace(req, pool_for("No spacecraft upgrade is supported."))
    assert negated["lexical_coverage_complete"]
    assert not negated["coverage_complete"]


@pytest.mark.parametrize("key,value", [
    ("excerpt", "Invented upgrade API"),
    ("content_sha256", "sha256:stale"),
    ("source_id", "modrinth:other"),
    ("window_end", 5),
    ("obligation_index", True),
    ("supports", "true"),
    ("reason", ""),
    ("requirement_sha256", "sha256:old"),
])
def test_host_rejects_forged_or_mismatched_observations(key, value):
    req = requirement()
    pool = pool_for("Players trade to upgrade spacecraft.")
    obs = observation(req, pool)
    obs[key] = value
    assert not semantic.validate_semantic_review(
        req, pool, review_for(req, pool, [obs])
    )["complete"]


def test_partial_acceptance_and_stale_source_fail_resume():
    req = requirement()
    req["acceptance"].append("Players can hire crew members.")
    pool = pool_for("Players trade to upgrade spacecraft.")
    review = review_for(req, pool, [observation(req, pool)])
    checked = semantic.validate_semantic_review(req, pool, review)
    assert not checked["complete"] and checked["missing_obligation_indices"] == [1]
    changed = deepcopy(pool)
    changed["queries"][0]["evidence_records"][0]["content"] += " Changed."
    assert not semantic.validate_semantic_review(req, changed, review)["accepted_proofs"]


def test_query_expansion_keeps_topic_and_never_searches_prose_words_alone():
    req = requirement()
    req["statement"] += " mine other special resources performance"
    queries = expansion_queries({"requirement": req, "original_task": "original Korean text"})
    assert "spacecraft" in queries
    assert all(q == "spacecraft" or q.startswith("spacecraft ") for q in queries)
    assert not set(queries).intersection({"mine", "other", "performance", "upgrade"})


def test_dedup_preserves_all_source_origins_and_queries():
    record = {"source_id": "modrinth:a", "content": "Full body"}
    pool = global_grounded_pool({
        f"r_{i}": {
            "queries": [{"query": f"query {i}", "evidence_records": [record]}]
        }
        for i in range(10)
    })
    assert len(pool["queries"]) == 1
    assert len(pool["queries"][0]["origin_domain_ids"]) == 10
    assert len(pool["queries"][0]["retrieval_queries"]) == 10


def test_utf8_windowing_preserves_entire_body():
    body = "우주선 trade upgrade.\n" * 100
    windows = list(semantic._windows(body, 150))
    assert "".join(window for _, _, window in windows) == body
    assert all(len(window.encode("utf-8")) <= 150 for _, _, window in windows)


def test_small_model_native_tool_calls_have_atomic_schema_and_bounded_input(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from minecraft_mod_ai.model_output_atomicity_contract import assert_atomic_model_schema

    calls = []

    class Router:
        profile = "test"
        registry = SimpleNamespace(role=lambda *a: SimpleNamespace(adapter="llama_cpp"))

        def generate_tool_decision(self, role, messages, **kwargs):
            calls.append(messages)
            assert_atomic_model_schema(kwargs["parameters"], surface="semantic review")
            assert len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) < 4096
            assert set(kwargs["parameters"]["properties"]) == {
                "verdict", "evidence_start", "evidence_end"
            }
            return {
                "verdict": "insufficient",
                "evidence_start": -1,
                "evidence_end": -1,
            }

    monkeypatch.setattr(semantic, "request_message_budget", lambda *a: 4096)
    req = requirement()
    pool = pool_for("Spacecraft documentation.\n" * 1000)
    result = semantic.review_requirement_sources(
        Router(), req, pool, requirement_candidate_trace(req, pool)
    )
    assert len(calls) > 1
    assert not result["complete"]


def test_log_scale_pool_cannot_expand_mandatory_planner_input(monkeypatch):
    req = requirement()
    pool = pool_for(
        "Players trade to upgrade spacecraft.",
        *["Unrelated body " * 200 for _ in range(15167)],
    )
    review = semantic.validate_semantic_review(
        req, pool, review_for(req, pool, [observation(req, pool)])
    )
    state = {
        "decisions": [req],
        "task_candidate_pool": pool,
        "research_queue": [{
            "research_id": "r_1",
            "requirement_ref": "req_1",
            "status": "complete",
        }],
        "evidence": [{
            "research_ref": "r_1",
            "sufficient": True,
            "claims": ["x" * 50636559],
            "candidate_trace": {"semantic_review": review},
        }],
    }
    evidence, refs = _requirement_grounding(state, "req_1")
    assert len(json.dumps(evidence).encode("utf-8")) < 44724
    assert refs == {"modrinth:0"}
    assert len(pool["queries"]) == 15168
    assert len(state["evidence"][0]["claims"][0]) == 50636559

    from minecraft_mod_ai import planning_criterion_fragments as fragments

    class CapturedModelBoundary(Exception):
        pass

    def inspect(_router, _identifiers, *, context, allowed_refs, **kwargs):
        encoded = json.dumps(context, ensure_ascii=False).encode("utf-8")
        assert len(encoded) < 44724
        assert allowed_refs == {"modrinth:0"}
        assert "Players trade to upgrade spacecraft." in encoded.decode("utf-8")
        raise CapturedModelBoundary

    monkeypatch.setattr(fragments, "_section_results", inspect)
    with pytest.raises(CapturedModelBoundary):
        fragments.generate_section_records(
            object(),
            requirement=req,
            criterion=req["acceptance"][0],
            section="state_model",
            evidence=evidence,
            allowed_refs=refs,
        )


def test_each_obligation_budget_does_not_include_all_sibling_obligations(monkeypatch):
    req = requirement()
    req["acceptance"] = [f"Players can upgrade spacecraft variant {i}." for i in range(300)]
    pool = pool_for("Spacecraft upgrades.")
    monkeypatch.setattr(semantic, "request_message_budget", lambda *a: 4096)
    monkeypatch.setattr(
        semantic,
        "generate_fixed_template_value",
        lambda *a, **k: {
            "verdict": "insufficient",
            "evidence_start": -1,
            "evidence_end": -1,
        },
    )
    result = semantic.review_requirement_sources(
        object(), req, pool, requirement_candidate_trace(req, pool)
    )
    assert not result["complete"]


def test_review_restarts_from_durable_observations(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    req = requirement()
    pool = pool_for("Spacecraft note A.", "Spacecraft note B.")
    router = SimpleNamespace(
        profile="fixture",
        registry=SimpleNamespace(role=lambda *a: SimpleNamespace()),
    )

    def model(*args, **kwargs):
        text = args[2][1]["content"]
        calls.append(text)
        if "note B" in text and len(calls) == 2:
            raise TimeoutError("interrupted")
        return {
            "verdict": "insufficient",
            "evidence_start": -1,
            "evidence_end": -1,
        }

    monkeypatch.setattr(semantic, "generate_fixed_template_value", model)
    with pytest.raises(Exception, match="interrupted"):
        semantic.review_requirement_sources(
            router, req, pool, requirement_candidate_trace(req, pool)
        )
    result = semantic.review_requirement_sources(
        router, req, pool, requirement_candidate_trace(req, pool)
    )
    assert not result["complete"]
    assert len(calls) == 3


def test_extending_pool_preserves_previously_completed_review(monkeypatch):
    from test_planning_candidate_evidence import _run_collection, _state, grounded
    from test_planning_candidate_evidence import requirement as req_fixture

    state, _, _ = _run_collection(
        monkeypatch, _state(), [{"r_001": grounded("spacecraft upgrade")}]
    )
    state["decisions"].append(req_fixture("req_002"))
    state["research_queue"].append({
        "research_id": "r_002",
        "requirement_ref": "req_002",
        "queries": ["new"],
        "status": "pending",
        "source_kinds": ["existing_mods"],
        "resolves": [],
    })
    updated, _, _ = _run_collection(
        monkeypatch,
        state,
        [{"r_002": grounded("spacecraft upgrade new", "modrinth:new")}],
    )
    from minecraft_mod_ai.planning_candidate_evidence import assert_candidate_research_complete

    assert_candidate_research_complete(updated)
    assert len(updated["task_candidate_pool"]["queries"]) == 2
    assert len(updated["candidate_requirement_trace"]) == 2


def test_exact_quote_requires_independent_entailment_check(monkeypatch):
    req = requirement()
    pool = pool_for("Spacecraft upgrades are not supported.")
    calls = []

    def model(*args, **kwargs):
        calls.append(kwargs["tool_name"])
        if kwargs["tool_name"] == "assess_requirement_source":
            return {"verdict": "supported", "evidence_start": 0, "evidence_end": 0}
        return {"verdict": "negated"}

    monkeypatch.setattr(semantic, "generate_fixed_template_value", model)
    result = semantic.review_requirement_sources(
        object(), req, pool, requirement_candidate_trace(req, pool)
    )
    assert calls == ["assess_requirement_source", "verify_requirement_entailment"]
    assert not result["complete"]


def test_verifier_sees_negating_header_not_only_cherry_picked_quote(monkeypatch):
    req = requirement()
    quote = "Players trade to upgrade spacecraft."
    body = "Unsupported features:\n" + quote
    pool = pool_for(body)

    def model(*args, **kwargs):
        context = json.loads(args[2][1]["content"])
        if kwargs["tool_name"] == "verify_requirement_entailment":
            assert context["source_window"] == body
            assert context["source_quote"] == body
            return {"verdict": "negated"}
        return {"verdict": "supported", "evidence_start": 0, "evidence_end": 0}

    monkeypatch.setattr(semantic, "generate_fixed_template_value", model)
    result = semantic.review_requirement_sources(
        object(), req, pool, requirement_candidate_trace(req, pool)
    )
    assert not result["complete"]


def test_backend_endpoint_change_and_malformed_cache_do_not_reuse_verdict(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base_url="http://localhost:8910", extra={"temperature": 0.1})
    router = SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(role=lambda *a: config),
    )
    calls = []
    monkeypatch.setattr(
        semantic,
        "generate_fixed_template_value",
        lambda *a, **k: (
            calls.append(1)
            or {
                "verdict": "insufficient",
                "evidence_start": -1,
                "evidence_end": -1,
            }
        ),
    )
    req = requirement()
    pool = pool_for("Spacecraft upgrade")
    trace = requirement_candidate_trace(req, pool)
    semantic.review_requirement_sources(router, req, pool, trace)
    config.base_url = "http://localhost:8911"
    semantic.review_requirement_sources(router, req, pool, trace)
    assert len(calls) == 2
    for path in (tmp_path / ".mmm/semantic-research-cache").glob("*.json"):
        path.write_text("[]", encoding="utf-8")
    semantic.review_requirement_sources(router, req, pool, trace)
    assert len(calls) == 3


def test_reference_research_without_requirement_does_not_invent_acceptance_review():
    from minecraft_mod_ai.planning_state_research import _research_brief

    state = {
        "research_queue": [{
            "research_id": "reference",
            "status": "pending",
            "source_kinds": ["repository"],
            "queries": ["named repository"],
        }]
    }
    brief, _, _ = _research_brief("Inspect named repository", state)
    assert "requirement" not in brief["domains"][0]
    assert brief["domains"][0]["queries"] == ["named repository"]


def test_semantic_rejection_still_triggers_material_corrective_search(monkeypatch):
    from test_planning_candidate_evidence import _run_collection, _state, grounded

    state = _state()
    # Contains both capability words but fixture reviewer does not admit uppercase text.
    result, calls, _ = _run_collection(monkeypatch, state, [
        {"r_001": grounded("SPACECRAFT UPGRADE is not supported.")},
        {"r_001": grounded("spacecraft upgrade through trading")},
    ])
    assert len(calls) == 2
    assert result["research_queue"][0]["research_state"] == "COMPLETE"
