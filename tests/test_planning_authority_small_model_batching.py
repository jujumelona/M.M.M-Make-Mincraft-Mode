from __future__ import annotations

from minecraft_mod_ai import planning_authority as authority


def test_authoritative_catalog_uses_two_constant_batch_turns(monkeypatch):
    prompt = "build a rover; drive it to a launch pad"
    router = object()
    calls: list[str] = []

    catalog = {
        "schema_version": "mmm/approved-requirement-graph-v1",
        "requirements": [
            {
                "requirement_id": "req-rover",
                "depends_on": [],
            },
            {
                "requirement_id": "req-drive",
                "depends_on": [],
            },
        ],
        "requirement_graph": {
            "node_ids": ["req-rover", "req-drive"],
            "edges": [],
        },
        "semantic_audit": {
            "normal_model_turns": 1,
            "semantic_model_turns": 1,
        },
        "catalog_sha256": "original",
    }

    def fake_semantic_builder(value, *, router=None):
        assert value == prompt
        assert router is not None
        calls.append("semantic_batch")
        return catalog

    def fake_retrieval_planner(active_router, value, requirements):
        assert active_router is router
        assert value == prompt
        assert [item["requirement_id"] for item in requirements] == [
            "req-rover",
            "req-drive",
        ]
        calls.append("retrieval_batch")
        return {"requirements": "opaque-to-authority"}

    monkeypatch.setattr(
        authority._semantic,
        "build_approved_requirement_catalog",
        fake_semantic_builder,
    )
    monkeypatch.setattr(
        authority._semantic,
        "_clause_records",
        lambda value: [
            {"clause_index": 0, "text": "build a rover"},
            {"clause_index": 1, "text": "drive it to a launch pad"},
        ],
    )
    monkeypatch.setattr(
        authority._semantic,
        "validate_approved_requirement_catalog",
        lambda value, *, prompt: None,
    )
    monkeypatch.setattr(
        authority._evidence,
        "_hash_without",
        lambda value, field: "sha256:test",
    )
    monkeypatch.setattr(
        authority._retrieval,
        "_call_retrieval_planner",
        fake_retrieval_planner,
    )
    monkeypatch.setattr(
        authority._retrieval,
        "_normalize_retrieval_plan",
        lambda value, requirements, payload: {
            "req-rover": {
                "depends_on": [],
                "search_queries": ["minecraft rover mod source", "minecraft vehicle implementation"],
            },
            "req-drive": {
                "depends_on": ["req-rover"],
                "search_queries": ["minecraft vehicle movement source", "minecraft rover driving mod"],
            },
        },
    )

    result = authority.build_authoritative_request_catalog(prompt, router)

    assert calls == ["semantic_batch", "retrieval_batch"]
    assert result["semantic_audit"]["normal_model_turns"] == 2
    assert result["semantic_audit"]["semantic_model_turns"] == 1
    assert result["semantic_audit"]["semantic_detail_model_turns"] == 0
    assert result["semantic_audit"]["retrieval_model_turns"] == 1
    assert result["semantic_audit"]["retrieval_query_planning"] == (
        "all_requirements_one_structured_batch"
    )
    assert result["requirement_graph"]["edges"] == [["req-rover", "req-drive"]]
    assert result["requirements"][1]["depends_on"] == ["req-rover"]


def test_batch_compatibility_seams_delegate_once(monkeypatch):
    semantic_calls: list[tuple[object, object]] = []
    retrieval_calls: list[tuple[object, str, object]] = []
    router = object()
    clauses = [{"clause_index": 0, "text": "one"}]
    requirements = [{"requirement_id": "req-one"}]

    monkeypatch.setattr(
        authority._semantic,
        "_call_semantic_model",
        lambda active_router, active_clauses: semantic_calls.append(
            (active_router, active_clauses)
        )
        or {"requirements": []},
    )
    monkeypatch.setattr(
        authority._retrieval,
        "_call_retrieval_planner",
        lambda active_router, prompt, active_requirements: retrieval_calls.append(
            (active_router, prompt, active_requirements)
        )
        or {"requirements": []},
    )

    authority._call_semantic_compiler(router, clauses)
    authority._call_retrieval_planner(router, "prompt", requirements)

    assert semantic_calls == [(router, clauses)]
    assert retrieval_calls == [(router, "prompt", requirements)]
