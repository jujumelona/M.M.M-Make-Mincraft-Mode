"""Resume must not re-search unchanged targets or reuse stale feature judgments."""
from minecraft_mod_ai import authored_reuse_bridge as bridge
from minecraft_mod_ai import grounded_source_reuse


def test_target_bound_resume_reuses_inspection_and_invalidates_changed_inputs(monkeypatch):
    calls = []
    def fake_plan(design, **_kwargs):
        calls.append(design)
        return {
            "schema_version": "mmm/grounded-repository-reuse-plan-v2",
            "source_plan_sha256": design["_pre_retrieval_plan"]["plan_sha256"],
            "capability_graph": design["_pre_retrieval_plan"]["capability_graph"],
            "bound_target": design["_platform_selection"]["target"],
            "requested_curseforge_dependencies": [],
            "capabilities": [],
            "reference_sources": [],
        }
    monkeypatch.setattr(grounded_source_reuse, "build_repository_reuse_plan", fake_plan)
    scope = {"requirements": [{"requirement_id": "r1", "search_query": "teleport"}]}
    args = {"minecraft_version": "1.21.1", "loader": "fabric", "request_scope": scope}
    first = bridge.resolve_authored_source_reuse(
        "Build a teleport block", {"behavior_contract": {"description": "teleport"}},
        **args,
    )
    assert len(calls) == 1
    second = bridge.resolve_authored_source_reuse(
        "Build a teleport block", {"behavior_contract": {"description": "teleport"}},
        prior_reuse_plan=first, **args,
    )
    assert len(calls) == 1
    assert second["planning_cache_reused"] is True
    assert second["incremental_source_reuse_costs"]["model_judgement_calls"] == 0

    bridge.resolve_authored_source_reuse(
        "Build a teleport block with cooldown", {"behavior_contract": {"description": "teleport"}},
        prior_reuse_plan=first, **args,
    )
    assert len(calls) == 2
    bridge.resolve_authored_source_reuse(
        "Build a teleport block", {"behavior_contract": {"description": "teleport and damage"}},
        prior_reuse_plan=first, **args,
    )
    assert len(calls) == 3
    bridge.resolve_authored_source_reuse(
        "Build a teleport block", {"behavior_contract": {"description": "teleport"}},
        prior_reuse_plan=first,
        minecraft_version="1.20.1", loader="fabric", request_scope=scope,
    )
    assert len(calls) == 4
    bridge.resolve_authored_source_reuse(
        "Build a teleport block", {"behavior_contract": {"description": "teleport"}},
        prior_reuse_plan=first,
        repository_candidates=[{"repository": "other/donor", "reference_only": True}],
        **args,
    )
    assert len(calls) == 5
