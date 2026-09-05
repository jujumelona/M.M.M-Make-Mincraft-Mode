from __future__ import annotations

from dataclasses import dataclass

import pytest

import minecraft_mod_ai.evidence_first_pipeline_contract as contract


@dataclass(frozen=True)
class _Batch:
    batch_id: str
    scope: str
    depends_on_batches: tuple[str, ...]
    deliverables: tuple[str, ...]
    exports: tuple[str, ...]
    task_contract: dict | None = None
    evidence_plan_sha256: str = ""
    acceptance_tests: tuple[str, ...] = ()


def test_install_keeps_existing_planning_and_target_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minecraft_mod_ai import evidence_first_planning, platform_resolver

    branch_owner = evidence_first_planning._branch_predicates
    target_owner = platform_resolver._optimize
    calls: list[str] = []

    monkeypatch.setattr(contract, "_INSTALLED", False)
    monkeypatch.setattr(
        contract,
        "_install_handoff_owner",
        lambda: calls.append("handoff"),
    )
    monkeypatch.setattr(
        contract,
        "_install_execution_impact",
        lambda: calls.append("execution"),
    )

    contract.install()

    assert calls == ["handoff", "execution"]
    assert evidence_first_planning._branch_predicates is branch_owner
    assert platform_resolver._optimize is target_owner


def test_handoff_is_the_batch_graph_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {
        "plan_sha256": "plan-sha",
        "request_catalog": {
            "prompt_sha256": "prompt-sha",
            "requirements": [
                {"requirement_id": "req_a", "capability": "a", "statement": "A"},
                {"requirement_id": "req_b", "capability": "b", "statement": "B"},
            ],
        },
        "tasks": [
            {
                "task_id": "task_a",
                "semantic_outcome": "Implement A",
                "requirement_refs": ["req_a"],
                "depends_on": [],
                "provides": ["a"],
                "acceptance": ["A passes"],
            },
            {
                "task_id": "task_b",
                "semantic_outcome": "Implement B",
                "requirement_refs": ["req_b"],
                "depends_on": [],
                "provides": ["b"],
                "acceptance": ["B passes"],
            },
        ],
    }
    handoff = {
        "source_plan_sha256": "plan-sha",
        "handoff_sha256": "handoff-sha",
        "work_graph": {
            "task_refs": ["task_a", "task_b"],
            "edges": [
                {"from_task_ref": "task_a", "to_task_ref": "task_b"}
            ],
        },
        "production_modules": [
            {
                "production_module_id": "pm-a",
                "task_ref": "task_a",
                "module_id": "common",
                "source_set": "main",
            }
        ],
        "asset_requests": [
            {
                "asset_request_id": "asset-b",
                "task_ref": "task_b",
                "locator": "assets/example/model.json",
            }
        ],
    }
    monkeypatch.setattr(contract, "validate_evidence_first_plan", lambda _plan: None)
    monkeypatch.setattr(contract, "build_evidence_first_handoff", lambda _plan: handoff)
    # Typed execution lowering has its own contract tests. This fixture intentionally
    # isolates the canonical WorkGraph-to-batch ownership behavior.
    monkeypatch.setattr(contract, "execution_plan", lambda value: value)
    monkeypatch.setattr(
        contract,
        "execution_handoff",
        lambda _plan, canonical, _lowered: canonical,
    )
    monkeypatch.setattr(contract, "validate_plan_collect_all", lambda _plan, _handoff: None)

    batches = contract._batches_from_handoff(plan, batch_type=_Batch)

    assert [item.batch_id for item in batches] == ["task_a", "task_b"]
    assert batches[0].depends_on_batches == ()
    assert batches[1].depends_on_batches == ("task_a",)
    assert batches[0].task_contract["handoff_sha256"] == "handoff-sha"
    assert batches[0].task_contract["production_bindings"][0]["module_id"] == "common"
    assert (
        batches[1].task_contract["asset_bindings"][0]["locator"]
        == "assets/example/model.json"
    )


def test_handoff_must_bind_the_exact_validated_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "plan_sha256": "expected",
        "request_catalog": {"requirements": []},
        "tasks": [],
    }
    monkeypatch.setattr(contract, "validate_evidence_first_plan", lambda _plan: None)
    monkeypatch.setattr(
        contract,
        "build_evidence_first_handoff",
        lambda _plan: {
            "source_plan_sha256": "stale",
            "handoff_sha256": "handoff",
            "work_graph": {"task_refs": [], "edges": []},
            "production_modules": [],
            "asset_requests": [],
        },
    )

    with pytest.raises(ValueError, match="exact source plan hash"):
        contract._batches_from_handoff(plan, batch_type=_Batch)


def _impact_tasks() -> list[dict]:
    return [
        {
            "task_id": "task_a",
            "depends_on": [],
            "owned_paths": ["src/main/java/example/A.java"],
        },
        {
            "task_id": "task_b",
            "depends_on": ["task_a"],
            "owned_paths": ["src/main/java/example/B.java"],
        },
    ]


def _observation() -> dict:
    return {
        "schema_version": "mmm/semantic-task-observation-v1",
        "task_id": "task_a",
        "touched_paths": ["src/main/java/example/A.java"],
        "affected_downstream_task_ids": ["task_b"],
        "observation_sha256": "old",
    }


def _index(a: str, b: str) -> dict:
    return {
        "files": [
            {"path": "src/main/java/example/A.java", "sha256": a},
            {"path": "src/main/java/example/B.java", "sha256": b},
        ]
    }


def test_execution_observation_replans_only_incomplete_affected_tasks() -> None:
    previous = _index("sha256:old-a", "sha256:old-b")
    expected_only = _index("sha256:new-a", "sha256:old-b")

    enriched = contract._enrich_execution_observation(
        _observation(),
        tasks=_impact_tasks(),
        previous_index=previous,
        current_index=expected_only,
        completed_task_ids=(),
    )

    assert enriched["replan_required"] is True
    assert enriched["impact_replan_scope"] == ["task_b"]
    assert enriched["unexpected_drift_paths"] == []
    assert enriched["observation_sha256"].startswith("sha256:")

    completed = contract._enrich_execution_observation(
        _observation(),
        tasks=_impact_tasks(),
        previous_index=previous,
        current_index=expected_only,
        completed_task_ids=("task_b",),
    )
    assert completed["replan_required"] is False
    assert completed["impact_replan_scope"] == []


def test_execution_observation_records_unexpected_index_drift() -> None:
    previous = _index("sha256:old-a", "sha256:old-b")
    unexpected = _index("sha256:new-a", "sha256:external-change")

    enriched = contract._enrich_execution_observation(
        _observation(),
        tasks=_impact_tasks(),
        previous_index=previous,
        current_index=unexpected,
        completed_task_ids=(),
    )

    assert enriched["replan_required"] is True
    assert enriched["unexpected_drift_paths"] == [
        "src/main/java/example/B.java"
    ]
    assert set(enriched["project_index_refresh"]["changed_paths"]) == {
        "src/main/java/example/A.java",
        "src/main/java/example/B.java",
    }
