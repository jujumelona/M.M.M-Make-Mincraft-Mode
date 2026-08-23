from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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


def test_semantic_branches_use_exact_requirement_evidence() -> None:
    requirements = [
        {
            "requirement_id": "req_network",
            "capability": "network.trade_sync",
            "statement": "Synchronize server-authoritative trade state.",
            "provides": ["capability:network.trade_sync"],
            "source_span": {"text": "Synchronize server-authoritative trade state."},
        },
        {
            "requirement_id": "req_music",
            "capability": "music.syncopation",
            "statement": "Add syncopation to music playback.",
            "provides": ["capability:music.syncopation"],
            "source_span": {"text": "Add syncopation to music playback."},
        },
    ]

    branches = contract._semantic_branch_predicates(
        requirements,
        (),
        {"project_topology": {"loaders": ["fabric"]}},
    )

    assert branches["needs_network"]["status"] == "ACTIVE"
    assert branches["needs_network"]["evidence_refs"] == ["req_network"]
    assert branches["needs_registry"]["status"] == "NOT_APPLICABLE"
    assert branches["needs_loader_leaf"]["status"] == "NOT_APPLICABLE"


def test_semantic_branches_bind_multiloader_to_target_topology() -> None:
    branches = contract._semantic_branch_predicates(
        (),
        (),
        {"project_topology": {"loaders": ["fabric", "neoforge"]}},
    )

    assert branches["needs_loader_leaf"] == {
        "predicate": "needs_loader_leaf",
        "status": "ACTIVE",
        "evidence_refs": ["target-topology:multiple-loaders"],
        "reason": "activated by exact requirement/component/topology evidence",
    }


def test_automatic_target_rejects_base_optimizer_fallback() -> None:
    result = SimpleNamespace()

    with pytest.raises(ValueError, match="joint evidence-backed reuse optimization"):
        contract._require_evidence_backed_optimization(
            result,
            automatic_target=True,
        )

    result._mmm_reuse_plan = {
        "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
        "capabilities": [{"capability": "trade.transaction", "mode": "fresh"}],
    }
    assert (
        contract._require_evidence_backed_optimization(
            result,
            automatic_target=True,
        )
        is result
    )


def test_handoff_is_the_batch_graph_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {
        "plan_sha256": "plan-sha",
        "request_catalog": {
            "prompt_sha256": "prompt-sha",
            "requirements": [
                {
                    "requirement_id": "req_a",
                    "capability": "a",
                    "statement": "A",
                },
                {
                    "requirement_id": "req_b",
                    "capability": "b",
                    "statement": "B",
                },
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
                # Intentionally empty: live lowering must use the handoff graph edge.
                "depends_on": [],
                "provides": ["b"],
                "acceptance": ["B passes"],
            },
        ],
    }
    monkeypatch.setattr(
        contract,
        "build_evidence_first_handoff",
        lambda _plan: {
            "handoff_sha256": "handoff-sha",
            "work_graph": {
                "task_refs": ["task_a", "task_b"],
                "edges": [
                    {
                        "from_task_ref": "task_a",
                        "to_task_ref": "task_b",
                    }
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
        },
    )

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


def test_execution_observation_replans_only_on_unexpected_index_drift() -> None:
    tasks = [
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
    observation = {
        "schema_version": "mmm/semantic-task-observation-v1",
        "task_id": "task_a",
        "touched_paths": ["src/main/java/example/A.java"],
        "affected_downstream_task_ids": ["task_b"],
        "observation_sha256": "old",
    }
    previous_index = {
        "files": [
            {
                "path": "src/main/java/example/A.java",
                "sha256": "sha256:old-a",
            },
            {
                "path": "src/main/java/example/B.java",
                "sha256": "sha256:old-b",
            },
        ]
    }

    expected_only = {
        "files": [
            {
                "path": "src/main/java/example/A.java",
                "sha256": "sha256:new-a",
            },
            {
                "path": "src/main/java/example/B.java",
                "sha256": "sha256:old-b",
            },
        ]
    }
    enriched = contract._enrich_execution_observation(
        observation,
        tasks=tasks,
        previous_index=previous_index,
        current_index=expected_only,
        completed_task_ids=(),
    )
    assert enriched["replan_required"] is False
    assert enriched["impact_replan_scope"] == ["task_b"]

    unexpected_drift = {
        "files": [
            {
                "path": "src/main/java/example/A.java",
                "sha256": "sha256:new-a",
            },
            {
                "path": "src/main/java/example/B.java",
                "sha256": "sha256:external-change",
            },
        ]
    }
    enriched = contract._enrich_execution_observation(
        observation,
        tasks=tasks,
        previous_index=previous_index,
        current_index=unexpected_drift,
        completed_task_ids=(),
    )
    assert enriched["replan_required"] is True
    assert set(enriched["project_index_refresh"]["changed_paths"]) == {
        "src/main/java/example/A.java",
        "src/main/java/example/B.java",
    }
    assert enriched["observation_sha256"].startswith("sha256:")
