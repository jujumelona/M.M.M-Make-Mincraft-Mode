from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.inference_time_scaling import (
    _generation_risk,
    _sanitize_legacy_repair_memory,
    _verifier_tier,
)
from minecraft_mod_ai.trajectory_memory import append_trajectory, build_work_trajectory
from minecraft_mod_ai.trajectory_replay import (
    build_generation_replay_context,
    replay_decisions,
)


def _task(*, node: str = "generate-custom-1") -> dict:
    return {
        "node_id": node,
        "stage": "generate:custom",
        "payload": {
            "kind": "integration",
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "1.21.1+build.3",
            "java_version": "21",
        },
    }


def _success_record() -> dict:
    return build_work_trajectory(
        _task(),
        outcome="SUCCESS",
        receipt={
            "build_status": "PASS",
            "commands": [
                {"name": "clean_build", "exit_code": 0, "timed_out": False},
                {"name": "gametest", "exit_code": 0, "timed_out": False},
            ],
            "operations": [
                {"operation": "edit", "status": "PASS"},
            ],
        },
    )


def _failure_record() -> dict:
    return build_work_trajectory(
        _task(node="generate-custom-fail"),
        outcome="FAIL",
        receipt={
            "jdt_error_count": 2,
            "operations": [{"operation": "edit", "status": "PASS"}],
        },
        error="cannot resolve current target symbol",
    )


def test_replay_decisions_use_verified_success_and_failure_boundaries() -> None:
    success = _success_record()
    failure = _failure_record()
    replay = replay_decisions([success, failure], mode="replay")
    counter = replay_decisions([failure, success], mode="counterfactual")
    assert replay
    assert replay[0].source_trajectory_id == success["trajectory_id"]
    assert replay[0].verification_level in {"L3", "L4", "L5"}
    assert counter
    assert counter[0].source_trajectory_id == failure["trajectory_id"]
    assert counter[0].avoid_action == "jdt_diagnostics"


def test_generation_replay_context_is_source_free_and_context_compatible(tmp_path: Path) -> None:
    success = _success_record()
    failure = _failure_record()
    assert append_trajectory(tmp_path, success) is True
    assert append_trajectory(tmp_path, failure) is True
    replay = build_generation_replay_context(
        tmp_path,
        "implement integration register sync and verify",
        router=None,
        target={
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mappings": "1.21.1+build.3",
            "java": "21",
        },
        mode="reuse",
    )
    assert replay is not None
    assert replay["source_free"] is True
    assert replay["trajectory_summaries"]
    rendered = str(replay)
    assert "source_body" not in rendered
    assert "source_code" not in rendered
    assert "current exact repository evidence" in replay["authority"]


def test_generation_replay_rejects_incompatible_platform_memory(tmp_path: Path) -> None:
    assert append_trajectory(tmp_path, _success_record()) is True
    replay = build_generation_replay_context(
        tmp_path,
        "implement integration",
        router=None,
        target={
            "minecraft_version": "1.20.1",
            "loader": "forge",
            "mappings": "official",
            "java": "17",
        },
        mode="replay",
    )
    assert replay is None


def test_verifier_tier_dominates_heuristic_candidate_quality() -> None:
    clean = _verifier_tier({"jdt_status": "AVAILABLE", "jdt_error_count": 0})
    unverified = _verifier_tier({"jdt_status": "NOT_RUN", "jdt_error_count": None})
    verifier_error = _verifier_tier({"jdt_status": "VERIFIER_ERROR", "jdt_error_count": None})
    broken = _verifier_tier({"jdt_status": "AVAILABLE", "jdt_error_count": 3})
    assert clean > unverified > verifier_error > broken


def test_legacy_repair_memory_drops_source_excerpt() -> None:
    sanitized = _sanitize_legacy_repair_memory(
        {
            "similarity": 0.9,
            "signature_sha256": "sha256:x",
            "evidence": {"build_status": "FAIL"},
            "repair_pattern": [
                {
                    "operation": "replace",
                    "path": "src/main/java/A.java",
                    "repair_excerpt": "SECRET_STALE_SOURCE_BODY",
                }
            ],
        }
    )
    assert sanitized["repair_pattern"] == [
        {"operation": "replace", "path": "src/main/java/A.java"}
    ]
    assert "SECRET_STALE_SOURCE_BODY" not in str(sanitized)


def test_high_risk_generation_is_eligible_for_test_time_scaling() -> None:
    module = SimpleNamespace(
        kind="integration",
        config={"network": True, "persistence": True},
        depends_on=("core", "network"),
        required_gates=("jdt", "gametest"),
    )
    assert _generation_risk(module) >= 2
