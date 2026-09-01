from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai import agentic_optimization_contract, model_router, work_graph
from minecraft_mod_ai import small_model_max_agent_contract as max_agent
from minecraft_mod_ai.active_repair_verifier_contract import _ambiguous
from minecraft_mod_ai.counterexample_verifier import _synthetic_verification
from minecraft_mod_ai.remote_skill_store_consent import (
    CONSENT_ENV,
    remote_write_allowed,
    require_remote_write_consent,
    sanitize_remote_payload,
)
from minecraft_mod_ai.remote_trajectory_store import _remote_path, queue_remote_record
from minecraft_mod_ai.small_model_hybrid_search_contract import _route
from minecraft_mod_ai.trajectory_memory import (
    append_trajectory,
    build_work_trajectory,
    relevant_trajectories,
    synthesize_temporary_skill,
)
from minecraft_mod_ai.trajectory_verification import (
    TRAJECTORY_SCHEMA_VERSION,
    record_remote_eligible,
    record_strong_skill_eligible,
)


def _repair_task() -> dict[str, object]:
    return {
        "node_id": "module-demo",
        "stage": "repair",
        "payload": {
            "kind": "custom_java",
            "source_body": "public class SecretSource {}",
            "members": [{"module_id": "demo"}],
        },
    }


def _build_receipt(*, gametest: bool = False) -> dict[str, object]:
    commands: list[dict[str, object]] = [
        {"name": "clean_build", "exit_code": 0, "timed_out": False}
    ]
    if gametest:
        commands.append({"name": "gametest", "exit_code": 0, "timed_out": False})
    return {"build": {"status": "PASS", "commands": commands}}


def test_colab_remote_trajectory_storage_is_explicit_opt_in() -> None:
    notebook = json.loads(Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(str(cell.get("source", "")) for cell in notebook.get("cells", []))
    assert "ALLOW_REMOTE_TRAJECTORY_STORE = False" in source
    assert "MMM_REMOTE_TRAJECTORY_STORE_CONSENT" in source
    assert '"1" if ALLOW_REMOTE_TRAJECTORY_STORE else "0"' in source


def test_remote_write_gate_defaults_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    row = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=_build_receipt(gametest=True))
    monkeypatch.delenv(CONSENT_ENV, raising=False)
    assert not remote_write_allowed()
    with pytest.raises(PermissionError):
        require_remote_write_consent()
    assert queue_remote_record(tmp_path, row) is False
    assert not (tmp_path / ".minecraft_ai" / "trajectory-memory" / "remote-outbox.jsonl").exists()


def test_remote_payload_sanitizer_removes_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONSENT_ENV, "1")
    assert remote_write_allowed()
    require_remote_write_consent()
    value = sanitize_remote_payload(
        {
            "task": "repair",
            "token": "must-not-leave",
            "nested": {
                "Authorization": "Bearer secret",
                "safe": "verified",
                "api_key": "hidden",
            },
        }
    )
    assert value == {"task": "repair", "nested": {"safe": "verified"}}


def test_model_claim_only_is_l0_and_never_persisted(tmp_path: Path) -> None:
    row = build_work_trajectory(
        _repair_task(),
        outcome="SUCCESS",
        receipt={"status": "PASS", "message": "model says done"},
    )
    assert row["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert row["verification"]["level"] == "L0"
    assert row["verification"]["memory_eligible"] is False
    assert append_trajectory(tmp_path, row) is False


def test_server_running_without_behavior_assertions_stays_l0(tmp_path: Path) -> None:
    row = build_work_trajectory(
        _repair_task(),
        outcome="SUCCESS",
        receipt={"runtime": {"status": "PASS", "server_running": True}},
    )
    assert row["verification"]["level"] == "L0"
    assert append_trajectory(tmp_path, row) is False


def test_generic_quality_pass_without_signed_evidence_does_not_become_l5() -> None:
    receipt = {
        **_build_receipt(gametest=True),
        "quality_evidence": {"status": "PASS", "evidence": ["model-claim"]},
    }
    row = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=receipt)
    assert row["verification"]["level"] == "L3"
    assert row["verification"]["checks"]["acceptance"] is False


def test_signed_quality_receipt_with_evidence_can_raise_l5() -> None:
    receipt = {
        **_build_receipt(gametest=True),
        "quality": {
            "dimension_id": "correctness",
            "status": "PASS",
            "receipt_id": "quality:correctness:abc123",
            "verified_by": "mmm.quality-evidence-adapter/v1",
            "evidence_refs": ["evidence:gametest:abc", "evidence:build:def"],
        },
    }
    row = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=receipt)
    assert row["verification"]["level"] == "L5"
    assert row["verification"]["checks"]["acceptance"] is True


def test_l2_build_pass_is_local_weak_memory_not_proven_skill(tmp_path: Path) -> None:
    row = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=_build_receipt())
    assert row["verification"]["level"] == "L2"
    assert row["verification"]["memory_eligible"] is True
    assert record_strong_skill_eligible(row) is False
    assert record_remote_eligible(row) is False
    assert append_trajectory(tmp_path, row) is True


def test_l3_gametest_pass_is_strong_and_remote_eligible(tmp_path: Path) -> None:
    row = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=_build_receipt(gametest=True))
    assert row["verification"]["level"] == "L3"
    assert row["verification"]["checks"]["tests"] is True
    assert record_strong_skill_eligible(row) is True
    assert record_remote_eligible(row) is True
    assert append_trajectory(tmp_path, row) is True
    assert _remote_path("repair") == "memory/v3/repair.jsonl"


def test_build_only_synthetic_probe_does_not_fake_l3() -> None:
    synthetic = _synthetic_verification(
        plan={"probes": ["gradle_build", "json_resource_parse"]},
        build_status="PASS",
        json_ok=True,
        commands=[{"name": "clean_build", "exit_code": 0, "timed_out": False}],
    )
    row = build_work_trajectory(
        _repair_task(),
        outcome="SUCCESS",
        receipt={
            "counterexample_result": {
                "status": "PASS",
                "synthetic_verification": synthetic,
                "commands": [{"name": "clean_build", "exit_code": 0, "timed_out": False}],
            }
        },
    )
    assert row["verification"]["level"] == "L2"
    assert row["verification"]["checks"]["tests"] is False


def test_verified_jdt_failure_becomes_negative_memory(tmp_path: Path) -> None:
    row = build_work_trajectory(
        _repair_task(),
        outcome="FAIL",
        receipt={"jdt_status": "FAIL", "jdt_error_count": 2},
        error="cannot resolve symbol RegistryKey",
    )
    assert row["verification"]["verified_failure"] is True
    assert row["verification"]["failure_level"] == "L1"
    assert row["verification"]["memory_eligible"] is True
    assert row["verification"]["remote_eligible"] is True
    assert append_trajectory(tmp_path, row) is True


def test_temporary_skill_uses_only_strong_success_and_verified_failure(tmp_path: Path) -> None:
    weak = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=_build_receipt())
    strong = build_work_trajectory(_repair_task(), outcome="SUCCESS", receipt=_build_receipt(gametest=True))
    failure = build_work_trajectory(
        _repair_task(),
        outcome="FAIL",
        receipt={"jdt_status": "FAIL", "jdt_error_count": 1},
        error="cannot resolve symbol RegistryKey",
    )
    assert append_trajectory(tmp_path, weak)
    assert append_trajectory(tmp_path, strong)
    assert append_trajectory(tmp_path, failure)
    records = relevant_trajectories(
        tmp_path,
        "repair custom_java RegistryKey",
        task_class="repair",
        router=None,
        limit=6,
    )
    skill = synthesize_temporary_skill("repair custom_java RegistryKey", records, task_class="repair")
    assert skill is not None
    assert skill["schema_version"] == "mmm/temporary-skill-v3"
    assert skill["ephemeral"] is True
    assert skill["proven_patterns"]
    assert skill["avoid_patterns"]
    assert strong["trajectory_id"] in skill["source_trajectory_ids"]
    assert failure["trajectory_id"] in skill["source_trajectory_ids"]
    assert weak["trajectory_id"] not in skill["source_trajectory_ids"]


def test_synthetic_counterexample_format_records_actual_probes() -> None:
    value = _synthetic_verification(
        plan={"probes": ["gradle_build", "json_resource_parse", "gametest_if_available"]},
        build_status="PASS",
        json_ok=True,
        commands=[
            {"name": "clean_build", "exit_code": 0, "timed_out": False},
            {"name": "gametest", "exit_code": 0, "timed_out": False},
        ],
    )
    assert value["schema_version"] == "mmm/synthetic-verification-v1"
    assert value["status"] == "PASS"
    assert value["isolated_snapshot"] is True
    assert {item["scenario_id"] for item in value["scenarios"]} == {
        "json_resource_parse",
        "gradle_clean_build",
        "fabric_gametest",
    }


def test_active_verifier_escalates_only_ambiguous_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_ACTIVE_REPAIR_VERIFIER", "auto")
    monkeypatch.setenv("MMM_ACTIVE_REPAIR_SCORE_MARGIN", "80")
    close = [
        (1000.0, 0, [], {"jdt_status": "PASS", "jdt_error_count": 0}),
        (950.0, 1, [], {"jdt_status": "PASS", "jdt_error_count": 0}),
    ]
    clear = [
        (1000.0, 0, [], {"jdt_status": "PASS", "jdt_error_count": 0}),
        (700.0, 1, [], {"jdt_status": "AVAILABLE", "jdt_error_count": 2}),
    ]
    assert _ambiguous(close) is True
    assert _ambiguous(clear) is False


def test_code_retrieval_routes_by_task_shape() -> None:
    assert _route("RegistryKey declaration in Foo.java") == "exact_symbol"
    assert _route("caller dependency import chain for Foo") == "dependency"
    assert _route("whole project architecture overview") == "global"
    assert _route("Fabric API mapping for Minecraft version 1.20.1") == "exact_version"
    assert _route("how should this mechanic be implemented") == "semantic"


def test_latest_bootstrap_installs_single_selector_without_causal_overlay() -> None:
    assert not getattr(max_agent.select_tool_schemas, "_mmm_causal_tool_frontier", False)
    assert not getattr(model_router.ModelRouter._prepare_generation_request, "_mmm_small_model_tool_retrieval", False)
    assert getattr(model_router.ModelRouter._prepare_generation_request, "_mmm_temporary_verified_skill", False)
    assert callable(getattr(agentic_optimization_contract, "_mmm_active_candidate_discriminator", None))
    assert getattr(work_graph.DurableWorkLedger.succeed, "_mmm_verified_work_trajectory", False)
    assert getattr(work_graph.DurableWorkLedger.fail, "_mmm_failed_work_trajectory", False)
