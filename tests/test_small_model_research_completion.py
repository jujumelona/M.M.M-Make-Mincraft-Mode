from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agentic_optimization_contract, model_router, work_graph
from minecraft_mod_ai import small_model_max_agent_contract as max_agent
from minecraft_mod_ai.active_repair_verifier_contract import install as install_active_verifier
from minecraft_mod_ai.remote_skill_store_consent import (
    CONSENT_ENV,
    remote_write_allowed,
    require_remote_write_consent,
    sanitize_remote_payload,
)
from minecraft_mod_ai.remote_trajectory_store import queue_remote_record
from minecraft_mod_ai.small_model_hybrid_search_contract import _route
from minecraft_mod_ai.trajectory_memory import (
    append_trajectory,
    build_work_trajectory,
    relevant_trajectories,
    synthesize_temporary_skill,
)


def test_colab_remote_trajectory_storage_is_explicit_opt_in() -> None:
    notebook = json.loads(Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        str(cell.get("source", ""))
        for cell in notebook.get("cells", [])
    )
    assert "ALLOW_REMOTE_TRAJECTORY_STORE = False" in source
    assert "MMM_REMOTE_TRAJECTORY_STORE_CONSENT" in source
    assert '"1" if ALLOW_REMOTE_TRAJECTORY_STORE else "0"' in source


def test_remote_write_gate_defaults_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CONSENT_ENV, raising=False)
    assert not remote_write_allowed()
    with pytest.raises(PermissionError):
        require_remote_write_consent()
    row = {
        "trajectory_id": "sha256:" + "a" * 64,
        "task_class": "repair",
        "outcome": "SUCCESS",
    }
    assert queue_remote_record(tmp_path, row) is False
    assert not (
        tmp_path
        / ".minecraft_ai"
        / "trajectory-memory"
        / "remote-outbox.jsonl"
    ).exists()


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


def test_trajectory_is_structural_and_temporary_skill_uses_verified_patterns(
    tmp_path: Path,
) -> None:
    task = {
        "node_id": "module-demo",
        "stage": "repair",
        "payload": {
            "kind": "custom_java",
            "source_body": "public class SecretSource {}",
            "members": [{"module_id": "demo"}],
        },
    }
    success = build_work_trajectory(
        task,
        outcome="SUCCESS",
        receipt={
            "status": "PASS",
            "jdt_status": "PASS",
            "source_body": "do not persist source",
            "token": "do not persist token",
        },
    )
    failure = build_work_trajectory(
        task,
        outcome="FAIL",
        receipt={"status": "FAIL", "jdt_status": "PASS"},
        error="cannot resolve symbol RegistryKey",
    )
    rendered = json.dumps(success, ensure_ascii=False)
    assert "SecretSource" not in rendered
    assert "do not persist source" not in rendered
    assert "do not persist token" not in rendered
    assert append_trajectory(tmp_path, success)
    assert append_trajectory(tmp_path, failure)

    records = relevant_trajectories(
        tmp_path,
        "repair custom_java RegistryKey",
        task_class="repair",
        router=None,
        limit=6,
    )
    assert {row["outcome"] for row in records} == {"SUCCESS", "FAIL"}
    skill = synthesize_temporary_skill(
        "repair custom_java RegistryKey",
        records,
        task_class="repair",
    )
    assert skill is not None
    assert skill["ephemeral"] is True
    assert skill["proven_patterns"]
    assert skill["avoid_patterns"]
    assert len(skill["source_trajectory_ids"]) == 2


def test_code_retrieval_routes_by_task_shape() -> None:
    assert _route("RegistryKey declaration in Foo.java") == "exact_symbol"
    assert _route("caller dependency import chain for Foo") == "dependency"
    assert _route("whole project architecture overview") == "global"
    assert _route("Fabric API mapping for Minecraft version 1.20.1") == "exact_version"
    assert _route("how should this mechanic be implemented") == "semantic"


def test_active_verifier_only_escalates_high_risk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def base(_self, _root, _operations, _evidence):
        return 100.0, {"jdt_status": "PASS", "jdt_error_count": 0}

    owner = SimpleNamespace(_verify_repair_candidate=base)
    install_active_verifier(owner)

    low_score, low = owner._verify_repair_candidate(
        object(),
        tmp_path,
        [{"operation": "write", "path": "src/main/java/demo/Foo.java"}],
        {},
    )
    assert low_score == 100.0
    assert low["active_verifier"] == "NOT_NEEDED"

    monkeypatch.setattr(
        "minecraft_mod_ai.active_repair_verifier_contract._active_build",
        lambda _self, _root, _ops: {"active_build_status": "PASS"},
    )
    high_score, high = owner._verify_repair_candidate(
        object(),
        tmp_path,
        [
            {"operation": "write", "path": "build.gradle"},
            {"operation": "write", "path": "src/main/java/demo/NetworkHandler.java"},
            {"operation": "write", "path": "src/main/java/demo/WorldHooks.java"},
        ],
        {},
    )
    assert high_score == 750.0
    assert high["active_verifier"] == "EXECUTED"
    assert high["active_build_status"] == "PASS"


def test_latest_bootstrap_installs_all_small_agent_completion_layers() -> None:
    assert getattr(max_agent.select_tool_schemas, "_mmm_causal_tool_frontier", False)
    assert getattr(
        model_router.ModelRouter._prepare_generation_request,
        "_mmm_temporary_verified_skill",
        False,
    )
    assert getattr(
        agentic_optimization_contract._verify_repair_candidate,
        "_mmm_uncertainty_active_verifier",
        False,
    )
    assert getattr(
        work_graph.DurableWorkLedger.succeed,
        "_mmm_verified_work_trajectory",
        False,
    )
    assert getattr(
        work_graph.DurableWorkLedger.fail,
        "_mmm_failed_work_trajectory",
        False,
    )
