from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import (
    complete_orchestrator,
    validation_checkpoint_policy,
    agentic_optimization_contract,
    validation_diagnostic_contract,
    validation_execution_contract,
)
from minecraft_mod_ai.java_lsp import JDTLanguageServerError, JavaLanguageService
from minecraft_mod_ai.trajectory_memory import memory_path, remote_cache_path
from minecraft_mod_ai.validation_diagnostic_contract import (
    diagnostic_errors,
    diagnostic_items,
    run_diagnostics,
)
from tools import colab_runtime_setup


def test_jdt_diagnostics_have_one_static_interpretation_authority() -> None:
    assert validation_execution_contract._diagnostic_errors is diagnostic_errors
    assert complete_orchestrator.JavaLanguageService is JavaLanguageService


def test_jdt_mapping_preserves_uri_and_only_errors_block() -> None:
    receipt = {
        "diagnostics": {
            "file:///B.java": [{"severity": 2, "message": "warning"}],
            "file:///A.java": [{"severity": 1, "message": "compile error"}],
        }
    }
    items = diagnostic_items(receipt)
    assert [item["uri"] for item in items] == ["file:///A.java", "file:///B.java"]
    assert [item["message"] for item in diagnostic_errors(receipt)] == ["compile error"]


def test_malformed_jdt_severity_fails_closed_instead_of_crashing() -> None:
    errors = diagnostic_errors(
        {"diagnostics": [{"severity": "invalid", "message": "bad receipt"}]}
    )
    assert [item["message"] for item in errors] == ["bad receipt"]


def test_malformed_nested_jdt_payload_fails_closed() -> None:
    grouped = diagnostic_errors(
        {"status": "AVAILABLE", "diagnostics": {"file:///A.java": "not-a-list"}}
    )
    listed = diagnostic_errors(
        {"status": "AVAILABLE", "diagnostics": [{"severity": 2}, "bad-item"]}
    )
    assert grouped[-1]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "not a list" in grouped[-1]["message"]
    assert listed[-1]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "non-mapping" in listed[-1]["message"]


def test_jdt_operational_failure_becomes_unavailable_receipt() -> None:
    class Broken:
        def diagnostics(self, *_args, **_kwargs):
            raise JDTLanguageServerError("transport closed")

    receipt = run_diagnostics(Broken, Path("."), timeout_seconds=1)
    assert receipt["status"] == "UNAVAILABLE"
    assert diagnostic_errors(receipt)[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"


def test_jdt_programming_typeerror_propagates() -> None:
    class Buggy:
        def diagnostics(self, *_args, **_kwargs):
            raise TypeError("internal programmer defect")

    with pytest.raises(TypeError, match="programmer defect"):
        run_diagnostics(Buggy, Path("."), timeout_seconds=1)


def test_legacy_jdt_double_without_relative_files_uses_signature_not_exception_probe() -> None:
    class Legacy:
        def diagnostics(self, root, *, timeout_seconds):
            return {"diagnostics": [], "root": str(root), "timeout": timeout_seconds}

    receipt = run_diagnostics(
        Legacy,
        Path("project"),
        relative_files=("src/main/java/A.java",),
        timeout_seconds=7,
    )
    assert receipt["timeout"] == 7


def test_agentic_verifier_does_not_retry_programming_typeerror(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    stage = tmp_path / "stage"
    root.mkdir()
    stage.mkdir()

    from minecraft_mod_ai import performance_final_contract, source_patch

    monkeypatch.setenv("MMM_REPAIR_CANDIDATE_JDT", "on")
    monkeypatch.setattr(
        performance_final_contract,
        "_clone_source_snapshot",
        lambda _root: stage,
    )

    class NoopPatcher:
        def __init__(self, _stage):
            pass

        def apply(self, _operations):
            return None

    monkeypatch.setattr(source_patch, "TransactionalSourcePatcher", NoopPatcher)
    calls = 0

    class Service:
        def diagnostics(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise TypeError("internal programmer defect")

    engine = SimpleNamespace(diagnostics_factory=lambda: Service())
    verifier_impl = inspect.unwrap(agentic_optimization_contract._verify_repair_candidate)
    _score, verifier = verifier_impl(engine, root, [], {})
    assert calls == 1, verifier.get("verifier_error")
    assert verifier["jdt_status"] == "VERIFIER_ERROR"
    assert "programmer defect" in verifier["verifier_error"]



def test_agentic_diagnostics_use_canonical_authority_without_legacy_adapter() -> None:
    source = inspect.getsource(agentic_optimization_contract)
    assert "repair_diagnostics_contract" not in source
    receipt = {
        "status": "AVAILABLE",
        "diagnostics": {
            "file:///src/main/java/A.java": [
                {"severity": 1, "message": "compile error", "code": "E1"}
            ]
        },
    }
    compact = agentic_optimization_contract._compact_evidence(
        {"diagnostics": receipt, "build": {"status": "PASS"}}
    )
    assert compact["diagnostics"] == [
        {
            "path": "file:///src/main/java/A.java",
            "message": "compile error",
            "code": "E1",
            "severity": 1,
        }
    ]
    assert agentic_optimization_contract._diagnostic_paths(
        {"diagnostics": receipt}
    ) == {"file:///src/main/java/A.java", "A.java"}

def test_colab_profile_switch_removes_stale_remote_credentials(monkeypatch) -> None:
    for name in colab_runtime_setup._REMOTE_PROFILE_ENV_NAMES:
        monkeypatch.setenv(name, "stale-secret")
    monkeypatch.setenv("MMM_LLAMA_SERVER_BIN", "/content/llama-server")
    colab_runtime_setup._clear_inactive_profile_environment(local_profile=True)
    assert all(name not in os.environ for name in colab_runtime_setup._REMOTE_PROFILE_ENV_NAMES)
    assert os.environ["MMM_LLAMA_SERVER_BIN"] == "/content/llama-server"


def test_colab_remote_transition_stops_stale_managed_server(monkeypatch) -> None:
    stopped = 0

    def shutdown() -> None:
        nonlocal stopped
        stopped += 1

    module = SimpleNamespace(
        _MANAGED_URL="http://127.0.0.1:8123",
        _shutdown_managed_server=shutdown,
    )
    monkeypatch.setitem(
        sys.modules,
        "minecraft_mod_ai.llama_server_autotune",
        module,
    )
    monkeypatch.setenv("LLAMA_SERVER_URL", module._MANAGED_URL)
    monkeypatch.setenv("MMM_LLAMA_SERVER_BIN", "/content/llama-server")

    colab_runtime_setup._reset_inactive_profile_state(local_profile=False)

    assert stopped == 1
    assert "LLAMA_SERVER_URL" not in os.environ
    assert "MMM_LLAMA_SERVER_BIN" not in os.environ


def test_colab_remote_profile_removes_stale_local_server_routing(monkeypatch) -> None:
    for name in colab_runtime_setup._LOCAL_PROFILE_ENV_NAMES:
        monkeypatch.setenv(name, "stale-local")
    monkeypatch.setenv("MMM_PLANNER_BASE_URL", "https://example.invalid/v1")
    colab_runtime_setup._clear_inactive_profile_environment(local_profile=False)
    assert all(name not in os.environ for name in colab_runtime_setup._LOCAL_PROFILE_ENV_NAMES)
    assert os.environ["MMM_PLANNER_BASE_URL"] == "https://example.invalid/v1"


def test_trajectory_memory_rejects_symlinked_state_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    try:
        (project / ".minecraft_ai").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symbolic links"):
        memory_path(project)
    with pytest.raises(RuntimeError, match="symbolic links"):
        remote_cache_path(project, "repair")


def test_trajectory_memory_rejects_symlinked_leaf_and_remote_cache_dir(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    state = project / ".minecraft_ai" / "trajectory-memory"
    outside = tmp_path / "outside"
    state.mkdir(parents=True)
    outside.mkdir()
    leaf_target = outside / "verified-trajectories.jsonl"
    leaf_target.write_text("sentinel\n", encoding="utf-8")
    try:
        (state / "verified-trajectories.jsonl").symlink_to(leaf_target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(RuntimeError, match="symbolic links"):
        memory_path(project)

    (state / "verified-trajectories.jsonl").unlink()
    (state / "remote-cache").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic links"):
        remote_cache_path(project, "repair")


def test_jdt_cache_fingerprint_tracks_canonical_diagnostic_policy() -> None:
    modules = validation_checkpoint_policy._validation_modules("validate-jdt")
    assert validation_diagnostic_contract in modules
    assert all(module.__name__ != "minecraft_mod_ai.orchestrator_jdt_gate_contract" for module in modules)
