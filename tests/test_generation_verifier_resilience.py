from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import agent_tool_runtime, java_lsp
from minecraft_mod_ai.generation_verifier_resilience import (
    host_jdt_startup_timeout_seconds,
    run_generation_verifier,
    synthesized_verifier_turn,
)
from minecraft_mod_ai.java_lsp import JDTLanguageServerError


def _project(tmp_path):
    root = tmp_path / "project"
    source = root / "src" / "main" / "java" / "Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("final class Example {}\n", encoding="utf-8")
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    return root, source


def _ok_result(root):
    return {
        "schema_version": "mmm/java-diagnostics-v2",
        "project_root": str(root),
        "files_opened": 1,
        "error_count": 0,
        "warning_count": 0,
        "diagnostics": {},
    }


def test_synthesized_verifier_turn_is_host_owned(monkeypatch):
    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS", "77")
    turn = synthesized_verifier_turn([{"role": "user", "content": "task"}])
    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "java_diagnostics"
    assert dict(call.arguments) == {"timeout_seconds": 77}
    assert call.id.startswith("host_verify_")


def test_generation_verifier_uses_bootstrap_budget_only_for_cold_owner(
    monkeypatch, tmp_path
):
    project, _source = _project(tmp_path)
    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_STARTUP_TIMEOUT_SECONDS", "240")
    calls = []

    class JavaOwner:
        def diagnostics(
            self, root, *, relative_files=None, timeout_seconds, full_scan=False
        ):
            calls.append(timeout_seconds)
            return {
                "complete": True,
                "session_id": "owner",
                "model_id": "model",
                "error_count": 0,
                "warning_count": 0,
                "diagnostics": {},
            }

    runtime = SimpleNamespace(workspace_root=str(project))
    for _ in range(2):
        result = run_generation_verifier(
            runtime,
            {"timeout_seconds": 77},
            runtime_module=agent_tool_runtime,
            java_service_factory=JavaOwner,
        )
        assert result["error_count"] == 0

    assert calls == [240.0, 77.0]


def test_generation_verifier_bootstrap_budget_is_bounded(monkeypatch):
    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_STARTUP_TIMEOUT_SECONDS", "30")
    assert host_jdt_startup_timeout_seconds() == 90

    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_STARTUP_TIMEOUT_SECONDS", "999")
    assert host_jdt_startup_timeout_seconds() == 600


def test_generation_verifier_preserves_completed_owner_diagnostics(tmp_path):
    project, _source = _project(tmp_path)
    calls = []

    class JavaOwner:
        def diagnostics(self, root, *, relative_files=None, timeout_seconds, full_scan=False):
            calls.append(full_scan)
            return {"complete": True, "session_id": "owner", "model_id": "model",
                    "error_count": 1, "warning_count": 0,
                    "diagnostics": {"B.java": [{"severity": 1, "message": "A.method unresolved"}]}}

    runtime = SimpleNamespace(workspace_root=str(project))
    for _ in range(2):
        result = run_generation_verifier(runtime, {}, runtime_module=agent_tool_runtime,
                                         java_service_factory=JavaOwner)
        assert result["error_count"] == 1
        assert result["diagnostics"]["B.java"][0]["message"] == "A.method unresolved"
    assert calls == [False, False]


def test_generation_verifier_reports_unbound_owner_as_unavailable(tmp_path):
    project, _source = _project(tmp_path)
    closed = []

    class IncompleteOwner:
        def diagnostics(self, root, *, relative_files=None, timeout_seconds, full_scan=False):
            return {"error_count": 0, "diagnostics": {}}

        def close(self):
            closed.append(True)

    runtime = SimpleNamespace(workspace_root=str(project))
    result = run_generation_verifier(
        runtime,
        {},
        runtime_module=agent_tool_runtime,
        java_service_factory=IncompleteOwner,
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["available"] is False
    assert result["verification_backend"] == "jdt_core"
    assert result["diagnostics"][0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "unbound" in result["diagnostics"][0]["message"]
    assert closed == [True]


def test_jdt_failure_returns_structured_unavailable_without_gradle_fallback(tmp_path):
    project, _source = _project(tmp_path)
    closed = []

    class FailingJava:
        def diagnostics(self, root, *, relative_files=None, timeout_seconds=0, full_scan=False):
            raise JDTLanguageServerError("workspace import failed")

        def close(self):
            closed.append(True)

    runtime = SimpleNamespace(workspace_root=str(project))
    result = run_generation_verifier(
        runtime,
        {},
        runtime_module=agent_tool_runtime,
        java_service_factory=FailingJava,
    )

    assert result["status"] == "UNAVAILABLE"
    assert result["available"] is False
    assert result["verification_backend"] == "jdt_core"
    assert result["diagnostics"][0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "workspace import failed" in result["diagnostics"][0]["message"]
    assert closed == [True]
    assert not hasattr(runtime, "_mmm_generation_java_service")
    assert not hasattr(runtime, "_mmm_generation_jdt_disabled_reason")


def test_jdt_readiness_waits_for_explicit_service_ready_status(tmp_path):
    import queue

    project, _source = _project(tmp_path)

    class FakeRpc:
        def __init__(self):
            self.messages = queue.Queue()
            self.messages.put({
                "method": "language/status",
                "params": {
                    "type": "ServiceReady",
                    "message": "workspace initialized",
                },
            })

    java_lsp._await_java_core_ready(
        FakeRpc(),
        project,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

def test_agent_runtime_owns_generation_verifier_dispatch_directly(monkeypatch, tmp_path):
    from minecraft_mod_ai import generation_verifier_resilience

    (tmp_path / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (tmp_path / "src" / "main" / "java").mkdir(parents=True)
    runtime = agent_tool_runtime.AgentToolRuntime(
        profile="test",
        workspace_root=tmp_path,
    )
    calls = []

    def fake_run(runtime_obj, arguments, *, runtime_module, java_service_factory=None):
        del java_service_factory
        calls.append((runtime_obj, dict(arguments), runtime_module))
        return {
            "schema_version": "mmm/java-diagnostics-v3",
            "status": "PASS",
            "complete": True,
            "error_count": 0,
            "warning_count": 0,
            "diagnostics": {},
        }

    monkeypatch.setattr(
        generation_verifier_resilience,
        "run_generation_verifier",
        fake_run,
    )
    result = runtime.call("generation", "java_diagnostics", {})

    assert result["status"] == "PASS"
    assert calls and calls[0][0] is runtime
    assert calls[0][2] is agent_tool_runtime


def test_generation_verifier_watchdog_aborts_hung_owner(monkeypatch, tmp_path):
    import threading
    import time

    from minecraft_mod_ai import generation_verifier_resilience as resilience

    project, _source = _project(tmp_path)
    aborted = threading.Event()

    class HangingJava:
        def diagnostics(
            self,
            root,
            *,
            relative_files=None,
            timeout_seconds=0,
            full_scan=False,
        ):
            del root, relative_files, timeout_seconds, full_scan
            while not aborted.wait(0.01):
                pass
            raise TimeoutError("owner aborted by watchdog")

        def abort(self):
            aborted.set()

    monkeypatch.setattr(
        resilience,
        "host_jdt_startup_timeout_seconds",
        lambda: 0.05,
    )
    monkeypatch.setattr(
        resilience,
        "_watchdog_grace_seconds",
        lambda: 0.01,
    )

    runtime = SimpleNamespace(workspace_root=str(project))
    started = time.monotonic()
    result = run_generation_verifier(
        runtime,
        {"timeout_seconds": 0.02},
        runtime_module=agent_tool_runtime,
        java_service_factory=HangingJava,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert aborted.is_set()
    assert result["status"] == "UNAVAILABLE"
    assert result["available"] is False
    assert result["diagnostics"][0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "wall-clock deadline exceeded" in result["diagnostics"][0]["message"]
    assert not hasattr(runtime, "_mmm_generation_java_service")
