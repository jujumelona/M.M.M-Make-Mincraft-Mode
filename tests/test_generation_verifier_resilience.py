from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import agent_tool_runtime, java_lsp
from minecraft_mod_ai.generation_verifier_resilience import (
    install,
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


def test_generation_verifier_preserves_completed_owner_diagnostics(tmp_path):
    project, _source = _project(tmp_path)
    calls = []

    class JavaOwner:
        def diagnostics(self, root, *, timeout_seconds, full_scan=False):
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


def test_generation_verifier_rejects_unbound_owner_result(tmp_path):
    project, _source = _project(tmp_path)
    closed = []

    class IncompleteOwner:
        def diagnostics(self, root, *, timeout_seconds, full_scan=False):
            return {"error_count": 0, "diagnostics": {}}

        def close(self):
            closed.append(True)

    runtime = SimpleNamespace(workspace_root=str(project))
    with pytest.raises(agent_tool_runtime.AgentToolRuntimeError, match="unbound"):
        run_generation_verifier(runtime, {}, runtime_module=agent_tool_runtime,
                                java_service_factory=IncompleteOwner)
    assert closed == [True]


def test_jdt_failure_fails_closed_when_no_verification_backend_is_healthy(tmp_path):
    project, _source = _project(tmp_path)
    closed = []

    class FailingJava:
        def diagnostics(self, root, *, relative_files=None, timeout_seconds=0, full_scan=False):
            raise JDTLanguageServerError("workspace import failed")

        def close(self):
            closed.append(True)

    runtime = SimpleNamespace(workspace_root=str(project))
    with pytest.raises(
        agent_tool_runtime.AgentToolRuntimeError,
        match="Generation verification has no healthy backend",
    ):
        run_generation_verifier(
            runtime,
            {},
            runtime_module=agent_tool_runtime,
            java_service_factory=FailingJava,
        )

    assert closed == [True]
    assert not hasattr(runtime, "_mmm_generation_java_service")
    assert not hasattr(runtime, "_mmm_generation_jdt_disabled_reason")


def test_jdt_readiness_uses_diagnostics_and_never_hover(monkeypatch, tmp_path):
    project, _source = _project(tmp_path)
    notifications = []

    class FakeRpc:
        def notify(self, method, params):
            notifications.append(method)

        def request(self, method, params, timeout):
            raise AssertionError(f"readiness must not call {method}")

    def diagnostics(
        _rpc, *, expected_uris, timeout_seconds, quiet_seconds, deadline=None
    ):
        return {next(iter(expected_uris)): []}

    monkeypatch.setattr(java_lsp, "_collect_diagnostics", diagnostics)
    java_lsp._await_java_core_ready(
        FakeRpc(),
        project,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert notifications == ["textDocument/didOpen", "textDocument/didClose"]


def test_install_elides_forced_verifier_model_turn():
    class DummyRuntime:
        def _call(self, stage, name, arguments, *, external_server_ids):
            return {"delegated": True}

        @staticmethod
        def _stage(stage):
            return stage

    def original_turn(*args, **kwargs):
        raise AssertionError("forced verifier must not invoke the coder model")

    fake_runtime_module = SimpleNamespace(
        AgentToolRuntime=DummyRuntime,
        _discover_model_project_root=agent_tool_runtime._discover_model_project_root,
        _bounded_result=agent_tool_runtime._bounded_result,
        AgentToolRuntimeError=agent_tool_runtime.AgentToolRuntimeError,
    )
    fake_progress_module = SimpleNamespace(_generate_turn_with_context_recovery=original_turn)
    fake_java_module = SimpleNamespace(_collect_diagnostics_traced=lambda *args, **kwargs: {})

    install(
        agent_tool_runtime_module=fake_runtime_module,
        progress_loop_module=fake_progress_module,
        java_lsp_trace_module=fake_java_module,
    )
    response = fake_progress_module._generate_turn_with_context_recovery(
        object(),
        config=object(),
        adapter=object(),
        request=object(),
        messages=[{"role": "user", "content": "task"}],
        media_paths=(),
        tool_choice={"type": "function", "function": {"name": "java_diagnostics"}},
        parallel_tool_calls=False,
    )
    assert response.tool_calls[0].name == "java_diagnostics"
