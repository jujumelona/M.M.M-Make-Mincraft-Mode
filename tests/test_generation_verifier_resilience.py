from __future__ import annotations

import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

from minecraft_mod_ai import agent_tool_runtime
from minecraft_mod_ai.generation_verifier_resilience import (
    _collect_diagnostics_progress_aware,
    gradle_fallback_receipt,
    install,
    run_generation_verifier,
    synthesized_verifier_turn,
)
from minecraft_mod_ai.java_lsp import JDTLanguageServerError
from minecraft_mod_ai.progress_aware_tool_loop import _verification_outcome


def test_synthesized_verifier_turn_is_host_owned(monkeypatch):
    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS", "77")
    turn = synthesized_verifier_turn([{"role": "user", "content": "task"}])

    assert turn.content == ""
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "java_diagnostics"
    assert dict(call.arguments) == {"timeout_seconds": 77}
    assert call.id.startswith("host_verify_")


def test_install_elides_forced_verifier_model_turn(monkeypatch):
    class DummyRuntime:
        def _call(self, stage, name, arguments, *, external_server_ids):
            return {"delegated": True}

        @staticmethod
        def _stage(stage):
            return stage

    model_calls = []

    def original_turn(*args, **kwargs):
        model_calls.append((args, kwargs))
        raise AssertionError("forced verifier must not invoke the coder model")

    fake_runtime_module = SimpleNamespace(
        AgentToolRuntime=DummyRuntime,
        _discover_model_project_root=agent_tool_runtime._discover_model_project_root,
        _bounded_result=agent_tool_runtime._bounded_result,
        AgentToolRuntimeError=agent_tool_runtime.AgentToolRuntimeError,
    )
    fake_progress_module = SimpleNamespace(
        _generate_turn_with_context_recovery=original_turn,
    )
    fake_java_module = SimpleNamespace(
        _collect_diagnostics_traced=lambda *args, **kwargs: {},
    )

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

    assert model_calls == []
    assert response.tool_calls[0].name == "java_diagnostics"


def test_gradle_fallback_failure_is_source_failure_not_verifier_unavailable(tmp_path):
    log = tmp_path / "gradle-build.log"
    log.write_text("error: cannot find symbol\n", encoding="utf-8")
    report = SimpleNamespace(
        passed=False,
        commands=(SimpleNamespace(log_path=str(log)),),
        to_dict=lambda: {
            "status": "FAIL",
            "commands": [{"log_path": str(log), "exit_code": 1}],
            "error": "Gradle build failed.",
        },
    )

    receipt = gradle_fallback_receipt(
        report,
        JDTLanguageServerError("JDT transport unavailable"),
    )
    outcome = _verification_outcome(
        "java_diagnostics",
        {"ok": True, "result": receipt},
    )

    assert receipt["status"] == "OK"
    assert receipt["verification_outcome"] == "FAIL"
    assert receipt["diagnostics"]["gradle://build"][0]["severity"] == 1
    assert "cannot find symbol" in receipt["diagnostics"]["gradle://build"][0]["message"]
    assert outcome == "FAIL"


def test_generation_verifier_ignores_model_timeout_and_falls_back(monkeypatch, tmp_path):
    project = tmp_path / "run" / "project"
    (project / "src" / "main" / "java").mkdir(parents=True)
    (project / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    monkeypatch.setenv("MMM_JDT_DIAGNOSTIC_IDLE_TIMEOUT_SECONDS", "83")

    class FailingJava:
        def diagnostics(self, root, *, relative_files=None, timeout_seconds=0):
            assert root == project
            assert timeout_seconds == 83
            assert relative_files == ["src/main/java/Example.java"]
            raise JDTLanguageServerError("cold import unavailable")

    class PassingReport:
        passed = True
        commands = ()

        @staticmethod
        def to_dict():
            return {"status": "PASS", "commands": [], "error": None}

    class FakeGradleRunner:
        def build(self, root, *, run_gametest):
            assert root == project
            assert run_gametest is False
            return PassingReport()

    runtime = SimpleNamespace(
        workspace_root=str(project),
    )
    result = run_generation_verifier(
        runtime,
        {
            "timeout_seconds": 1,
            "relative_files": ["./src/main/java/Example.java"],
        },
        runtime_module=agent_tool_runtime,
        java_service_factory=FailingJava,
        gradle_runner_factory=lambda _cache: FakeGradleRunner(),
    )

    assert result["verification_backend"] == "gradle_build_fallback"
    assert result["verification_outcome"] == "PASS"
    assert result["diagnostics"] == {}
    assert result["_mmm_observation"]["truncated"] is False


def test_jdt_progress_refreshes_idle_deadline():
    expected_uri = "file:///workspace/Example.java"

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    class Reader:
        @staticmethod
        def is_alive():
            return True

    rpc = SimpleNamespace(
        messages=queue.Queue(),
        process=Process(),
        stderr=deque(),
        _reader=Reader(),
        reader_failure=None,
        stdout_eof=False,
        protocol_counts={},
        server_progress_tail=deque(maxlen=30),
    )

    def producer():
        # Total elapsed time exceeds the initial 0.10 s idle deadline, but valid JDT
        # progress arrives before each idle window expires.
        time.sleep(0.06)
        rpc.messages.put({"method": "$/progress", "params": {"value": {"kind": "report"}}})
        time.sleep(0.06)
        rpc.messages.put(
            {
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": expected_uri, "diagnostics": []},
            }
        )

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    result = _collect_diagnostics_progress_aware(
        rpc,
        expected_uris={expected_uri},
        timeout_seconds=0.10,
        quiet_seconds=0.01,
        page_index=0,
    )
    thread.join(timeout=1.0)

    assert result == {expected_uri: []}


def test_non_verifier_turn_still_delegates():
    calls = []

    class DummyRuntime:
        def _call(self, stage, name, arguments, *, external_server_ids):
            calls.append((stage, name, arguments, external_server_ids))
            return {"ok": True}

        @staticmethod
        def _stage(stage):
            return stage

    def original_turn(*args, **kwargs):
        return "delegated"

    fake_runtime_module = SimpleNamespace(
        AgentToolRuntime=DummyRuntime,
        _discover_model_project_root=agent_tool_runtime._discover_model_project_root,
        _bounded_result=agent_tool_runtime._bounded_result,
        AgentToolRuntimeError=agent_tool_runtime.AgentToolRuntimeError,
    )
    fake_progress_module = SimpleNamespace(
        _generate_turn_with_context_recovery=original_turn,
    )
    fake_java_module = SimpleNamespace(
        _collect_diagnostics_traced=lambda *args, **kwargs: {},
    )

    install(
        agent_tool_runtime_module=fake_runtime_module,
        progress_loop_module=fake_progress_module,
        java_lsp_trace_module=fake_java_module,
    )

    assert fake_progress_module._generate_turn_with_context_recovery(
        object(),
        config=object(),
        adapter=object(),
        request=object(),
        messages=[],
        media_paths=(),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    ) == "delegated"
