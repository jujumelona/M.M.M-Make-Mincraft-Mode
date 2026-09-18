from __future__ import annotations

from pathlib import Path

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


def test_jdt_readiness_uses_document_symbols_not_hover(tmp_path):
    project, _source = _project(tmp_path)
    notifications = []
    requests = []

    class FakeRpc:
        def notify(self, method, params):
            notifications.append(method)

        def request(self, method, params, timeout):
            requests.append(method)
            assert method == "textDocument/documentSymbol"
            uri = params["textDocument"]["uri"]
            name = Path(uri.removeprefix("file://")).stem
            return [{"name": name, "kind": 5}]

    java_lsp._await_java_core_ready(
        FakeRpc(),
        project,
        timeout_seconds=1.0,
        quiet_seconds=0.0,
    )

    assert notifications == ["textDocument/didOpen", "textDocument/didClose"]
    assert requests == ["textDocument/documentSymbol"]

def test_progress_loop_elides_forced_verifier_model_turn_without_runtime_rebind():
    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai.model_adapters import GenerationRequest

    class FailingAdapter:
        def generate_turn(self, _request):
            raise AssertionError("forced verifier must not invoke the coder model")

    request = GenerationRequest(
        messages=({"role": "user", "content": "task"},),
        tools=(),
        tool_choice={"type": "function", "function": {"name": "java_diagnostics"}},
        parallel_tool_calls=False,
    )
    messages = [{"role": "user", "content": "task"}]
    response = loop._generate_turn_with_context_recovery(
        object(),
        config=SimpleNamespace(),
        adapter=FailingAdapter(),
        request=request,
        messages=messages,
        media_paths=(),
        tool_choice={"type": "function", "function": {"name": "java_diagnostics"}},
        parallel_tool_calls=False,
    )
    assert response.tool_calls[0].name == "java_diagnostics"


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


def test_terminal_coder_summary_is_host_owned_and_fixed_contract():
    import json
    from minecraft_mod_ai import progress_aware_tool_loop as loop

    passed = json.loads(loop._host_coder_summary(verification="PASS"))
    deferred = json.loads(
        loop._host_coder_summary(verification="DEFERRED_TO_TARGET_COMPILE")
    )

    assert set(passed) == {"summary"}
    assert "passed generation-time host verification" in passed["summary"]
    assert set(deferred) == {"summary"}
    assert "target_compile" in deferred["summary"]


def test_terminal_coder_summary_rejects_unknown_host_state():
    import pytest
    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai.model_adapters import ModelConfigurationError

    with pytest.raises(ModelConfigurationError, match="HOST_SUMMARY_STATE_INVALID"):
        loop._host_coder_summary(verification="UNKNOWN")


def _run_compile_backed_generation_flow(
    monkeypatch,
    *,
    compile_results,
    expect_repairs: int = 0,
):
    import json

    from minecraft_mod_ai import progress_aware_tool_loop as loop
    from minecraft_mod_ai import small_model_task_capsule_contract as capsules
    from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse, ToolCall

    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    monkeypatch.setattr(
        capsules,
        "current_task_required_gates",
        lambda: ("target_compile",),
    )

    class MutationAdapter:
        def __init__(self):
            self.calls = 0

        def generate_turn(self, request):
            self.calls += 1
            names = {item["function"]["name"] for item in request.tools}
            assert names == {"apply_source_edit"}
            if self.calls > 1:
                rendered = "\n".join(
                    str(message.get("content") or "") for message in request.messages
                )
                assert "target_compile" in rendered
                assert "package net.minecraft.item does not exist" in rendered
            broken_source = (
                "package dev.mmm.debugfixture; "
                "import net.minecraft.item.Item; "
                "public final class DebugToken {}\n"
            )
            fixed_source = (
                "package dev.mmm.debugfixture; public final class DebugToken {}\n"
            )
            arguments = (
                {
                    "operation": "create_file",
                    "path": target,
                    "content": broken_source,
                }
                if self.calls == 1
                else {
                    "operation": "replace_exact",
                    "path": target,
                    "old": broken_source,
                    "new": fixed_source,
                    "count": 1,
                }
            )
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id=f"edit-{self.calls}",
                        name="apply_source_edit",
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, separators=(",", ":")),
                    ),
                )
            )

    class Runtime:
        def __init__(self):
            self.compile_results = list(compile_results)
            self.compile_calls = 0

        def call(self, stage, name, _arguments):
            assert stage == "generation"
            if name == "apply_source_edit":
                return {
                    "schema_version": "mmm/source-patch-receipt-v1",
                    "status": "APPLIED",
                    "operations": [
                        {
                            "operation": "create" if self.compile_calls == 0 else "replace",
                            "path": target,
                            "before_sha256": None,
                            "after_sha256": "sha256:" + str(self.compile_calls + 1) * 64,
                        }
                    ],
                }
            if name == "target_compile":
                result = self.compile_results[self.compile_calls]
                self.compile_calls += 1
                return result
            raise AssertionError(name)

    request = GenerationRequest(
        messages=(
            {
                "role": "developer",
                "content": json.dumps(
                    {
                        "primary_path": target,
                        "writable_paths": [target],
                        "reuse_action": "fresh",
                    }
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "phase": "implement_module",
                        "task": "Implement the approved debug token.",
                    }
                ),
            },
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "description": "edit source",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    adapter = MutationAdapter()
    runtime = Runtime()
    result = loop.generate_with_tools(
        SimpleNamespace(_agent_require_fresh_evidence=False),
        config=SimpleNamespace(
            adapter="test",
            max_context=32768,
            max_input_tokens=0,
            max_new_tokens=512,
        ),
        adapter=adapter,
        request=request,
        runtime=runtime,
        stage="generation",
        role="coder",
    )
    assert adapter.calls == expect_repairs + 1
    return json.loads(result), runtime.compile_calls


def test_compile_backed_generation_defers_only_when_target_compiler_unavailable(monkeypatch):
    payload, compile_calls = _run_compile_backed_generation_flow(
        monkeypatch,
        compile_results=[
            {
                "schema_version": "mmm/generation-target-compile-v1",
                "status": "UNAVAILABLE",
                "target_path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "diagnostics": [],
                "reason": "compiler unavailable",
            }
        ],
    )

    assert set(payload) == {"summary"}
    assert "target_compile" in payload["summary"]
    assert compile_calls == 1


def test_compile_failure_is_repaired_by_same_coder_before_generation_completes(monkeypatch):
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    payload, compile_calls = _run_compile_backed_generation_flow(
        monkeypatch,
        compile_results=[
            {
                "schema_version": "mmm/generation-target-compile-v1",
                "status": "FAIL",
                "target_path": target,
                "diagnostics": [
                    {
                        "path": target,
                        "severity": 1,
                        "source": "javac",
                        "code": "javac:error:2",
                        "message": "package net.minecraft.item does not exist",
                    }
                ],
                "reason": "target compiler reported task-owned source defects",
            },
            {
                "schema_version": "mmm/generation-target-compile-v1",
                "status": "PASS",
                "target_path": target,
                "diagnostics": [],
                "reason": "target compiler passed",
            },
        ],
        expect_repairs=1,
    )

    assert "passed generation-time host verification" in payload["summary"]
    assert compile_calls == 2


def test_compile_pass_terminates_without_repair_or_jdt_turn(monkeypatch):
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    payload, compile_calls = _run_compile_backed_generation_flow(
        monkeypatch,
        compile_results=[
            {
                "schema_version": "mmm/generation-target-compile-v1",
                "status": "PASS",
                "target_path": target,
                "diagnostics": [],
                "reason": "target compiler passed",
            }
        ],
    )

    assert "passed generation-time host verification" in payload["summary"]
    assert compile_calls == 1

