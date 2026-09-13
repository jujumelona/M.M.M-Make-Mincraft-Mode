from __future__ import annotations

import json
from types import SimpleNamespace

import minecraft_mod_ai.generation_verifier_runtime_contract as verifier_runtime_contract
from minecraft_mod_ai.generation_verifier_runtime_contract import (
    install,
    latest_mutated_source_files,
)


def _assistant_call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def test_latest_mutated_source_files_targets_most_recent_source_mutation() -> None:
    messages = [
        _assistant_call(
            "apply_source_edit",
            {"path": "src/main/java/Old.java", "operation": "create_file"},
        ),
        {"role": "tool", "content": "{}"},
        _assistant_call(
            "apply_source_edit",
            {"path": "./src/main/java/New.java", "operation": "replace"},
        ),
        {"role": "tool", "content": "{}"},
    ]

    assert latest_mutated_source_files(messages) == ("src/main/java/New.java",)


def test_latest_mutated_source_files_does_not_treat_resource_edit_as_java() -> None:
    messages = [
        _assistant_call(
            "apply_source_edit",
            {"path": "src/main/resources/assets/demo/item.json", "operation": "replace"},
        )
    ]

    assert latest_mutated_source_files(messages) == ()


def test_install_targets_synthesized_verifier_and_closes_service() -> None:
    closed: list[str] = []

    class Runtime:
        def close(self) -> None:
            closed.append("runtime")

    class Service:
        def close(self) -> None:
            closed.append("jdt")

    def base_synthesized_turn(messages):
        raise AssertionError("runtime contract must own synthesized verifier targeting")

    def base_fallback(*args, **kwargs):
        return {"verification_backend": "injected"}

    verifier = SimpleNamespace(
        synthesized_verifier_turn=base_synthesized_turn,
        host_jdt_idle_timeout_seconds=lambda: 91,
        _run_gradle_fallback=base_fallback,
    )
    runtime_module = SimpleNamespace(AgentToolRuntime=Runtime)

    install(
        agent_tool_runtime_module=runtime_module,
        verifier_module=verifier,
    )

    turn = verifier.synthesized_verifier_turn(
        [
            _assistant_call(
                "apply_source_edit",
                {"path": "src/main/java/Example.java", "operation": "replace"},
            ),
            {"role": "tool", "content": "{}"},
        ]
    )
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "java_diagnostics"
    assert dict(call.arguments) == {
        "timeout_seconds": 91,
        "relative_files": ["src/main/java/Example.java"],
    }

    runtime = Runtime()
    runtime._mmm_generation_java_service = Service()
    runtime.close()
    assert closed == ["jdt", "runtime"]
    assert not hasattr(runtime, "_mmm_generation_java_service")


def test_install_routes_default_fallback_through_compile_only_adapter(monkeypatch, tmp_path) -> None:
    factories = []

    class Runtime:
        def close(self) -> None:
            return None

    class FakeCompileAdapter:
        def __init__(self, cache):
            factories.append(cache)

        def build(self, project_root, *, run_gametest):
            assert run_gametest is False
            return SimpleNamespace(project_root=project_root)

    def base_fallback(
        runtime,
        project_root,
        jdt_error,
        *,
        runtime_module,
        gradle_runner_factory,
    ):
        runner = gradle_runner_factory(tmp_path / "cache")
        runner.build(project_root, run_gametest=False)
        return {
            "verification_backend": "gradle_build_fallback",
            "diagnostics": {},
        }

    monkeypatch.setattr(
        verifier_runtime_contract,
        "_CompileOnlyFallbackAdapter",
        FakeCompileAdapter,
    )
    verifier = SimpleNamespace(
        synthesized_verifier_turn=lambda messages: None,
        host_jdt_idle_timeout_seconds=lambda: 90,
        _run_gradle_fallback=base_fallback,
    )
    runtime_module = SimpleNamespace(AgentToolRuntime=Runtime)
    install(
        agent_tool_runtime_module=runtime_module,
        verifier_module=verifier,
    )

    result = verifier._run_gradle_fallback(
        SimpleNamespace(),
        tmp_path / "project",
        RuntimeError("jdt unavailable"),
        runtime_module=SimpleNamespace(),
        gradle_runner_factory=None,
    )

    assert factories == [tmp_path / "cache"]
    assert result["verification_backend"] == "gradle_compile_fallback"


def test_install_preserves_explicit_fallback_factory() -> None:
    sentinel_factory = object()
    observed = []

    class Runtime:
        def close(self) -> None:
            return None

    def base_fallback(
        runtime,
        project_root,
        jdt_error,
        *,
        runtime_module,
        gradle_runner_factory,
    ):
        observed.append(gradle_runner_factory)
        return {"verification_backend": "test_backend"}

    verifier = SimpleNamespace(
        synthesized_verifier_turn=lambda messages: None,
        host_jdt_idle_timeout_seconds=lambda: 90,
        _run_gradle_fallback=base_fallback,
    )
    install(
        agent_tool_runtime_module=SimpleNamespace(AgentToolRuntime=Runtime),
        verifier_module=verifier,
    )

    result = verifier._run_gradle_fallback(
        SimpleNamespace(),
        SimpleNamespace(),
        RuntimeError("jdt unavailable"),
        runtime_module=SimpleNamespace(),
        gradle_runner_factory=sentinel_factory,
    )

    assert observed == [sentinel_factory]
    assert result["verification_backend"] == "test_backend"
