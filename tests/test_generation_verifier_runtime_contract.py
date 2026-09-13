from __future__ import annotations

import json
from types import SimpleNamespace

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

    verifier = SimpleNamespace(
        synthesized_verifier_turn=base_synthesized_turn,
        host_jdt_idle_timeout_seconds=lambda: 91,
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
