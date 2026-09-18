from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready
from minecraft_mod_ai.host_grounding import _SCHEMA_VERSION as HOST_GROUNDING_SCHEMA
from minecraft_mod_ai.model_adapters import GenerationRequest


TARGET = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _host_grounded_fresh_java_message() -> dict[str, object]:
    return {
        "role": "developer",
        "content": {
            "schema_version": "mmm/small-model-task-capsule",
            "reuse_action": "fresh",
            "mutation_target": {"path": TARGET},
            "primary_path": TARGET,
            "writable_paths": [TARGET],
            "host_grounding": {
                "schema_version": HOST_GROUNDING_SCHEMA,
                "policy": {
                    "resolved_before_first_coder_decode": True,
                    "baseline_grounding_owned_by_host": True,
                    "baseline_grounding_optional_for_model": False,
                    "model_tool_choice_required_for_baseline": False,
                    "writes_still_require_approved_pipeline": True,
                },
                "evidence_bindings": {
                    "project_exact_rag": {
                        "receipt": {
                            "project_sha256": "sha256:project",
                            "observations_sha256": "sha256:observations",
                        }
                    },
                    "approved_research_rag": {
                        "receipt": {
                            "selected_fact_count": 0,
                            "selected_record_count": 0,
                        }
                    },
                },
            },
        },
    }


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_fresh_java_zero_research_facts_does_not_invalidate_host_grounding() -> None:
    message = _host_grounded_fresh_java_message()
    assert host_baseline_evidence_ready((message,)) is True


def test_unresolved_host_grounding_remains_not_ready() -> None:
    message = _host_grounded_fresh_java_message()
    content = message["content"]
    assert isinstance(content, dict)
    grounding = content["host_grounding"]
    assert isinstance(grounding, dict)
    policy = grounding["policy"]
    assert isinstance(policy, dict)
    policy["resolved_before_first_coder_decode"] = False

    assert host_baseline_evidence_ready((message,)) is False


class _StopFirstTurn(RuntimeError):
    pass


class _CapturingAdapter:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate_turn(self, request: GenerationRequest):
        self.requests.append(request)
        raise _StopFirstTurn


def _fresh_owned_anchor_message_without_host_grounding() -> dict[str, object]:
    task_id = "task_debug_token"
    return {
        "role": "user",
        "content": {
            "phase": "implement_module",
            "task": "Implement the approved Minecraft/Fabric mod feature in the current project.",
            "module": {
                "module_id": task_id,
                "kind": "custom_java",
                "config": {
                    "evidence_task": {
                        "task_id": task_id,
                        "owned_anchors": [
                            {
                                "kind": "symbol",
                                "locator": f"{TARGET}#DebugToken",
                                "ownership": "exclusive",
                                "status": "host_reserved",
                                "module_id": ":",
                                "source_set": "main",
                            }
                        ],
                        "production_bindings": [
                            {
                                "task_ref": task_id,
                                "reuse_action": "fresh",
                                "owned_anchors": [
                                    {
                                        "kind": "symbol",
                                        "locator": f"{TARGET}#DebugToken",
                                        "ownership": "exclusive",
                                        "status": "host_reserved",
                                        "module_id": ":",
                                        "source_set": "main",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        },
    }


def test_host_grounded_ready_fresh_target_enters_act_before_any_rag(monkeypatch) -> None:
    adapter = _CapturingAdapter()
    monkeypatch.setattr(loop, "implementation_requested", lambda _messages: True)
    monkeypatch.setattr(loop, "mutation_history_applied", lambda _messages: False)
    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )

    request = GenerationRequest(
        messages=(_host_grounded_fresh_java_message(),),
        tools=(
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("apply_source_edit"),
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    router = SimpleNamespace(_agent_require_fresh_evidence=True)
    runtime = SimpleNamespace()

    with pytest.raises(_StopFirstTurn):
        loop.generate_with_tools(
            router,
            config=SimpleNamespace(),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )

    assert len(adapter.requests) == 1
    first = adapter.requests[0]
    assert [tool["function"]["name"] for tool in first.tools] == ["apply_source_edit"]
    assert first.tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    assert first.parallel_tool_calls is False


def test_exact_owned_fresh_target_enters_act_without_generic_rag(monkeypatch) -> None:
    adapter = _CapturingAdapter()
    monkeypatch.setattr(loop, "implementation_requested", lambda _messages: True)
    monkeypatch.setattr(loop, "mutation_history_applied", lambda _messages: False)
    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )

    request = GenerationRequest(
        messages=(_fresh_owned_anchor_message_without_host_grounding(),),
        tools=(
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("java_workspace_symbols"),
            _tool("external_mcp_call"),
            _tool("apply_source_edit"),
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    router = SimpleNamespace(_agent_require_fresh_evidence=False)
    runtime = SimpleNamespace()

    with pytest.raises(_StopFirstTurn):
        loop.generate_with_tools(
            router,
            config=SimpleNamespace(),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )

    assert len(adapter.requests) == 1
    first = adapter.requests[0]
    assert [tool["function"]["name"] for tool in first.tools] == ["apply_source_edit"]
    assert first.tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    assert first.parallel_tool_calls is False
