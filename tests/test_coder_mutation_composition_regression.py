from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.model_adapters import GenerationRequest, ModelConfigurationError


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _implement_message() -> dict[str, object]:
    return {
        "role": "user",
        "content": json.dumps(
            {
                "phase": "implement_module",
                "task": "Implement the approved feature in the current project.",
            }
        ),
    }


def _rag_observation() -> dict[str, object]:
    return {
        "role": "tool",
        "name": "search_code_rag",
        "tool_call_id": "rag-1",
        "content": json.dumps(
            {
                "ok": True,
                "result": {
                    "hits": [{"path": "src/main/java/example/Test.java"}],
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                    },
                },
            }
        ),
    }


def _failed_edit(index: int) -> dict[str, object]:
    return {
        "role": "tool",
        "name": "apply_source_edit",
        "tool_call_id": f"edit-{index}",
        "content": json.dumps(
            {
                "ok": False,
                "tool": "apply_source_edit",
                "error": (
                    "AgentToolRuntimeError: exact edit did not match "
                    "[workspace_impact=unchanged]"
                ),
            }
        ),
    }


class _NeverCalledAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_turn(self, request: GenerationRequest):
        self.calls += 1
        raise AssertionError(f"model should not be called again: {request.tool_choice!r}")


def test_live_named_mutation_frontier_cannot_bypass_retry_contract() -> None:
    """Reproduce the live wrapper order that previously leaked to the 12-round guard."""

    rag = _schema("search_code_rag")
    edit = _schema("apply_source_edit")
    surface = (rag, edit)
    inner = _NeverCalledAdapter()
    adapter = CausalFrontierAdapter(
        _WritableProgressAdapter(inner),
        stage="generation",
        role="coder",
        require_fresh_evidence=True,
        authorized_surface=surface,
        preference={},
    )
    request = GenerationRequest(
        messages=(
            _implement_message(),
            _rag_observation(),
            _failed_edit(1),
            _failed_edit(2),
        ),
        tools=surface,
        tool_validation_schemas=surface,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    with pytest.raises(ModelConfigurationError, match="bounded mutation retry budget"):
        adapter.generate_turn(request)

    assert inner.calls == 0


def test_patch_transport_success_without_applied_receipt_consumes_retry() -> None:
    """Completion and retry accounting must use the same mutation-proof contract."""

    patch = _schema("apply_source_patch")
    observations = tuple(
        {
            "role": "tool",
            "name": "apply_source_patch",
            "tool_call_id": f"patch-{index}",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "apply_source_patch",
                    "result": {"status": "PASS"},
                }
            ),
        }
        for index in (1, 2)
    )
    inner = _NeverCalledAdapter()
    adapter = _WritableProgressAdapter(inner)
    request = GenerationRequest(
        messages=(_implement_message(), *observations),
        tools=(patch,),
        tool_validation_schemas=(patch,),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_patch"},
        },
        parallel_tool_calls=False,
    )

    with pytest.raises(ModelConfigurationError, match="bounded mutation retry budget"):
        adapter.generate_turn(request)

    assert inner.calls == 0
