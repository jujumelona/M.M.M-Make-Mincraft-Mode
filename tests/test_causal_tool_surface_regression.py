from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import GenerationRequest
from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.small_model_hybrid_search_contract import _modes
from minecraft_mod_ai.tool_validation_surface_contract import (
    _validation_surface,
    install as install_tool_validation_surface,
)


def _schema(name: str, properties: dict | None = None) -> dict:
    properties = properties or {}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def test_visible_schema_is_the_authoritative_same_name_schema() -> None:
    install_tool_validation_surface()
    visible = _schema(
        "apply_source_edit",
        {"operation": {"type": "string", "enum": ["replace_exact"]}},
    )
    stale = _schema(
        "apply_source_edit",
        {"operation": {"type": "string", "enum": ["create"]}},
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        tools=(visible,),
        tool_validation_schemas=(stale,),
        tool_choice="auto",
    )
    message = {
        "content": (
            "<tool_call><function=apply_source_edit>"
            "<parameter=operation>replace_exact</parameter>"
            "</function></tool_call>"
        )
    }
    response = llama_cpp_adapter._qwen_tool_generation_response(message, request)
    assert response.tool_calls[0].arguments["operation"] == "replace_exact"


def test_duplicate_validation_owner_fails_closed() -> None:
    first = _schema("apply_source_edit", {"path": {"type": "string"}})
    second = _schema("apply_source_edit", {"operation": {"type": "string"}})
    with pytest.raises(RuntimeError, match="duplicate tool schema name"):
        _validation_surface((), (first, second))


def test_tool_absent_from_selected_surface_is_rejected_by_qwen_parser() -> None:
    install_tool_validation_surface()
    visible = _schema("search_code_rag", {"query": {"type": "string"}})
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        tools=(visible,),
        tool_validation_schemas=(visible,),
        tool_choice="auto",
    )
    message = {
        "content": (
            "<tool_call><function=apply_source_edit>"
            "<parameter=path>src/Main.java</parameter>"
            "</function></tool_call>"
        )
    }
    with pytest.raises(RuntimeError, match="unexposed tool 'apply_source_edit'"):
        llama_cpp_adapter._qwen_tool_generation_response(message, request)


def test_reasoning_continuation_preserves_complete_request_contract() -> None:
    tool = _schema("search_code_rag", {"query": {"type": "string"}})
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        response_format="json",
        response_schema={"type": "object"},
        tools=(tool,),
        tool_validation_schemas=(tool,),
        tool_choice="auto",
        parallel_tool_calls=True,
        task="repair-task",
        prompt="repair-prompt",
        metadata={"trace": "stable-selector"},
    )
    continued = llama_cpp_adapter._reasoning_continuation_request(request, "thinking")
    assert continued.tool_validation_schemas == request.tool_validation_schemas
    assert continued.tools == request.tools
    assert continued.tool_choice == "auto"
    assert continued.task == "repair-task"
    assert continued.prompt == "repair-prompt"
    assert continued.metadata == {"trace": "stable-selector"}


def test_generic_code_rag_dense_escalation_is_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    assert _modes("semantic", False, False) == ((False, False, "lexical"),)
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    modes = _modes("semantic", False, False)
    assert modes[0] == (False, False, "lexical")
    assert modes[1] == (False, True, "lexical+rerank")
    assert modes[2] == (True, True, "semantic+rerank")
