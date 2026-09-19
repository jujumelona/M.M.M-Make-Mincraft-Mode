from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.progress_aware_tool_loop as loop
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
)
from minecraft_mod_ai.model_router import _usable_rag_result


def _fresh_context(*, host_authorized: bool = True) -> loop.TargetMutationContext:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    return loop.TargetMutationContext(
        target_path=target,
        target_symbol="DebugToken",
        is_new_file=True,
        writable_paths=(target,) if host_authorized else (),
        creatable_paths=(target,) if host_authorized else (),
        target_pinned=host_authorized,
        evidence_source="host_task_authority" if host_authorized else "untrusted_observation",
    )


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_metadata_only_project_rag_from_trace_is_not_usable() -> None:
    payload = {
        "parsed_text": None,
        "resources": [],
        "structured_content": {
            "schema_version": "mmm/rag-result-v2",
            "sources": [{"source_id": "provenance-only", "version_scope": "26.2"}],
            "target": {"minecraft_version": "26.2", "loader": "fabric", "mappings": ""},
        },
        "text": [],
    }
    assert _usable_rag_result(payload) is False
    assert loop._authoritative_java_evidence(payload) is False


def test_generic_evidence_cannot_authorize_fresh_java_act() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context(host_authorized=False))
    generic = {
        "schema_version": "other/evidence-v1",
        "content": "general project convention without exact API symbols",
    }
    assert state.record_evidence(generic, usable=True) is True
    assert state.has_fresh_evidence is True
    assert state.has_authoritative_java_evidence is False
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is False


def test_target_neutral_project_rag_never_authorizes_fresh_java() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    payload = {
        "schema_version": "mmm/rag-result-v2",
        "content": "Fabric items are registered during initialization.",
    }
    assert state.record_evidence(payload, usable=True) is True
    assert state.has_authoritative_java_evidence is False


def test_code_rag_with_concrete_minecraft_api_authorizes_fresh_java() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context(host_authorized=False))
    payload = {
        "schema_version": "mmm/code-rag-result-v1",
        "hits": [
            {
                "source_path": "src/main/java/dev/mmm/ExistingItems.java",
                "text": "import net.minecraft.world.item.Item; final class ExistingItems {}",
            }
        ],
        "receipt": {"result_count": 1, "coverage_score": 1.0, "relevance_score": 1.0},
    }
    assert _usable_rag_result(payload) is True
    assert state.record_evidence(payload, usable=True) is True
    assert state.has_authoritative_java_evidence is True
    assert loop._target_evidence_ready(
        state, require_rag=True, fresh_java_target=True
    ) is True


def test_jdt_symbols_authorize_fresh_java_only_when_symbols_exist() -> None:
    state = loop.HostRunState(mutation_context=_fresh_context())
    empty = {"schema_version": "mmm/java-symbols-v1", "symbols": []}
    assert state.record_evidence(empty, usable=True) is True
    assert state.has_authoritative_java_evidence is False

    concrete = {
        "schema_version": "mmm/java-symbols-v1",
        "symbols": [{"name": "Item", "location": {"uri": "file:///workspace/Item.java"}}],
    }
    assert state.record_evidence(concrete, usable=True) is True
    assert state.has_authoritative_java_evidence is True


def test_fresh_java_observe_skips_target_neutral_project_rag() -> None:
    selected = loop._filter_tools_for_phase(
        (
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("external_mcp_call"),
            _tool("java_workspace_symbols"),
        ),
        loop.LoopPhase.OBSERVE,
        "coder",
        mutation_context=_fresh_context(),
        attempted_sources=frozenset(),
        semantic_retrieval_choice=True,
    )
    assert [item["function"]["name"] for item in selected] == ["search_code_rag"]


def test_completion_boundary_gets_one_in_state_recovery(monkeypatch) -> None:
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    class Adapter:
        def __init__(self) -> None:
            self.requests: list[GenerationRequest] = []

        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise LlamaCompletionBoundaryError(
                    "completion token limit",
                    kind=OUTPUT_EXHAUSTED,
                    completion_tokens=512,
                    max_tokens=512,
                )
            return GenerationResponse(content="recovered")

    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )
    adapter = Adapter()
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        tools=(_tool("apply_source_edit"),),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    messages = [dict(message) for message in request.messages]
    result = loop._generate_turn_with_context_recovery(
        SimpleNamespace(),
        config=SimpleNamespace(),
        adapter=adapter,
        request=request,
        messages=messages,
        media_paths=(),
        tool_choice=request.tool_choice,
        parallel_tool_calls=False,
    )
    assert result.content == "recovered"
    assert len(adapter.requests) == 2
    recovery_text = str(adapter.requests[1].messages[-1]["content"])
    assert "MMM_ATOMIC_OUTPUT_RECOVERY_V1" in recovery_text
    assert "one small semantic edit" in recovery_text


def test_completion_boundary_recovery_does_not_loop(monkeypatch) -> None:
    from minecraft_mod_ai.llama_finish_reason_contract import (
        OUTPUT_EXHAUSTED,
        LlamaCompletionBoundaryError,
    )

    class Adapter:
        def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
            raise LlamaCompletionBoundaryError(
                "completion token limit",
                kind=OUTPUT_EXHAUSTED,
                completion_tokens=512,
                max_tokens=512,
            )

    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )
    request = GenerationRequest(
        messages=(
            {"role": "user", "content": "repair"},
            {"role": "system", "content": "MMM_ATOMIC_OUTPUT_RECOVERY_V1 already attempted"},
        ),
        tools=(_tool("apply_source_edit"),),
        tool_choice={"type": "function", "function": {"name": "apply_source_edit"}},
        parallel_tool_calls=False,
    )
    with pytest.raises(ModelConfigurationError, match="ATOMIC_ACTION_OUTPUT_STALLED"):
        loop._generate_turn_with_context_recovery(
            SimpleNamespace(),
            config=SimpleNamespace(),
            adapter=Adapter(),
            request=request,
            messages=[dict(message) for message in request.messages],
            media_paths=(),
            tool_choice=request.tool_choice,
            parallel_tool_calls=False,
        )
