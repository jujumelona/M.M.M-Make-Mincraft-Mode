from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from minecraft_mod_ai.model_adapters.base import (
    GenerationRequest,
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.progress_aware_tool_loop import (
    HostRunState,
    generate_with_tools,
)


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }


def test_host_run_state_query_deduplication() -> None:
    state = HostRunState()
    assert state.record_query("search_code_rag", {"query": "BlockRegistry"}) is True
    assert state.record_query("search_code_rag", {"query": "BlockRegistry"}) is False
    assert state.record_query("search_code_rag", {"query": "ItemRegistry"}) is True


def test_host_run_state_evidence_fingerprinting() -> None:
    state = HostRunState()
    ev1 = {"hits": [{"path": "Block.java", "score": 0.9}], "coverage_score": 0.9}
    ev2 = {"hits": [{"path": "Block.java", "score": 0.9}], "coverage_score": 0.5}

    assert state.record_evidence(ev1, usable=True) is True
    assert state.record_evidence(ev2, usable=True) is False


def test_host_run_state_mutation_tracking() -> None:
    state = HostRunState()
    applied_payload = {
        "ok": True,
        "_mmm_source_mutation": {
            "tool": "apply_source_patch",
            "status": "APPLIED_BY_HOST_RUNTIME",
        },
    }
    failed_payload = {"ok": False, "error": "Patch rejected"}

    assert state.record_mutation("apply_source_patch", failed_payload) is False
    assert state.workspace_changed is False

    assert state.record_mutation("apply_source_patch", applied_payload) is True
    assert state.workspace_changed is True
    assert "apply_source_patch" in state.applied_mutations


def test_no_progress_cutoff_at_two_streaks() -> None:
    router = MagicMock()
    router._generation_scope.return_value = nullcontext()
    router._agent_require_fresh_evidence = True

    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = 16384
    config.max_new_tokens = 4096
    config.extra = {"runtime_contract": "qwen", "qwen_family": "qwen3.5"}

    adapter = MagicMock()
    adapter.generate_turn.return_value = GenerationResponse(
        tool_calls=(
            ToolCall(
                id="call_1",
                name="search_code_rag",
                arguments={"query": "BlockRegistry"},
                raw_arguments='{"query":"BlockRegistry"}',
            ),
        )
    )

    runtime = MagicMock()
    runtime.call.return_value = {
        "hits": [{"path": "Block.java"}],
        "receipt": {"result_count": 1, "coverage_score": 0.9, "relevance_score": 0.9},
    }

    request = GenerationRequest(
        messages=({"role": "user", "content": '{"phase": "implement_module", "task": "implement block"}'},),
        tools=(_tool_schema("search_code_rag"), _tool_schema("apply_source_patch")),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    with pytest.raises(ModelConfigurationError, match="no-progress boundary"):
        generate_with_tools(
            router,
            config=config,
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )
