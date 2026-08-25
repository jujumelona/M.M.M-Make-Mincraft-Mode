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


def test_filter_tools_for_phase_is_strictly_fail_closed() -> None:
    from minecraft_mod_ai.progress_aware_tool_loop import (
        LoopPhase,
        _filter_tools_for_phase,
    )

    read_tool = _tool_schema("search_code_rag")
    mutate_tool = _tool_schema("apply_source_patch")
    verify_tool = _tool_schema("java_diagnostics")

    # In ACT phase with only read tools available, must return empty tuple fail-closed, NOT all tools
    act_tools = _filter_tools_for_phase((read_tool,), LoopPhase.ACT, role="coder")
    assert act_tools == ()

    # In OBSERVE phase with only mutation tools available, must return empty tuple fail-closed
    obs_tools = _filter_tools_for_phase((mutate_tool,), LoopPhase.OBSERVE, role="coder")
    assert obs_tools == ()

    # In VERIFY phase with only mutation tools available, must return empty tuple fail-closed
    ver_tools = _filter_tools_for_phase((mutate_tool,), LoopPhase.VERIFY, role="coder")
    assert ver_tools == ()

    # When matching tools exist, returns only those matching tools preserving exposure order
    mixed = (read_tool, mutate_tool, verify_tool)
    assert _filter_tools_for_phase(mixed, LoopPhase.OBSERVE, role="coder") == (read_tool,)
    assert _filter_tools_for_phase(mixed, LoopPhase.ACT, role="coder") == (mutate_tool,)
    assert _filter_tools_for_phase(mixed, LoopPhase.VERIFY, role="coder") == (read_tool, verify_tool)


def test_out_of_phase_tool_call_is_rejected_fail_closed() -> None:
    """When a model emits an out-of-phase tool (e.g. search_code_rag in ACT phase), host rejects execution."""
    router = MagicMock()
    router._generation_scope.return_value = nullcontext()
    router._agent_require_fresh_evidence = False

    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = 16384
    config.max_new_tokens = 4096
    config.extra = {"runtime_contract": "qwen", "qwen_family": "qwen3.5"}

    # We start directly in ACT phase by requesting implementation on a grounded message
    # Model emits out-of-phase tool: search_code_rag instead of apply_source_patch
    adapter = MagicMock()
    adapter.generate_turn.return_value = GenerationResponse(
        tool_calls=(
            ToolCall(
                id="call_1",
                name="search_code_rag",
                arguments={"query": "test"},
                raw_arguments='{"query":"test"}',
            ),
        )
    )

    runtime = MagicMock()

    request = GenerationRequest(
        messages=(
            {"role": "system", "content": "grounded context"},
            {"role": "user", "content": '{"phase": "implement_module", "initial_exact_source_context": "class A {}"}'},
        ),
        tools=(_tool_schema("search_code_rag"), _tool_schema("apply_source_patch")),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    # The tool loop should reject search_code_rag because ACT phase only allows apply_source_patch
    # Since search_code_rag was rejected and no progress was made across 2 streaks, it raises no-progress boundary
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

    # Runtime call was never executed for the out-of-phase tool!
    runtime.call.assert_not_called()

