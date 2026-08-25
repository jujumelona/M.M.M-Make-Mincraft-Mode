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
            {"role": "user", "content": '{"phase": "implement_module", "initial_exact_source_context": {"files": {"src/A.java": "public class A { void apply() {} }"}}}'},
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


def test_mutation_ready_requires_concrete_source_or_fresh_evidence() -> None:
    from minecraft_mod_ai.progress_aware_tool_loop import is_mutation_ready

    state = HostRunState()
    # 1. High level hash receipt only without concrete files/source: NOT mutation ready
    abstract_messages = [
        {"role": "user", "content": '{"project_sha256": "abc", "observations_sha256": "def"}'}
    ]
    assert is_mutation_ready(abstract_messages, state) is False

    # 2. Bare file names without code/spans: NOT mutation ready (cannot edit without source)
    bare_file_messages = [
        {
            "role": "user",
            "content": '{"phase": "implement_module", "initial_exact_source_context": {"files": ["Main.java"]}}',
        }
    ]
    assert is_mutation_ready(bare_file_messages, state) is False

    # 3. Dynamic search result returning bare hit without code snippet: NOT mutation ready
    state.record_evidence({"hits": [{"path": "Main.java"}]}, usable=True)
    assert is_mutation_ready(abstract_messages, state) is False

    # 4. Dynamic search result returning code snippet / lines: IS mutation ready
    state.record_evidence(
        {"hits": [{"path": "Main.java", "snippet": "public class Main { public static void init() {} }"}]},
        usable=True,
    )
    assert is_mutation_ready(abstract_messages, state) is True

    # 5. Concrete source context in initial message: IS mutation ready
    concrete_state = HostRunState()
    concrete_messages = [
        {
            "role": "user",
            "content": '{"phase": "implement_module", "initial_exact_source_context": {"files": {"src/Main.java": "package com.example; public class Main {}"}}}',
        }
    ]
    assert is_mutation_ready(concrete_messages, concrete_state) is True

    # 6. Explicit new file creation target: IS mutation ready
    new_file_state = HostRunState()
    new_file_messages = [
        {
            "role": "user",
            "content": '{"phase": "implement_module", "operation": "create_file", "path": "src/NewBlock.java"}',
        }
    ]
    assert is_mutation_ready(new_file_messages, new_file_state) is True


def test_mutation_failure_transitions_to_observe_for_recovery() -> None:
    """When a mutation fails, loop transitions to OBSERVE so agent can inspect diagnostics instead of repeating blindly."""
    router = MagicMock()
    router._generation_scope.return_value = nullcontext()
    router._agent_require_fresh_evidence = False

    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = 16384
    config.max_new_tokens = 4096
    config.extra = {"runtime_contract": "qwen", "qwen_family": "qwen3.5"}

    seen_phases: list[list[str]] = []

    def mock_generate_turn(req: GenerationRequest) -> GenerationResponse:
        tool_names = [t["function"]["name"] for t in req.tools]
        seen_phases.append(tool_names)
        if "apply_source_patch" in tool_names:
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="call_patch",
                        name="apply_source_patch",
                        arguments={"patch": "invalid"},
                        raw_arguments='{"patch":"invalid"}',
                    ),
                )
            )
        else:
            # In OBSERVE phase, model queries workspace
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="call_rag",
                        name="search_code_rag",
                        arguments={"query": "Fix"},
                        raw_arguments='{"query":"Fix"}',
                    ),
                )
            )

    adapter = MagicMock()
    adapter.generate_turn.side_effect = mock_generate_turn

    def mock_runtime_call(stage: str, name: str, args: dict) -> dict:
        if name == "apply_source_patch":
            raise ValueError("Patch rejected: target file not found")
        # Return concrete source snippet so mutation_ready can be satisfied
        return {"hits": [{"path": "Fix.java", "snippet": "public class Fix { void run() {} }"}]}

    runtime = MagicMock()
    runtime.call.side_effect = mock_runtime_call

    request = GenerationRequest(
        messages=(
            {"role": "user", "content": '{"phase": "implement_module", "initial_exact_source_context": {"files": {"src/A.java": "public class A { void apply() {} }"}}}'},
        ),
        tools=(_tool_schema("search_code_rag"), _tool_schema("apply_source_patch")),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    # Initial turn: concrete context -> ACT phase (tools=[apply_source_patch])
    # Patch fails -> transitions to OBSERVE phase (tools=[search_code_rag])
    # Turn 2: OBSERVE phase -> search_code_rag returns concrete snippet -> fresh evidence -> ACT phase
    # Turn 3: ACT phase -> patch fails again -> no progress limit reached
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

    # Verify that after the first failed mutation, tools were opened to OBSERVE (search_code_rag)
    assert len(seen_phases) >= 2
    assert "apply_source_patch" in seen_phases[0]
    assert "search_code_rag" in seen_phases[1]


