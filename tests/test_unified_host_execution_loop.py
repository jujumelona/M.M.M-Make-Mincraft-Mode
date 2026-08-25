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


def test_hierarchical_localization_file_symbol_function_body() -> None:
    """Agentless & AutoCodeRover style: File -> Symbol -> Function Body -> Mutation Ready."""
    from minecraft_mod_ai.progress_aware_tool_loop import is_mutation_ready, TargetMutationContext

    state = HostRunState()
    base_messages = [{"role": "user", "content": '{"phase": "implement_module", "task": "fix drop logic"}'}]

    # Step 1: File localization only -> NOT mutation ready
    state.record_evidence({"hits": [{"path": "src/ModBlock.java"}]}, usable=True)
    assert isinstance(state.mutation_context, TargetMutationContext)
    assert state.mutation_context.target_path == "src/ModBlock.java"
    assert state.mutation_context.source_body is None
    assert is_mutation_ready(base_messages, state) is False

    # Step 2: Symbol localization in file -> NOT mutation ready (body missing)
    state.record_evidence(
        {
            "symbols": [
                {
                    "name": "getDroppedStacks",
                    "containerName": "ModBlock",
                    "location": {
                        "uri": "file:///src/ModBlock.java",
                        "range": {"start": {"line": 42}, "end": {"line": 58}},
                    },
                }
            ]
        },
        usable=True,
    )
    assert state.mutation_context.target_symbol == "ModBlock#getDroppedStacks"
    assert state.mutation_context.start_line == 42
    assert state.mutation_context.source_body is None
    assert is_mutation_ready(base_messages, state) is False

    # Step 3: Function/Method body retrieved -> MUTATION READY!
    state.record_evidence(
        {
            "hits": [
                {
                    "path": "src/ModBlock.java",
                    "symbol": "getDroppedStacks",
                    "snippet": (
                        "public List<ItemStack> getDroppedStacks(BlockState state, LootContext.Builder builder) {\n"
                        "    return Collections.singletonList(new ItemStack(Items.DIAMOND));\n"
                        "}"
                    ),
                    "start_line": 42,
                    "end_line": 46,
                }
            ]
        },
        usable=True,
    )
    assert state.mutation_context.is_mutation_ready is True
    assert is_mutation_ready(base_messages, state) is True


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


def test_filter_tools_for_phase_hierarchical_localization_stages() -> None:
    """The host controller dynamically filters read tools per hierarchical localization stage."""
    from minecraft_mod_ai.progress_aware_tool_loop import (
        LocalizationStage,
        LoopPhase,
        TargetMutationContext,
        _filter_tools_for_phase,
    )

    file_tool = _tool_schema("search_code_rag")
    symbol_tool = _tool_schema("java_workspace_symbols")
    mutate_tool = _tool_schema("apply_source_patch")
    all_tools = (file_tool, symbol_tool, mutate_tool)

    # 1. NEED_FILE: Host exposes file discovery tool only
    ctx_need_file = TargetMutationContext(target_path=None)
    assert ctx_need_file.localization_stage == LocalizationStage.NEED_FILE
    filtered = _filter_tools_for_phase(all_tools, LoopPhase.OBSERVE, "coder", mutation_context=ctx_need_file)
    assert filtered == (file_tool,)

    # 2. NEED_SYMBOL: Host exposes symbol discovery tool
    ctx_need_symbol = TargetMutationContext(target_path="src/Main.java", target_symbol=None)
    assert ctx_need_symbol.localization_stage == LocalizationStage.NEED_SYMBOL
    filtered = _filter_tools_for_phase(all_tools, LoopPhase.OBSERVE, "coder", mutation_context=ctx_need_symbol)
    assert filtered == (symbol_tool,)

    # 3. NEED_BODY: Host exposes snippet/body tool
    ctx_need_body = TargetMutationContext(target_path="src/Main.java", target_symbol="Main#init", source_body=None)
    assert ctx_need_body.localization_stage == LocalizationStage.NEED_BODY
    filtered = _filter_tools_for_phase(all_tools, LoopPhase.OBSERVE, "coder", mutation_context=ctx_need_body)
    assert filtered == (file_tool,)

    # 4. READY: In ACT phase, host exposes mutation tool
    ctx_ready = TargetMutationContext(
        target_path="src/Main.java",
        target_symbol="Main#init",
        source_body="public void init() { System.out.println(1); }",
    )
    assert ctx_ready.localization_stage == LocalizationStage.READY
    filtered = _filter_tools_for_phase(all_tools, LoopPhase.ACT, "coder", mutation_context=ctx_ready)
    assert filtered == (mutate_tool,)


def test_target_mutation_context_cumulative_merge() -> None:
    """TargetMutationContext merges step-by-step discoveries without losing prior fields."""
    from minecraft_mod_ai.progress_aware_tool_loop import LocalizationStage, TargetMutationContext

    ctx1 = TargetMutationContext(target_path="src/Item.java", evidence_source="search_code_rag")
    assert ctx1.localization_stage == LocalizationStage.NEED_SYMBOL

    ctx2 = TargetMutationContext(target_symbol="Item#use", start_line=20, end_line=35, evidence_source="java_workspace_symbols")
    merged1 = ctx1.merge(ctx2)
    assert merged1.target_path == "src/Item.java"
    assert merged1.target_symbol == "Item#use"
    assert merged1.start_line == 20
    assert merged1.end_line == 35
    assert merged1.localization_stage == LocalizationStage.NEED_BODY

    ctx3 = TargetMutationContext(source_body="public ActionResult use(World w) { return PASS; }")
    merged2 = merged1.merge(ctx3)
    assert merged2.target_path == "src/Item.java"
    assert merged2.target_symbol == "Item#use"
    assert merged2.source_body == "public ActionResult use(World w) { return PASS; }"
def test_java_workspace_symbols_records_evidence_and_progresses_localization() -> None:
    """java_workspace_symbols is an evidence tool that advances LocalizationStage and makes semantic progress."""
    from minecraft_mod_ai.progress_aware_tool_loop import generate_with_tools

    router = MagicMock()
    router._generation_scope.return_value = nullcontext()
    router._agent_require_fresh_evidence = False

    config = MagicMock()
    config.adapter = "llama_cpp"
    config.max_context = 16384
    config.max_new_tokens = 4096
    config.extra = {"runtime_contract": "qwen", "qwen_family": "qwen3.5"}

    turns = [
        # Turn 1: In OBSERVE (NEED_FILE), model calls search_code_rag
        GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_file",
                    name="search_code_rag",
                    arguments={"query": "ModBlock"},
                    raw_arguments='{"query":"ModBlock"}',
                ),
            )
        ),
        # Turn 2: In OBSERVE (NEED_SYMBOL), model calls java_workspace_symbols
        GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_sym",
                    name="java_workspace_symbols",
                    arguments={"project_root": ".", "query": "getDroppedStacks"},
                    raw_arguments='{"project_root":".","query":"getDroppedStacks"}',
                ),
            )
        ),
        # Turn 3: In OBSERVE (NEED_BODY), model calls search_code_rag for the body
        GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_body",
                    name="search_code_rag",
                    arguments={"query": "ModBlock getDroppedStacks"},
                    raw_arguments='{"query":"ModBlock getDroppedStacks"}',
                ),
            )
        ),
        # Turn 4: In ACT (READY), model calls apply_source_patch
        GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_patch",
                    name="apply_source_patch",
                    arguments={"patch": "applied"},
                    raw_arguments='{"patch":"applied"}',
                ),
            )
        ),
        # Turn 5: Final completion in prose
        GenerationResponse(content="Successfully updated drop logic."),
    ]

    adapter = MagicMock()
    adapter.generate_turn.side_effect = turns

    def mock_runtime_call(stage: str, name: str, args: dict) -> dict:
        if name == "search_code_rag" and "ModBlock getDroppedStacks" in str(args):
            return {
                "hits": [
                    {
                        "path": "src/ModBlock.java",
                        "symbol": "getDroppedStacks",
                        "snippet": "public List<ItemStack> getDroppedStacks() { return List.of(); }",
                    }
                ]
            }
        elif name == "search_code_rag":
            return {"hits": [{"path": "src/ModBlock.java"}]}
        elif name == "java_workspace_symbols":
            return {
                "symbols": [
                    {
                        "name": "getDroppedStacks",
                        "containerName": "ModBlock",
                        "location": {
                            "uri": "file:///src/ModBlock.java",
                            "range": {"start": {"line": 42}, "end": {"line": 50}},
                        },
                    }
                ]
            }
        elif name == "apply_source_patch":
            return {"ok": True, "applied": True, "patched_files": ["src/ModBlock.java"]}
        return {}

    runtime = MagicMock()
    runtime.call.side_effect = mock_runtime_call

    request = GenerationRequest(
        messages=(
            {"role": "user", "content": '{"phase": "implement_module", "task": "fix drop logic"}'},
        ),
        tools=(
            _tool_schema("search_code_rag"),
            _tool_schema("java_workspace_symbols"),
            _tool_schema("apply_source_patch"),
        ),
        tool_choice=None,
        parallel_tool_calls=False,
    )

    result = generate_with_tools(
        router,
        config=config,
        adapter=adapter,
        request=request,
        runtime=runtime,
        stage="generation",
        role="coder",
    )

    assert result == "Successfully updated drop logic."


def test_execution_step_trace_records_immutable_trajectory_and_summary() -> None:
    """Every turn records an immutable ExecutionStepTrace capturing model call, results, and state deltas."""
    from minecraft_mod_ai.progress_aware_tool_loop import (
        ExecutionStepTrace,
        format_trajectory_summary,
    )

    trace1 = ExecutionStepTrace(
        step_index=1,
        phase_before="OBSERVE",
        localization_stage_before="NEED_FILE",
        mutation_context_before=None,
        exposed_tools=["search_code_rag"],
        tool_choice=None,
        input_messages_count=2,
        model_response_content=None,
        model_tool_calls=[{"name": "search_code_rag", "arguments": {"query": "ModItem"}}],
        query_signatures=["search_code_rag:moditem"],
        tool_results=[{"name": "search_code_rag", "ok": True, "error": None}],
        mutation_context_after={"target_path": "src/ModItem.java", "localization_stage": "NEED_SYMBOL"},
        localization_stage_after="NEED_SYMBOL",
        phase_after="OBSERVE",
        turn_made_progress=True,
        no_progress_streak_after=0,
        action_decision="tool_wave_executed",
    )

    trace2 = ExecutionStepTrace(
        step_index=2,
        phase_before="OBSERVE",
        localization_stage_before="NEED_SYMBOL",
        mutation_context_before={"target_path": "src/ModItem.java"},
        exposed_tools=["java_workspace_symbols"],
        tool_choice=None,
        input_messages_count=4,
        model_response_content=None,
        model_tool_calls=[{"name": "java_workspace_symbols", "arguments": {"query": "use"}}],
        query_signatures=["java_workspace_symbols:use"],
        tool_results=[{"name": "java_workspace_symbols", "ok": True, "error": None}],
        mutation_context_after={"target_path": "src/ModItem.java", "target_symbol": "use", "localization_stage": "NEED_BODY"},
        localization_stage_after="NEED_BODY",
        phase_after="OBSERVE",
        turn_made_progress=True,
        no_progress_streak_after=0,
        action_decision="tool_wave_executed",
    )

    summary = format_trajectory_summary([trace1, trace2])
    assert "Step 1 [OBSERVE:NEED_FILE -> OBSERVE:NEED_SYMBOL]" in summary
    assert "search_code_rag" in summary
    assert "Step 2 [OBSERVE:NEED_SYMBOL -> OBSERVE:NEED_BODY]" in summary
    assert "java_workspace_symbols" in summary
    assert "progress=True streak=0" in summary





