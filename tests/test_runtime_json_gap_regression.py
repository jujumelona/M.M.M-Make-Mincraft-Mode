from types import SimpleNamespace

from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai import planning_stall_guard_contract as guard


def test_json_requests_never_enable_native_llama_grammar():
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))
    request = SimpleNamespace(
        messages=({"role": "user", "content": "return JSON"},),
        tools=(),
        response_format="json",
    )
    payload = _server_payload(adapter, request)
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_tool_json_turn_never_mixes_native_grammar_controls():
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=512))
    tool = {
        "type": "function",
        "function": {
            "name": "search_project_rag",
            "description": "search",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    request = SimpleNamespace(
        messages=({"role": "user", "content": "use the tool"},),
        tools=(tool,),
        tool_choice="auto",
        response_format="json",
    )
    payload = _server_payload(adapter, request)
    assert payload["tools"] == [tool]
    assert payload["tool_choice"] == "auto"
    assert "response_format" not in payload
    assert "json_schema" not in payload
    assert "grammar" not in payload
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_terminal_gap_is_not_counted_as_verified_completion():
    progress = guard._PlanningProgress(total=2)
    progress_token = guard._ACTIVE_PROGRESS.set(progress)
    cursor_token = guard._ACTIVE_PROGRESS_CURSOR.set(None)
    try:
        guard._research_progress_hook(
            {
                "event": "domain_gap_receipt",
                "domain_id": "broken-domain",
                "page_index": 2,
                "page_count": 2,
            }
        )
        snapshot = progress.snapshot()
        assert snapshot["completed"] == 0
        assert snapshot["gaps"] == 1
        assert snapshot["terminal"] == 1

        guard._research_progress_hook(
            {
                "event": "domain_complete",
                "domain_id": "verified-domain",
                "page_index": 1,
                "page_count": 1,
            }
        )
        snapshot = progress.snapshot()
        assert snapshot["completed"] == 1
        assert snapshot["gaps"] == 1
        assert snapshot["terminal"] == 2
    finally:
        guard._ACTIVE_PROGRESS_CURSOR.reset(cursor_token)
        guard._ACTIVE_PROGRESS.reset(progress_token)


def test_runtime_trajectory_retrieval_keeps_execution_context_contract():
    import inspect

    from minecraft_mod_ai import temporary_skill_contract, trajectory_memory

    assert "current_context" in inspect.signature(trajectory_memory.relevant_trajectories, follow_wrapped=False).parameters
    assert temporary_skill_contract._trajectory_memory is trajectory_memory
    assert "relevant_trajectories" not in temporary_skill_contract.__dict__
