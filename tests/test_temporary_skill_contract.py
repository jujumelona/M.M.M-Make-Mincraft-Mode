from __future__ import annotations

from minecraft_mod_ai import temporary_skill_contract
from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_docs",
        description="Search the available documentation.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def test_temporary_skill_preserves_native_tool_transport_contract(monkeypatch, tmp_path) -> None:
    tool = _tool()
    validation_tool = ToolDefinition(
        name="validate_docs",
        description="Validate a documentation result.",
        parameters={"type": "object", "properties": {}},
    )
    metadata = {"trace_id": "host-owned-transport"}
    request = GenerationRequest(
        task="execute_skill",
        prompt="Find the documented answer.",
        tools=(tool,),
        tool_validation_schemas=(tool, validation_tool),
        tool_choice="required",
        parallel_tool_calls=False,
        metadata=metadata,
        messages=({"role": "user", "content": "Find the documented answer."},),
    )

    class FakeRouter:
        def __init__(self) -> None:
            self._agent_workspace_root = tmp_path

        def _prepare_generation_request(self, role, messages, **kwargs):
            return "generation", object(), request.tools, request

    class FakeModelRouterModule:
        ModelRouter = FakeRouter

    monkeypatch.setattr(temporary_skill_contract, "remote_configured", lambda: False)
    monkeypatch.setattr(
        temporary_skill_contract,
        "_temporary_skill",
        lambda *args, **kwargs: {
            "source_trajectory_ids": ["trajectory-a", "trajectory-b"],
            "proven_patterns": ["retrieve before editing"],
            "avoid_patterns": [],
        },
    )
    temporary_skill_contract._install_model_skill(FakeModelRouterModule)

    stage, runtime, tools, prepared = FakeRouter()._prepare_generation_request(
        "coder",
        request.messages,
    )

    assert stage == "generation"
    assert runtime is not None
    assert tools == request.tools
    assert prepared.tools == request.tools
    assert prepared.tool_validation_schemas == request.tool_validation_schemas
    assert prepared.tool_choice == request.tool_choice == "required"
    assert prepared.parallel_tool_calls is False
    assert prepared.task == request.task
    assert prepared.prompt == request.prompt
    assert prepared.metadata == request.metadata
    assert prepared.media_paths == request.media_paths
    assert prepared.response_format == request.response_format
    assert prepared.response_schema == request.response_schema
    assert prepared.messages[0]["role"] == "system"
    assert prepared.messages[0]["content"].startswith(
        "MMM TEMPORARY VERIFIED SKILL:\n"
    )
    assert prepared.messages[1] == request.messages[0]


def test_temporary_skill_without_workspace_is_transport_noop() -> None:
    request = GenerationRequest(
        task="plain",
        prompt="Run normally.",
        tools=(_tool(),),
        tool_validation_schemas=(_tool(),),
        tool_choice="auto",
        metadata={"trace_id": "plain"},
    )

    class FakeRouter:
        _agent_workspace_root = None

        def _prepare_generation_request(self, role, messages, **kwargs):
            return "generation", None, request.tools, request

    class FakeModelRouterModule:
        ModelRouter = FakeRouter

    temporary_skill_contract._install_model_skill(FakeModelRouterModule)

    stage, runtime, tools, prepared = FakeRouter()._prepare_generation_request(
        "coder",
        request.messages,
    )

    assert stage == "generation"
    assert runtime is None
    assert tools == request.tools
    assert prepared is request
