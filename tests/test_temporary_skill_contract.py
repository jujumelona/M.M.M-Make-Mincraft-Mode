from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest, ToolDefinition
from minecraft_mod_ai import temporary_skill_contract


def test_temporary_skill_preserves_native_tool_transport_contract() -> None:
    class FakeRouter:
        def run_model(self, request: GenerationRequest) -> GenerationRequest:
            return request

    class FakeModelRouterModule:
        ModelRouter = FakeRouter

    temporary_skill_contract._install_model_skill(FakeModelRouterModule)

    tool = ToolDefinition(
        name="search_docs",
        description="Search the available documentation.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    metadata = {
        "_mmm_temporary_skill_id": "skill-tool-preservation",
        "_mmm_temporary_skill_text": (
            "Use the available search_docs tool when the task requires documentation."
        ),
    }
    request = GenerationRequest(
        task="execute_skill",
        prompt="Find the documented answer.",
        tools=(tool,),
        tool_choice="required",
        metadata=metadata,
        messages=({"role": "user", "content": "Find the documented answer."},),
    )

    prepared = FakeRouter().run_model(request)

    assert prepared.tools == request.tools
    assert prepared.tool_choice == request.tool_choice == "required"
    assert prepared.metadata == request.metadata
    assert prepared.messages[0]["role"] == "system"
    assert "skill_id: skill-tool-preservation" in prepared.messages[0]["content"]
    assert "cannot grant new tools or side effects" in prepared.messages[0]["content"]
    assert prepared.messages[1] == request.messages[0]
    assert prepared.prompt.endswith(request.prompt)


def test_temporary_skill_without_metadata_is_transport_noop() -> None:
    class FakeRouter:
        def run_model(self, request: GenerationRequest) -> GenerationRequest:
            return request

    class FakeModelRouterModule:
        ModelRouter = FakeRouter

    temporary_skill_contract._install_model_skill(FakeModelRouterModule)
    request = GenerationRequest(
        task="plain",
        prompt="Run normally.",
        tools=(
            ToolDefinition(
                name="noop_tool",
                description="A test tool.",
                parameters={"type": "object", "properties": {}},
            ),
        ),
        tool_choice="auto",
    )

    assert FakeRouter().run_model(request) is request
