from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from minecraft_mod_ai import model_router
from minecraft_mod_ai.model_adapters import GenerationRequest, ModelConfigurationError
from minecraft_mod_ai.model_adapters import llama_cpp_adapter


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_llama_reasoning_continuation_preserves_full_request_contract() -> None:
    visible = _schema("visible")
    authorized = _schema("authorized")
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        media_paths=(Path("fixture.png"),),
        response_format="json_object",
        response_schema={"type": "object"},
        tools=(visible,),
        tool_validation_schemas=(visible, authorized),
        tool_choice="auto",
        parallel_tool_calls=False,
        task="sentinel-task",
        prompt="sentinel-prompt",
        metadata={"sentinel": "metadata"},
    )
    core = inspect.unwrap(llama_cpp_adapter._reasoning_continuation_request)

    continued = core(request, "reasoning")

    assert continued.media_paths == ()
    assert continued.messages[:1] == request.messages
    assert continued.response_format == request.response_format
    assert continued.response_schema == request.response_schema
    assert continued.tools == request.tools
    assert continued.tool_validation_schemas == request.tool_validation_schemas
    assert continued.tool_choice == request.tool_choice
    assert continued.parallel_tool_calls == request.parallel_tool_calls
    assert continued.task == request.task
    assert continued.prompt == request.prompt
    assert dict(continued.metadata) == dict(request.metadata)


def test_model_router_core_derives_tool_turns_with_dataclass_replace() -> None:
    core = inspect.unwrap(model_router.ModelRouter._generate_with_tools)
    source = inspect.getsource(core)

    # Derived turns must inherit the entire GenerationRequest contract. If a future
    # field is added to the dataclass, replace() carries it automatically rather than
    # requiring every tool-loop branch to remember to copy it.
    assert "GenerationRequest(" not in source
    assert source.count("replace(") >= 3
    assert "tool_validation_schemas=()" in source


def test_model_router_tool_name_projection_rejects_duplicate_ownership() -> None:
    duplicate = (_schema("lookup"), _schema("lookup"))

    with pytest.raises(ModelConfigurationError, match="Duplicate model tool schema name"):
        model_router._tool_schema_names(duplicate)


def test_model_router_tool_name_projection_rejects_malformed_schema() -> None:
    with pytest.raises(ModelConfigurationError, match="lacks function metadata"):
        model_router._tool_schema_names(({"type": "function"},))
