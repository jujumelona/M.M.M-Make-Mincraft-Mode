from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.model_output_atomicity_contract import structured_output_token_ceiling
from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value


class _Adapter:
    def __init__(self) -> None:
        self.request = None

    def generate(self, request):
        self.request = request
        return "ok"

    def generate_turn(self, request):
        self.request = request
        tool = request.tools[0]
        name = tool["function"]["name"]
        field = next(iter(tool["function"]["parameters"]["properties"]))
        return GenerationResponse(
            tool_calls=(
                ToolCall(
                    id="call_test",
                    name=name,
                    arguments={field: "ok"},
                    raw_arguments='{"' + field + '":"ok"}',
                ),
            )
        )


class _Router(ModelRouter):
    def __init__(self) -> None:
        self._agent_require_fresh_evidence = False
        self.adapter = _Adapter()

    def _generation_adapter(self, role: str):
        del role
        return SimpleNamespace(adapter="llama_cpp"), self.adapter

    def _tools_enabled(self, **kwargs):
        del kwargs
        return False

    @contextmanager
    def _generation_scope(self, config):
        del config
        yield


def _planner_json_response(adapter, request):
    adapter.request = request
    return '{"value":"ok"}'


def test_generate_text_propagates_output_token_ceiling_into_request_metadata() -> None:
    router = _Router()

    assert router.generate_text(
        "coder",
        [{"role": "user", "content": "write one file"}],
        enable_tools=False,
        output_token_ceiling=4096,
    ) == "ok"

    assert router.adapter.request is not None
    assert router.adapter.request.metadata["mmm_output_token_ceiling"] == 4096



def test_generate_planner_json_derives_proven_budget_from_schema() -> None:
    router = _Router()
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 64}},
        "required": ["value"],
        "additionalProperties": False,
    }
    router.adapter.generate = lambda request: _planner_json_response(router.adapter, request)
    result = generate_fixed_template_value(
        router,
        "planner",
        [{"role": "user", "content": "fill one bounded field"}],
        response_schema=parameters,
    )

    assert result == {"value": "ok"}
    assert router.adapter.request is not None
    assert (
        router.adapter.request.metadata["mmm_output_token_ceiling"]
        == structured_output_token_ceiling(parameters)
    )
    assert router.adapter.request.metadata["mmm_force_non_thinking"] is True


def test_generate_planner_json_bounds_and_decodes_numeric_transport() -> None:
    router = _Router()
    original_schema = {
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def numeric_response(request):
        router.adapter.request = request
        return '{"value":"1.25"}'

    router.adapter.generate = numeric_response
    result = generate_fixed_template_value(
        router,
        "planner",
        [{"role": "user", "content": "fill one number"}],
        response_schema=original_schema,
    )

    assert result == {"value": 1.25}
    assert router.adapter.request is not None
    wire_field = router.adapter.request.response_schema["properties"]["value"]
    # A lexical number has a finite decoder bound, even when the logical
    # output schema did not choose an arbitrary numerical range.
    assert wire_field["type"] == "string"
    assert wire_field["maxLength"] == 32
    assert router.adapter.request.metadata["mmm_force_non_thinking"] is True


def test_generate_planner_json_rejects_budget_below_schema_proof() -> None:
    router = _Router()
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 64}},
        "required": ["value"],
        "additionalProperties": False,
    }
    required = structured_output_token_ceiling(parameters)

    with pytest.raises(Exception, match="FIXED_TEMPLATE_OUTPUT_BUDGET_TOO_SMALL"):
        generate_fixed_template_value(
            router,
            "planner",
            [{"role": "user", "content": "fill one bounded field"}],
            response_schema=parameters,
            output_token_ceiling=required - 1,
        )

    assert router.adapter.request is None


def test_planner_native_tool_decision_transport_is_forbidden() -> None:
    router = _Router()
    with pytest.raises(Exception, match="PLANNER_NATIVE_TOOL_TRANSPORT_REMOVED"):
        router.generate_tool_decision(
            "planner",
            [{"role": "user", "content": "irrelevant"}],
            tool_name="deprecated_planner_tool",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string", "maxLength": 16}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    assert router.adapter.request is None
