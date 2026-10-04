from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters.base import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.worksheet_atomic_chunker import planner_page_output_token_ceiling


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



def test_generate_tool_decision_derives_planner_budget_from_schema() -> None:
    router = _Router()
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string", "maxLength": 64}},
        "required": ["value"],
        "additionalProperties": False,
    }
    result = router.generate_tool_decision(
        "planner",
        [{"role": "user", "content": "fill one bounded field"}],
        tool_name="bounded_probe",
        parameters=parameters,
    )

    assert result == {"value": "ok"}
    assert router.adapter.request is not None
    assert (
        router.adapter.request.metadata["mmm_output_token_ceiling"]
        == planner_page_output_token_ceiling(parameters)
    )
    assert router.adapter.request.metadata["mmm_force_non_thinking"] is True


def test_generate_tool_decision_rejects_unbounded_planner_schema_without_budget() -> None:
    router = _Router()

    with pytest.raises(Exception, match="PLANNER_TOOL_DECISION_BUDGET_REQUIRED"):
        router.generate_tool_decision(
            "planner",
            [{"role": "user", "content": "fill one number"}],
            tool_name="unbounded_probe",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )

    assert router.adapter.request is None
