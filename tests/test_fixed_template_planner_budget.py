from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.execution_contract_policy import (
    ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING,
)
from minecraft_mod_ai.fixed_template_generation import (
    _generate_native_template_arguments,
    generate_fixed_template_value,
)


_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "minLength": 1, "maxLength": 32},
    },
    "required": ["value"],
    "additionalProperties": False,
}


class _Registry:
    @staticmethod
    def role(_profile: str, _role: str):
        return SimpleNamespace(adapter="llama_cpp")


class _PlannerRouter:
    profile = "test"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls: list[tuple[tuple[dict, ...], dict]] = []
        self.tool_calls = 0

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        copied = tuple(dict(message) for message in messages)
        self.text_calls.append((copied, dict(kwargs)))
        return json.dumps({"value": "ok"})

    def generate_tool_decision(self, *_args, **_kwargs):
        self.tool_calls += 1
        raise AssertionError("planner fixed templates must never use native tools")


def test_planner_fixed_template_uses_schema_json_not_native_tool() -> None:
    router = _PlannerRouter()

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "author one bounded page"},),
        response_schema=_SCHEMA,
        enable_tools=False,
    )

    assert value == {"value": "ok"}
    assert router.tool_calls == 0
    assert len(router.text_calls) == 1
    _messages, kwargs = router.text_calls[0]
    assert kwargs["response_format"] == "json"
    assert kwargs["response_schema"] == _SCHEMA
    assert kwargs["enable_tools"] is False
    assert kwargs["force_non_thinking"] is True
    assert kwargs["output_token_ceiling"] == ATOMIC_CONCERN_OUTPUT_TOKEN_CEILING


def test_planner_fixed_template_honors_host_page_ceiling() -> None:
    router = _PlannerRouter()

    value = generate_fixed_template_value(
        router,
        "planner",
        ({"role": "user", "content": "author one bounded page"},),
        response_schema=_SCHEMA,
        enable_tools=False,
        output_token_ceiling=777,
    )

    assert value == {"value": "ok"}
    assert router.tool_calls == 0
    assert router.text_calls[0][1]["output_token_ceiling"] == 777


def test_planner_native_tool_transport_is_fail_closed() -> None:
    router = _PlannerRouter()

    with pytest.raises(
        RuntimeError,
        match="FIXED_TEMPLATE_PLANNER_NATIVE_TOOL_FORBIDDEN",
    ):
        _generate_native_template_arguments(
            router,
            "planner",
            ({"role": "user", "content": "x"},),
            tool_name="forbidden",
            parameters=_SCHEMA,
            description="must not run",
        )

    assert router.tool_calls == 0
