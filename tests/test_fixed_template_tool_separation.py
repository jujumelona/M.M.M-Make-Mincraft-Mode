from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value


_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {"type": "string", "minLength": 1, "maxLength": 256},
        "input": {"type": "string", "minLength": 1, "maxLength": 256},
        "output": {"type": "string", "minLength": 1, "maxLength": 256},
    },
    "required": ["operation", "input", "output"],
    "additionalProperties": False,
}
_RESULT = {
    "operation": "collect resource",
    "input": "resource node",
    "output": "resource item",
}


class _Registry:
    def __init__(self, adapter: str = "llama_cpp") -> None:
        self.adapter = adapter

    def role(self, profile: str, role: str):
        assert profile == "local"
        assert role == "coder"
        return SimpleNamespace(adapter=self.adapter)


class _ToolCapableRouter:
    profile = "local"
    registry = _Registry()

    def __init__(self) -> None:
        self.tool_calls: list[dict[str, object]] = []
        self.text_calls = 0

    def generate_text(self, *args, **kwargs):
        self.text_calls += 1
        raise AssertionError(
            "enable_tools=False must disable the semantic prelude, not fixed-template transport"
        )

    def generate_tool_decision(self, role, messages, **kwargs):
        self.tool_calls.append({"role": role, "messages": messages, **kwargs})
        return dict(_RESULT)


class _TextOnlyRouter:
    profile = "local"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.text_calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(_RESULT)


class _NoTransportRouter:
    profile = "local"
    registry = _Registry()


class _MockRouter(_TextOnlyRouter):
    registry = _Registry("mock")

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError("mock fixtures must stay on structured text transport")


def _generate(router, *, enable_tools: bool = False):
    return generate_fixed_template_value(
        router,
        "coder",
        [{"role": "user", "content": "Fill the record."}],
        response_schema=_SCHEMA,
        enable_tools=enable_tools,
        tool_name="submit_one_feature_algorithm_steps_part_1_of_2",
    )


def test_tools_disabled_real_router_still_uses_fixed_template_tool_transport() -> None:
    router = _ToolCapableRouter()

    assert _generate(router, enable_tools=False) == _RESULT
    assert router.text_calls == 0
    assert len(router.tool_calls) == 1
    assert router.tool_calls[0]["parameters"] == _SCHEMA


def test_text_only_router_falls_back_to_structured_text_transport() -> None:
    router = _TextOnlyRouter()

    assert _generate(router, enable_tools=False) == _RESULT
    assert len(router.text_calls) == 1
    call = router.text_calls[0]
    assert call["response_format"] == "json"
    assert call["response_schema"] == _SCHEMA
    assert call["enable_tools"] is False


def test_mock_router_keeps_fixture_transport_even_when_tool_surface_exists() -> None:
    router = _MockRouter()

    assert _generate(router, enable_tools=True) == _RESULT
    assert len(router.text_calls) == 1


def test_router_without_any_fixed_template_transport_fails_explicitly() -> None:
    with pytest.raises(RuntimeError, match="FIXED_TEMPLATE_TRANSPORT_UNAVAILABLE"):
        _generate(_NoTransportRouter(), enable_tools=False)
