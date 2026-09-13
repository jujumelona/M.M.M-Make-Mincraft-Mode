from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.fixed_template_generation import generate_fixed_template_value


class _Registry:
    def role(self, profile: str, role: str):
        assert profile == "local"
        assert role == "planner"
        return SimpleNamespace(adapter="llama_cpp")


class _RealAdapterRouter:
    profile = "local"
    registry = _Registry()

    def __init__(self) -> None:
        self.text_calls: list[dict[str, object]] = []

    def generate_text(self, role, messages, **kwargs):
        self.text_calls.append({"role": role, "messages": messages, **kwargs})
        return json.dumps(
            {
                "operation": "collect resource",
                "input": "resource node",
                "output": "resource item",
            }
        )

    def generate_tool_decision(self, *args, **kwargs):
        raise AssertionError(
            "enable_tools=False fixed-template generation must not enter tool transport"
        )


def test_fixed_template_with_tools_disabled_uses_structured_text_path() -> None:
    router = _RealAdapterRouter()
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "minLength": 1},
            "input": {"type": "string", "minLength": 1},
            "output": {"type": "string", "minLength": 1},
        },
        "required": ["operation", "input", "output"],
        "additionalProperties": False,
    }

    result = generate_fixed_template_value(
        router,
        "planner",
        [{"role": "user", "content": "Fill the record."}],
        response_schema=schema,
        enable_tools=False,
        tool_name="submit_one_feature_algorithm_steps_part_1_of_2",
    )

    assert result == {
        "operation": "collect resource",
        "input": "resource node",
        "output": "resource item",
    }
    assert len(router.text_calls) == 1
    call = router.text_calls[0]
    assert call["response_format"] == "json"
    assert call["response_schema"] == schema
    assert call["enable_tools"] is False
