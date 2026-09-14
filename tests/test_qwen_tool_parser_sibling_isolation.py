from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.qwen_tool_parser import (
    ToolCallValidationError,
    parse_qwen_tool_markup,
)


def _schema(required: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {required: {"type": "string"}},
        "required": [required],
        "additionalProperties": False,
    }


def _call(name: str, key: str, value: str) -> str:
    return (
        "<tool_call>"
        f"<function={name}>"
        f"<parameter={key}>{value}</parameter>"
        "</function>"
        "</tool_call>"
    )


def test_malformed_sibling_does_not_discard_valid_calls_before_or_after_it() -> None:
    schemas = {"first": _schema("value"), "second": _schema("value")}
    text = "".join(
        (
            _call("first", "value", "one"),
            _call("second", "wrong", "rejected"),
            _call("first", "value", "two"),
        )
    )

    visible, calls = parse_qwen_tool_markup(text, schemas)

    assert [(call.name, dict(call.arguments)) for call in calls] == [
        ("first", {"value": "one"}),
        ("first", {"value": "two"}),
    ]
    assert "rejected malformed tool call" in visible
    assert "unknown parameter 'wrong'" in visible


def test_all_malformed_tool_calls_remain_a_hard_failure() -> None:
    schemas = {"only": _schema("value")}

    with pytest.raises(ToolCallValidationError, match="unknown parameter 'wrong'"):
        parse_qwen_tool_markup(_call("only", "wrong", "bad"), schemas)


def test_plain_text_contract_is_unchanged() -> None:
    text = "ordinary assistant prose"
    assert parse_qwen_tool_markup(text, {}) == (text, ())
