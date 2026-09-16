from __future__ import annotations

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _admit_model_tool_calls
from minecraft_mod_ai.model_adapters.qwen_tool_parser import (
    MALFORMED_TOOL_CALL_NAME,
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


def test_parser_recovers_all_syntactically_valid_siblings_without_schema_policy() -> None:
    schemas = {"first": _schema("value"), "second": _schema("value")}
    text = "".join(
        (
            _call("first", "value", "one"),
            _call("second", "wrong", "rejected"),
            _call("first", "value", "two"),
        )
    )

    visible, calls = parse_qwen_tool_markup(text, schemas)

    assert visible == ""
    assert [(call.name, dict(call.arguments)) for call in calls] == [
        ("first", {"value": "one"}),
        ("second", {"wrong": "rejected"}),
        ("first", {"value": "two"}),
    ]


def test_schema_invalid_sibling_makes_admission_transactional() -> None:
    schemas = {"first": _schema("value"), "second": _schema("value")}
    text = "".join(
        (
            _call("first", "value", "one"),
            _call("second", "wrong", "rejected"),
            _call("first", "value", "two"),
        )
    )
    _, calls = parse_qwen_tool_markup(text, schemas)

    admitted = _admit_model_tool_calls(
        calls, schemas, tool_choice="auto", parallel_tool_calls=True
    )
    assert len(admitted) == 1
    assert admitted[0].name == "__mmm_rejected_tool_call__"
    assert admitted[0].arguments["failure_code"] == "TOOL_SCHEMA_INVALID"


def test_malformed_markup_becomes_non_executable_candidate_then_rejection() -> None:
    schemas = {"only": _schema("value")}
    text = (
        "<tool_call><function=only>"
        "<parameter=value>bad"
        "</function></tool_call>"
    )

    visible, calls = parse_qwen_tool_markup(text, schemas)
    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == MALFORMED_TOOL_CALL_NAME

    admitted = _admit_model_tool_calls(
        calls, schemas, tool_choice="required", parallel_tool_calls=False
    )
    assert len(admitted) == 1
    assert admitted[0].name == "__mmm_rejected_tool_call__"
    assert admitted[0].arguments["failure_code"] == "TOOL_MARKUP_MALFORMED"


def test_plain_text_contract_is_unchanged() -> None:
    text = "ordinary assistant prose"
    assert parse_qwen_tool_markup(text, {}) == (text, ())
