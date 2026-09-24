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


def test_truncated_wrapped_source_edit_is_one_rejected_candidate() -> None:
    text = (
        "<tool_call>\n<function=apply_source_edit>\n"
        "<parameter=path>src/Feature.java</parameter>\n"
        "<parameter=new>public final class Feature {"
    )

    visible, calls = parse_qwen_tool_markup(text)

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == MALFORMED_TOOL_CALL_NAME
    assert calls[0].arguments["original_tool"] == "apply_source_edit"
    assert calls[0].arguments["raw_arguments"] == text
    admitted = _admit_model_tool_calls(
        calls, {}, tool_choice="required", parallel_tool_calls=False
    )
    assert len(admitted) == 1
    assert admitted[0].arguments["failure_code"] == "TOOL_MARKUP_MALFORMED"
    assert "missing </function>" in admitted[0].arguments["error"]


def test_unclosed_function_cannot_borrow_a_siblings_closing_tags() -> None:
    malformed = "<tool_call><function=first><parameter=value>partial"
    sibling = _call("second", "value", "complete")

    visible, calls = parse_qwen_tool_markup(malformed + sibling)

    assert visible == ""
    assert len(calls) == 2
    assert calls[0].name == MALFORMED_TOOL_CALL_NAME
    assert calls[0].arguments["raw_arguments"] == malformed
    assert calls[1].name == "second"
    assert calls[1].arguments == {"value": "complete"}
    admitted = _admit_model_tool_calls(
        calls, {"second": _schema("value")},
        tool_choice="required", parallel_tool_calls=True,
    )
    assert len(admitted) == 1
    assert admitted[0].name == "__mmm_rejected_tool_call__"


def test_truncated_wrapper_does_not_absorb_an_unwrapped_sibling() -> None:
    malformed = "<tool_call><function=first><parameter=value>partial"
    sibling = "<function=second><parameter=value>complete</parameter></function>"

    visible, calls = parse_qwen_tool_markup(malformed + sibling)

    assert visible == ""
    assert len(calls) == 2
    assert calls[0].name == MALFORMED_TOOL_CALL_NAME
    assert calls[0].arguments["raw_arguments"] == malformed
    assert calls[1].name == "second"
