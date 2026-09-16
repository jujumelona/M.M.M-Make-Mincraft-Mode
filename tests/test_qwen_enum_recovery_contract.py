from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _admit_model_tool_calls
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup


def _schema() -> dict[str, dict]:
    return {
        "apply_source_edit": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["replace_exact", "delete_file"],
                }
            },
            "required": ["operation"],
        }
    }


def _parse(operation: str, *, name: str = "apply_source_edit"):
    text = (
        f"<function={name}>"
        f"<parameter=operation>{operation}</parameter>"
        "</function>"
    )
    visible, calls = parse_qwen_tool_markup(text, _schema())
    assert visible == ""
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize(
    "raw",
    ["REPLACE_EXACT", "replace-exact", "replace exact", "replaceExact"],
)
def test_parser_preserves_noncanonical_enum_spelling(raw: str) -> None:
    call = _parse(raw)
    assert call.arguments == {"operation": raw}


def test_parser_decodes_quoted_canonical_enum_as_transport_value() -> None:
    call = _parse('"replace_exact"')
    assert call.arguments == {"operation": "replace_exact"}


def test_admission_canonicalizes_noncanonical_enum_in_one_place() -> None:
    call = _parse("REPLACE_EXACT")
    admitted = _admit_model_tool_calls(
        (call,), _schema(), tool_choice="auto", parallel_tool_calls=False
    )
    assert len(admitted) == 1
    assert admitted[0].name == "apply_source_edit"
    assert admitted[0].arguments == {"operation": "replace_exact"}


def test_tool_name_alias_is_resolved_at_admission_not_parse_time() -> None:
    call = _parse("delete_file", name="apply_source_patch")
    assert call.name == "apply_source_patch"

    admitted = _admit_model_tool_calls(
        (call,), _schema(), tool_choice="auto", parallel_tool_calls=False
    )
    assert len(admitted) == 1
    assert admitted[0].name == "apply_source_edit"
    assert admitted[0].arguments == {"operation": "delete_file"}


def test_unrelated_unexposed_tool_is_preserved_by_parser_then_rejected_by_admission() -> None:
    text = "<function=other_tool></function>"
    visible, calls = parse_qwen_tool_markup(text, _schema())
    assert visible == ""
    assert calls[0].name == "other_tool"

    admitted = _admit_model_tool_calls(
        calls, _schema(), tool_choice="auto", parallel_tool_calls=False
    )
    assert admitted[0].name == "__mmm_rejected_tool_call__"
    assert admitted[0].arguments["failure_code"] == "TOOL_NOT_VISIBLE"
