from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.qwen_tool_parser import (
    _canonical_string_enum,
    _parse_qwen_function,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("replace_exact", "replace_exact"),
        ('"replace_exact"', "replace_exact"),
        ("  REPLACE_EXACT  ", "replace_exact"),
        ("replace-exact", "replace_exact"),
        ("replace exact", "replace_exact"),
        ("replaceExact", "replace_exact"),
    ],
)
def test_canonical_string_enum_recovers_formatting_only(raw: str, expected: str) -> None:
    allowed = ["replace_exact", "insert_before", "insert_after", "replace"]
    assert _canonical_string_enum(raw, allowed) == expected


def test_canonical_string_enum_does_not_guess_semantic_alias() -> None:
    allowed = ["replace_exact", "insert_before", "insert_after", "replace"]
    assert _canonical_string_enum("edit", allowed) is None


def test_canonical_string_enum_fails_closed_on_ambiguous_normalization() -> None:
    assert _canonical_string_enum("FOO BAR", ["foo-bar", "foo_bar"]) is None


def _schema():
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


def test_parser_canonicalizes_quoted_string_enum_without_wrapper() -> None:
    call, _end = _parse_qwen_function(
        '<function=apply_source_edit><parameter=operation>"replace_exact"</parameter></function>',
        0,
        _schema(),
        call_index=0,
    )
    assert call.name == "apply_source_edit"
    assert call.arguments == {"operation": "replace_exact"}


def test_removed_whole_file_alias_is_not_canonicalized() -> None:
    with pytest.raises(RuntimeError, match="value outside enum"):
        _parse_qwen_function(
            "<function=apply_source_edit><parameter=operation>update_file</parameter></function>",
            0,
            _schema(),
            call_index=0,
        )


def test_canonical_permission_name_is_rewritten_to_exposed_alias_before_parse() -> None:
    call, _end = _parse_qwen_function(
        "<function=apply_source_patch><parameter=operation>delete_file</parameter></function>",
        0,
        _schema(),
        call_index=0,
    )
    assert call.name == "apply_source_edit"
    assert call.arguments == {"operation": "delete_file"}


def test_unrelated_unexposed_tool_is_still_rejected() -> None:
    with pytest.raises(RuntimeError, match="unexposed tool 'other_tool'"):
        _parse_qwen_function(
            "<function=other_tool></function>",
            0,
            _schema(),
            call_index=0,
        )
