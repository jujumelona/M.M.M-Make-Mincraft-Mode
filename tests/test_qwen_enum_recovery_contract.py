from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "minecraft_mod_ai"
    / "qwen_enum_recovery_contract.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_enum_recovery_contract_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qwen_enum_recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen_enum_recovery)


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
    assert qwen_enum_recovery.canonical_string_enum(raw, allowed) == expected


def test_canonical_string_enum_does_not_guess_semantic_alias() -> None:
    allowed = ["replace_exact", "insert_before", "insert_after", "replace"]
    assert (
        qwen_enum_recovery.canonical_string_enum("edit", allowed)
        is qwen_enum_recovery._NO_MATCH
    )


def test_canonical_string_enum_fails_closed_on_ambiguous_normalization() -> None:
    assert (
        qwen_enum_recovery.canonical_string_enum("FOO BAR", ["foo-bar", "foo_bar"])
        is qwen_enum_recovery._NO_MATCH
    )


def _outside_enum_error(tool_name: str, key: str) -> RuntimeError:
    return RuntimeError(
        f"Qwen tool {tool_name!r} emitted value outside enum for parameter {key!r}"
    )


def test_install_canonicalizes_quoted_string_before_original_decoder() -> None:
    def original_decode(tool_name, key, raw, schema):
        if raw not in schema["enum"]:
            raise _outside_enum_error(tool_name, key)
        return raw

    def original_completion(adapter, server_url, request):
        return "unused"

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    assert (
        fake_module._decode_parameter_value(
            "apply_source_edit",
            "operation",
            '"replace_exact"',
            {"type": "string", "enum": ["replace_exact", "insert_before"]},
        )
        == "replace_exact"
    )
    assert fake_module._tool_semantic_completion is original_completion


def test_tool_specific_alias_is_canonicalized_without_generation_retry() -> None:
    def original_decode(tool_name, key, raw, schema):
        if raw not in schema["enum"]:
            raise _outside_enum_error(tool_name, key)
        return raw

    def original_completion(adapter, server_url, request):
        return "unchanged"

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    assert (
        fake_module._decode_parameter_value(
            "apply_source_edit",
            "operation",
            "update_file",
            {"type": "string", "enum": ["replace_file", "delete_file"]},
        )
        == "replace_file"
    )
    assert fake_module._tool_semantic_completion is original_completion


def test_invalid_semantic_enum_becomes_typed_protocol_error_without_retry() -> None:
    def original_decode(tool_name, key, raw, schema):
        raise _outside_enum_error(tool_name, key)

    calls = 0

    def original_completion(adapter, server_url, request):
        nonlocal calls
        calls += 1
        return "not-owned-here"

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    with pytest.raises(qwen_enum_recovery.QwenEnumValueError) as captured:
        fake_module._decode_parameter_value(
            "apply_source_edit",
            "operation",
            "edit",
            {
                "type": "string",
                "enum": ["replace_exact", "insert_before", "insert_after", "replace"],
            },
        )

    assert captured.value.tool_name == "apply_source_edit"
    assert captured.value.parameter_name == "operation"
    assert captured.value.raw_value == "edit"
    assert calls == 0
    assert fake_module._tool_semantic_completion is original_completion


def test_unknown_parameter_remains_parser_error_for_causal_recovery_owner() -> None:
    def original_decode(tool_name, key, raw, schema):
        raise RuntimeError(
            f"Qwen tool {tool_name!r} emitted unknown parameter {key!r}"
        )

    def original_completion(adapter, server_url, request):
        return "unchanged"

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    with pytest.raises(RuntimeError, match="unknown parameter"):
        fake_module._decode_parameter_value(
            "apply_source_edit",
            "after_line",
            "12",
            {"type": "integer"},
        )
    assert fake_module._tool_semantic_completion is original_completion
