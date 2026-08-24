from __future__ import annotations

import importlib.util
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _FakeCall:
    id: str
    name: str
    arguments: dict[str, object]
    raw_arguments: str


def _fake_module(argument: str):
    decode = object()
    seen_schemas = []

    def original_parse(text, start, schemas, *, call_index):
        del start, call_index
        seen_schemas.append(schemas)
        name_start = len("<function=")
        name_end = text.find(">", name_start)
        name = text[name_start:name_end]
        schema = schemas[name]
        operation_schema = schema["properties"]["operation"]
        assert "enum" not in operation_schema
        return (
            _FakeCall(
                id="raw",
                name=name,
                arguments={"operation": argument},
                raw_arguments="{}",
            ),
            len(text),
        )

    return (
        SimpleNamespace(
            _FUNCTION_OPEN="<function=",
            _parse_qwen_function=original_parse,
            _decode_parameter_value=decode,
        ),
        decode,
        seen_schemas,
    )


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
        }
    }


def test_install_uses_one_parser_hook_and_leaves_decoder_owner_untouched() -> None:
    fake_module, original_decode, seen_schemas = _fake_module('"replace_exact"')
    qwen_enum_recovery.install(fake_module)

    call, _end = fake_module._parse_qwen_function(
        "<function=apply_source_edit></function>",
        0,
        _schema(),
        call_index=0,
    )

    assert call.name == "apply_source_edit"
    assert call.arguments == {"operation": "replace_exact"}
    assert fake_module._decode_parameter_value is original_decode
    assert len(seen_schemas) == 1


def test_removed_whole_file_alias_is_not_canonicalized() -> None:
    fake_module, _original_decode, _seen_schemas = _fake_module("update_file")
    qwen_enum_recovery.install(fake_module)

    with pytest.raises(qwen_enum_recovery.QwenEnumValueError):
        fake_module._parse_qwen_function(
            "<function=apply_source_edit></function>",
            0,
            _schema(),
            call_index=0,
        )


def test_canonical_permission_name_is_rewritten_to_exposed_alias_before_parse() -> None:
    fake_module, _original_decode, _seen_schemas = _fake_module("delete_file")
    qwen_enum_recovery.install(fake_module)

    call, _end = fake_module._parse_qwen_function(
        "<function=apply_source_patch></function>",
        0,
        _schema(),
        call_index=0,
    )

    assert call.name == "apply_source_edit"
    assert call.arguments == {"operation": "delete_file"}


def test_unrelated_unexposed_tool_is_still_rejected_by_original_parser() -> None:
    def original_parse(text, start, schemas, *, call_index):
        del start, call_index
        name_start = len("<function=")
        name_end = text.find(">", name_start)
        name = text[name_start:name_end]
        if name not in schemas:
            raise RuntimeError(f"Qwen requested an unexposed tool {name!r}")
        raise AssertionError("unexpected exposed tool")

    fake_module = SimpleNamespace(
        _FUNCTION_OPEN="<function=",
        _parse_qwen_function=original_parse,
        _decode_parameter_value=object(),
    )
    qwen_enum_recovery.install(fake_module)

    with pytest.raises(RuntimeError, match="unexposed tool 'other_tool'"):
        fake_module._parse_qwen_function(
            "<function=other_tool></function>",
            0,
            _schema(),
            call_index=0,
        )
