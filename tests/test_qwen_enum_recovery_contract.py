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
class _Request:
    messages: tuple[dict[str, str], ...] = ()
    parallel_tool_calls: bool = True


def _outside_enum_error(tool_name: str, key: str) -> RuntimeError:
    return RuntimeError(
        f"Qwen tool {tool_name!r} emitted value outside enum for parameter {key!r}"
    )


def test_install_canonicalizes_quoted_string_before_original_decoder() -> None:
    def original_decode(tool_name, key, raw, schema):
        if raw not in schema["enum"]:
            raise _outside_enum_error(tool_name, key)
        return raw

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=lambda adapter, server_url, request: "unused",
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


def test_invalid_semantic_enum_is_discarded_and_retried_once() -> None:
    calls: list[_Request] = []

    def original_decode(tool_name, key, raw, schema):
        raise _outside_enum_error(tool_name, key)

    def original_completion(adapter, server_url, request):
        calls.append(request)
        if len(calls) == 1:
            raise qwen_enum_recovery.QwenEnumValueError(
                tool_name="apply_source_edit",
                parameter_name="operation",
                raw_value="edit",
                allowed_values=["replace_exact", "insert_before", "insert_after", "replace"],
            )
        return "corrected"

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    request = _Request(messages=({"role": "user", "content": "repair"},))
    assert fake_module._tool_semantic_completion(None, "http://local", request) == "corrected"
    assert len(calls) == 2
    assert calls[1].parallel_tool_calls is False
    assert calls[1].messages[-1]["role"] == "system"
    assert "discarded without execution" in calls[1].messages[-1]["content"]
    assert "replace_exact" in calls[1].messages[-1]["content"]


def test_second_invalid_enum_fails_after_one_retry() -> None:
    attempts = 0

    def original_decode(tool_name, key, raw, schema):
        raise _outside_enum_error(tool_name, key)

    def original_completion(adapter, server_url, request):
        nonlocal attempts
        attempts += 1
        raise qwen_enum_recovery.QwenEnumValueError(
            tool_name="apply_source_edit",
            parameter_name="operation",
            raw_value="edit",
            allowed_values=["replace_exact", "insert_before", "insert_after", "replace"],
        )

    fake_module = SimpleNamespace(
        _decode_parameter_value=original_decode,
        _tool_semantic_completion=original_completion,
    )
    qwen_enum_recovery.install(fake_module)

    with pytest.raises(RuntimeError, match="after one bounded corrective retry"):
        fake_module._tool_semantic_completion(None, "http://local", _Request())
    assert attempts == 2
