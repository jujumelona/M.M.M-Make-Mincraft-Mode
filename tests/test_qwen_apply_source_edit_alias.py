from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _parse_qwen_tool_markup
from minecraft_mod_ai.model_tool_aliases import (
    canonical_model_tool,
    is_model_tool_alias,
)


def _schemas() -> dict[str, dict[str, object]]:
    return {
        "apply_source_edit": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "operation": {
                    "type": "string",
                    "enum": ["replace", "insert", "delete"],
                },
                "content": {"type": "string"},
            },
            "required": ["path", "operation"],
            "additionalProperties": False,
        }
    }


def _parse(parameters: str):
    text = (
        "<tool_call><function=apply_source_edit>"
        f"{parameters}"
        "</function></tool_call>"
    )
    return _parse_qwen_tool_markup(text, _schemas())


def test_apply_source_edit_file_alias_normalizes_to_path() -> None:
    visible, calls = _parse(
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace</parameter>"
    )

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].arguments == {
        "path": "src/main/java/example/Test.java",
        "operation": "replace",
    }


def test_apply_source_edit_apply_alias_normalizes_to_operation() -> None:
    visible, calls = _parse(
        "<parameter=path>src/A.java</parameter>"
        "<parameter=apply>replace</parameter>"
    )

    assert visible == ""
    assert calls[0].arguments == {"path": "src/A.java", "operation": "replace"}


def test_apply_source_edit_apply_object_wrapper_is_unwrapped() -> None:
    visible, calls = _parse(
        '<parameter=apply>{"file":"src/A.java","action":"replace","content":"x"}</parameter>'
    )

    assert visible == ""
    assert calls[0].arguments == {
        "path": "src/A.java",
        "operation": "replace",
        "content": "x",
    }


def test_apply_source_edit_arguments_wrapper_is_unwrapped() -> None:
    _, calls = _parse(
        '<parameter=arguments>{"target_path":"src/A.java","op":"replace"}</parameter>'
    )
    assert calls[0].arguments == {"path": "src/A.java", "operation": "replace"}


def test_nested_argument_wrappers_are_bounded_and_normalized() -> None:
    _, calls = _parse(
        '<parameter=arguments>{"params":{"file":"src/A.java","apply":"replace"}}</parameter>'
    )
    assert calls[0].arguments == {"path": "src/A.java", "operation": "replace"}


def test_apply_source_edit_rejects_file_and_path_together() -> None:
    with pytest.raises(RuntimeError, match="conflicting sources.*parameter 'path'"):
        _parse(
            "<parameter=file>src/A.java</parameter>"
            "<parameter=path>src/B.java</parameter>"
            "<parameter=operation>replace</parameter>"
        )


def test_apply_source_edit_still_rejects_unknown_parameters() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"unknown parameter 'bogus'.*allowed=.*required=.*accepted_aliases=",
    ):
        _parse(
            "<parameter=path>src/A.java</parameter>"
            "<parameter=operation>replace</parameter>"
            "<parameter=bogus>nope</parameter>"
        )


def test_unknown_nested_parameter_is_not_dropped() -> None:
    with pytest.raises(RuntimeError, match="unknown parameter 'bogus'"):
        _parse(
            '<parameter=apply>{"file":"src/A.java","action":"replace","bogus":1}</parameter>'
        )


def test_recovered_operation_still_obeys_schema_enum() -> None:
    with pytest.raises(RuntimeError, match="outside enum.*parameter 'operation'"):
        _parse(
            "<parameter=path>src/A.java</parameter>"
            "<parameter=apply>not-an-operation</parameter>"
        )


def test_patch_file_alias_resolves_only_to_exposed_source_edit() -> None:
    text = (
        "<tool_call><function=patch_file>"
        "<parameter=file>src/A.java</parameter>"
        "<parameter=action>replace</parameter>"
        "<parameter=content>x</parameter>"
        "</function></tool_call>"
    )

    visible, calls = _parse_qwen_tool_markup(text, _schemas())

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "apply_source_edit"
    assert calls[0].arguments == {
        "path": "src/A.java",
        "operation": "replace",
        "content": "x",
    }


def test_patch_file_alias_is_preserved_when_source_edit_is_not_exposed() -> None:
    text = (
        "<tool_call><function=patch_file>"
        "<parameter=file>src/A.java</parameter>"
        "<parameter=action>replace</parameter>"
        "</function></tool_call>"
    )

    visible, calls = _parse_qwen_tool_markup(text, {})
    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "patch_file"
    assert calls[0].arguments == {"file": "src/A.java", "action": "replace"}


def test_patch_file_is_not_a_permission_alias() -> None:
    assert canonical_model_tool("patch_file") == "patch_file"
    assert not is_model_tool_alias("patch_file")
