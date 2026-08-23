from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _parse_qwen_tool_markup


def _schemas() -> dict[str, dict[str, object]]:
    return {
        "apply_source_edit": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "operation": {"type": "string"},
            },
            "required": ["path", "operation"],
            "additionalProperties": False,
        }
    }


def test_apply_source_edit_file_alias_normalizes_to_path() -> None:
    text = (
        "<tool_call><function=apply_source_edit>"
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace</parameter>"
        "</function></tool_call>"
    )

    visible, calls = _parse_qwen_tool_markup(text, _schemas())

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].arguments == {
        "path": "src/main/java/example/Test.java",
        "operation": "replace",
    }


def test_apply_source_edit_rejects_file_and_path_together() -> None:
    text = (
        "<function=apply_source_edit>"
        "<parameter=file>src/A.java</parameter>"
        "<parameter=path>src/B.java</parameter>"
        "<parameter=operation>replace</parameter>"
        "</function>"
    )

    with pytest.raises(RuntimeError, match="both alias 'file'.*canonical parameter 'path'"):
        _parse_qwen_tool_markup(text, _schemas())


def test_apply_source_edit_still_rejects_unknown_parameters() -> None:
    text = (
        "<function=apply_source_edit>"
        "<parameter=path>src/A.java</parameter>"
        "<parameter=operation>replace</parameter>"
        "<parameter=bogus>nope</parameter>"
        "</function>"
    )

    with pytest.raises(RuntimeError, match="unknown parameter 'bogus'"):
        _parse_qwen_tool_markup(text, _schemas())
