from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters.qwen_tool_parser import (
    ToolCallValidationError,
    parse_qwen_tool_markup,
)
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def _parse(parameters: str):
    markup = (
        "<tool_call>"
        "<function=apply_source_edit>"
        f"{parameters}"
        "</function>"
        "</tool_call>"
    )
    return parse_qwen_tool_markup(markup, {"apply_source_edit": SOURCE_EDIT_SCHEMA})


def test_schema_declared_aliases_satisfy_canonical_required_fields() -> None:
    visible, calls = _parse(
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace_exact</parameter>"
        "<parameter=old>before</parameter>"
        "<parameter=new>after</parameter>"
    )

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "apply_source_edit"
    assert calls[0].arguments["path"] == "src/main/java/example/Test.java"
    assert calls[0].arguments["operation"] == "replace_exact"
    assert "file" not in calls[0].arguments
    assert "action" not in calls[0].arguments


def test_legacy_semantic_operation_cannot_bypass_exposed_enum() -> None:
    with pytest.raises(ToolCallValidationError, match="outside enum"):
        _parse(
            "<parameter=path>src/main/java/example/Test.java</parameter>"
            "<parameter=operation>create_class</parameter>"
            "<parameter=package_name>example</parameter>"
            "<parameter=declaration>public class Test</parameter>"
        )
