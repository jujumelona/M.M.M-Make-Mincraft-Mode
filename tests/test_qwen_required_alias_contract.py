from __future__ import annotations

from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _admit_model_tool_calls
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
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


def _admit(calls):
    return _admit_model_tool_calls(
        calls,
        {"apply_source_edit": SOURCE_EDIT_SCHEMA},
        tool_choice="required",
        parallel_tool_calls=False,
    )


def test_parser_preserves_parameter_aliases_without_canonicalizing_them() -> None:
    visible, calls = _parse(
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace_exact</parameter>"
        "<parameter=old>before</parameter>"
        "<parameter=new>after</parameter>"
    )

    assert visible == ""
    assert len(calls) == 1
    assert calls[0].arguments["file"] == "src/main/java/example/Test.java"
    assert calls[0].arguments["action"] == "replace_exact"
    assert "path" not in calls[0].arguments
    assert "operation" not in calls[0].arguments


def test_parameter_aliases_are_canonicalized_only_at_admission() -> None:
    _, calls = _parse(
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace_exact</parameter>"
        "<parameter=old>before</parameter>"
        "<parameter=new>after</parameter>"
    )

    admitted = _admit(calls)
    assert len(admitted) == 1
    call = admitted[0]
    assert call.name == "apply_source_edit"
    assert call.arguments["path"] == "src/main/java/example/Test.java"
    assert call.arguments["operation"] == "replace_exact"
    assert call.arguments["old"] == "before"
    assert call.arguments["new"] == "after"
    assert "file" not in call.arguments
    assert "action" not in call.arguments


def test_legacy_semantic_operation_cannot_bypass_exposed_enum() -> None:
    _, calls = _parse(
        "<parameter=path>src/main/java/example/Test.java</parameter>"
        "<parameter=operation>create_class</parameter>"
        "<parameter=package_name>example</parameter>"
        "<parameter=declaration>public class Test</parameter>"
    )

    admitted = _admit(calls)
    assert admitted[0].name == "__mmm_rejected_tool_call__"
    assert admitted[0].arguments["failure_code"] == "TOOL_SCHEMA_INVALID"
