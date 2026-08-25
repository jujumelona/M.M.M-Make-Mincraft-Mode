import pytest

from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup


def _schema(*, include_workspace_root: bool = False):
    properties = {
        "operation": {"type": "string"},
        "path": {"type": "string"},
    }
    if include_workspace_root:
        properties["workspace_root"] = {"type": "string"}
    return {
        "apply_source_edit": {
            "type": "object",
            "properties": properties,
            "required": ["operation", "path"],
            "additionalProperties": False,
        }
    }


def test_workspace_root_is_ignored_when_host_owned():
    _, calls = parse_qwen_tool_markup(
        "<tool_call><function=apply_source_edit>"
        "<parameter=operation>replace</parameter>"
        "<parameter=path>src/Main.java</parameter>"
        "<parameter=workspace_root>/tmp/workspace</parameter>"
        "</function></tool_call>",
        _schema(),
    )

    assert len(calls) == 1
    assert calls[0].arguments == {
        "operation": "replace",
        "path": "src/Main.java",
    }


def test_nested_workspace_root_is_ignored_when_host_owned():
    _, calls = parse_qwen_tool_markup(
        "<tool_call><function=apply_source_edit>"
        '<parameter=arguments>{"operation":"replace","path":"src/Main.java",'
        '"workspace_root":"/tmp/workspace"}</parameter>'
        "</function></tool_call>",
        _schema(),
    )

    assert calls[0].arguments == {
        "operation": "replace",
        "path": "src/Main.java",
    }


def test_unrelated_unknown_parameter_remains_strict_error():
    with pytest.raises(RuntimeError, match="unknown parameter 'random_root'"):
        parse_qwen_tool_markup(
            "<tool_call><function=apply_source_edit>"
            "<parameter=operation>replace</parameter>"
            "<parameter=path>src/Main.java</parameter>"
            "<parameter=random_root>/tmp/workspace</parameter>"
            "</function></tool_call>",
            _schema(),
        )


def test_workspace_root_is_preserved_when_schema_declares_it():
    _, calls = parse_qwen_tool_markup(
        "<tool_call><function=apply_source_edit>"
        "<parameter=operation>replace</parameter>"
        "<parameter=path>src/Main.java</parameter>"
        "<parameter=workspace_root>/tmp/workspace</parameter>"
        "</function></tool_call>",
        _schema(include_workspace_root=True),
    )

    assert calls[0].arguments == {
        "operation": "replace",
        "path": "src/Main.java",
        "workspace_root": "/tmp/workspace",
    }
