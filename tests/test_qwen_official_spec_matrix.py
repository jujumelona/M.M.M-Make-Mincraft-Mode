from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
from minecraft_mod_ai.qwen_family_capabilities import (
    _OFFICIAL_CAPABILITIES,
    qwen_family_capabilities,
)


def test_qwen_official_capabilities_spec_matrix() -> None:
    """Verify official capabilities matrix for Qwen 3.5, 3.6, and 3.8."""
    c35 = _OFFICIAL_CAPABILITIES["qwen3.5"]
    assert c35.family == "qwen3.5"
    assert c35.tool_markup == "qwen3_coder_xml"
    assert c35.action_thinking_control == "enable_thinking_false"
    assert c35.preserve_thinking is False
    assert c35.reasoning_effort is False
    assert c35.assistant_prefill is True
    assert c35.action_template_kwargs() == {"enable_thinking": False}

    c36 = _OFFICIAL_CAPABILITIES["qwen3.6"]
    assert c36.family == "qwen3.6"
    assert c36.preserve_thinking is True
    assert c36.reasoning_effort is False
    assert c36.assistant_prefill is True
    assert c36.action_template_kwargs() == {"enable_thinking": False, "preserve_thinking": False}

    c38 = _OFFICIAL_CAPABILITIES["qwen3.8"]
    assert c38.family == "qwen3.8"
    assert c38.preserve_thinking is True
    assert c38.reasoning_effort is True
    assert c38.assistant_prefill is True
    assert c38.action_template_kwargs() == {"enable_thinking": False, "preserve_thinking": False}


def test_qwen_tool_markup_xml_parser() -> None:
    """Verify parsing of Qwen XML format tool calls."""
    schemas = {
        "apply_source_edit": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["operation", "path"],
        }
    }

    raw = (
        "I will apply the requested edit now.\n"
        "<tool_call>\n"
        '<function=apply_source_edit>\n'
        '<parameter=operation>\ncreate\n</parameter>\n'
        '<parameter=path>\nsrc/main/java/Example.java\n</parameter>\n'
        '<parameter=content>\npublic class Example {}\n</parameter>\n'
        '</function>\n'
        '</tool_call>'
    )

    visible, calls = parse_qwen_tool_markup(raw, schemas)
    assert "I will apply the requested edit now." in visible
    assert len(calls) == 1
    call = calls[0]
    assert call.name == "apply_source_edit"
    assert call.arguments["operation"] == "create"
    assert call.arguments["path"] == "src/main/java/Example.java"
    assert "public class Example {}" in call.arguments["content"]
