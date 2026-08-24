from __future__ import annotations

import minecraft_mod_ai.model_adapters as model_adapters
from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup


def test_llama_adapter_imports_the_canonical_qwen_parser() -> None:
    assert llama_cpp_adapter._parse_qwen_tool_markup is parse_qwen_tool_markup
    assert (
        llama_cpp_adapter._parse_qwen_tool_markup.__module__
        == "minecraft_mod_ai.model_adapters.qwen_tool_parser"
    )


def test_llama_adapter_has_no_duplicate_qwen_parser_helpers() -> None:
    duplicate_helpers = (
        "_parse_qwen_function",
        "_find_parameter_close",
        "_unwrap_parameter_text",
        "_decode_parameter_value",
        "_schema_value_type",
        "_json_type",
        "_skip_space",
    )
    assert not [name for name in duplicate_helpers if hasattr(llama_cpp_adapter, name)]


def test_model_adapters_init_has_no_parser_monkey_patch_surface() -> None:
    assert not hasattr(model_adapters, "parse_qwen_tool_markup")
    assert not hasattr(model_adapters, "_llama_cpp_adapter")
