from __future__ import annotations

import inspect

from minecraft_mod_ai import research_code_context
from minecraft_mod_ai.model_adapters import qwen_tool_parser
from minecraft_mod_ai.source_edit_scalar_protocol_contract import (
    SOURCE_EDIT_PARAMETER_ALIASES,
    SOURCE_EDIT_SCHEMA,
)


def test_dependency_query_has_one_runtime_owner() -> None:
    wrapped = research_code_context.ResearchCodeContext._query_paths
    base = getattr(wrapped, "__wrapped__", None)
    assert base is not None
    source = inspect.getsource(base)
    assert "research_coder_repair_reuse" not in source
    assert "_dependency_neighborhood_query" not in source


def test_transport_aliases_do_not_duplicate_scalar_protocol_aliases() -> None:
    properties = set(SOURCE_EDIT_SCHEMA["properties"])
    transport = set(qwen_tool_parser._APPLY_SOURCE_EDIT_TRANSPORT_ALIASES)
    scalar_aliases = set(SOURCE_EDIT_PARAMETER_ALIASES)
    assert transport.isdisjoint(properties)
    assert scalar_aliases <= properties
    assert scalar_aliases.isdisjoint(transport)
    assert all(
        qwen_tool_parser._APPLY_SOURCE_EDIT_ALIASES[alias] == canonical
        for alias, canonical in SOURCE_EDIT_PARAMETER_ALIASES.items()
    )
