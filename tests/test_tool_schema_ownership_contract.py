from __future__ import annotations

import pytest

from minecraft_mod_ai.tool_schema_ownership_contract import (
    ToolSchemaOwnershipError,
    validate_tool_schema_surface,
)


def _schema(name: str, parameters: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


def test_composed_runtime_surface_rejects_duplicate_tool_names() -> None:
    with pytest.raises(
        ToolSchemaOwnershipError,
        match="duplicate tool schema name 'same_tool'",
    ):
        validate_tool_schema_surface(
            (_schema("same_tool"), _schema("same_tool")),
            surface="agent-runtime:generation",
        )


def test_reserved_external_namespace_requires_external_owner() -> None:
    with pytest.raises(
        ToolSchemaOwnershipError,
        match="reserved external MCP namespace",
    ):
        validate_tool_schema_surface(
            (_schema("external_mcp_call"),),
            surface="agent-runtime:frontdoor",
            reserved_external_schemas={},
        )


def test_reserved_external_schema_cannot_be_shadowed_by_first_party_shape() -> None:
    official = _schema(
        "external_mcp_call",
        {
            "type": "object",
            "required": ["capability", "arguments"],
            "properties": {
                "capability": {"type": "string"},
                "arguments": {"type": "object"},
            },
        },
    )
    shadow = _schema(
        "external_mcp_call",
        {
            "type": "object",
            "required": ["command"],
            "properties": {"command": {"type": "string"}},
        },
    )
    with pytest.raises(
        ToolSchemaOwnershipError,
        match="reserved external MCP dispatch owner",
    ):
        validate_tool_schema_surface(
            (shadow,),
            surface="agent-runtime:generation",
            reserved_external_schemas={"external_mcp_call": official},
        )


def test_final_model_facing_schema_must_match_expected_projection() -> None:
    expected = {
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["replace_exact", "insert_before", "insert_after"],
            }
        },
    }
    stale = _schema(
        "apply_source_edit",
        {
            "type": "object",
            "required": ["operation"],
            "properties": {
                "operation": {"type": "string", "enum": ["create", "replace", "edit"]}
            },
        },
    )
    with pytest.raises(
        ToolSchemaOwnershipError,
        match="does not match its final model-facing parameter contract",
    ):
        validate_tool_schema_surface(
            (stale,),
            surface="agent-runtime:generation",
            expected_parameters={"apply_source_edit": expected},
        )


def test_valid_external_and_model_projection_surface_passes_unchanged() -> None:
    external = _schema("external_mcp_schema")
    edit_parameters = {
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }
    edit = _schema("apply_source_edit", edit_parameters)
    rows = (external, edit)
    assert validate_tool_schema_surface(
        rows,
        surface="agent-runtime:generation",
        reserved_external_schemas={"external_mcp_schema": external},
        expected_parameters={"apply_source_edit": edit_parameters},
    ) == rows
