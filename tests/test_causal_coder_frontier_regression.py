from __future__ import annotations

from typing import Any

from minecraft_mod_ai.causal_tool_frontier_contract import goals_for_query
from minecraft_mod_ai.causal_tool_graph import executable_frontier


def _schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_coder_outcome_outranks_external_mcp_as_implementation_means() -> None:
    query = (
        "Implement the approved Minecraft/Fabric feature in the current project. "
        "Use external MCP capabilities only when they provide useful evidence."
    )

    assert goals_for_query(query) == ("act",)
    assert goals_for_query("Inspect external MCP capabilities for this project.") == (
        "external",
    )


def test_coder_frontier_reaches_real_source_patch_instead_of_external_only() -> None:
    query = (
        "Implement the approved module. Inspect evidence first and apply a real source "
        "patch; external MCP tools are supporting tools, not the final task."
    )
    schemas = (
        _schema("search_code_rag"),
        _schema("apply_source_patch"),
        _schema("external_mcp_call"),
    )

    initial = executable_frontier(
        schemas,
        state=frozenset({"workspace_bound"}),
        goals=goals_for_query(query),
        limit=3,
        max_depth=8,
    )
    assert initial == ("search_code_rag",)

    after_evidence = executable_frontier(
        schemas,
        state=frozenset(
            {
                "workspace_bound",
                "project_observed",
                "code_evidence",
                "evidence_ready",
            }
        ),
        goals=goals_for_query(query),
        limit=3,
        max_depth=8,
    )
    assert after_evidence == ("apply_source_patch",)
