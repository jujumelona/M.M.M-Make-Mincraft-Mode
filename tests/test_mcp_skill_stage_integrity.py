from __future__ import annotations

from minecraft_mod_ai import mcp_server
from minecraft_mod_ai.model_tool_aliases import (
    canonical_model_tool,
    resolve_exposed_model_tool,
)
from minecraft_mod_ai.skill_catalog import REVIEWED_TOOL_STAGES


def _server_stages(name: str) -> frozenset[str]:
    # ``all`` is a host-only debug aggregation mode, never a Skill execution stage.
    return frozenset(mcp_server._TOOL_STAGES[name] - {"all"})


def test_first_party_mcp_and_skill_policy_share_exact_tool_names() -> None:
    assert set(mcp_server._TOOL_STAGES) == set(REVIEWED_TOOL_STAGES)


def test_first_party_mcp_and_skill_policy_share_exact_stage_assignments() -> None:
    mismatches = {
        name: (_server_stages(name), REVIEWED_TOOL_STAGES[name])
        for name in sorted(REVIEWED_TOOL_STAGES)
        if _server_stages(name) != REVIEWED_TOOL_STAGES[name]
    }
    assert mismatches == {}


def test_model_tool_aliases_never_create_new_server_permissions() -> None:
    assert canonical_model_tool("apply_source_edit") == "apply_source_patch"
    assert "apply_source_edit" not in mcp_server._TOOL_STAGES
    assert canonical_model_tool("apply_source_edit") in REVIEWED_TOOL_STAGES


def test_canonical_permission_name_resolves_only_through_current_exposure() -> None:
    assert (
        resolve_exposed_model_tool("apply_source_patch", ("apply_source_edit",))
        == "apply_source_edit"
    )
    assert (
        resolve_exposed_model_tool("apply_source_patch", ("search_code_rag",))
        is None
    )
