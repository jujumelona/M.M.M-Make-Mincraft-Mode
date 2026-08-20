from __future__ import annotations

import json

from minecraft_mod_ai.agent_capability_context import (
    build_agent_capability_context,
    filter_tool_schemas_for_role,
    skills_for_tool,
)
from minecraft_mod_ai.model_tool_aliases import canonical_model_tool


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_source_edit_inherits_patch_permission_without_new_policy_name() -> None:
    assert canonical_model_tool("apply_source_edit") == "apply_source_patch"
    tools = filter_tool_schemas_for_role(
        "generation",
        "coder",
        (_schema("apply_source_edit"),),
    )
    assert [item["function"]["name"] for item in tools] == ["apply_source_edit"]


def test_source_edit_does_not_escape_canonical_stage() -> None:
    tools = filter_tool_schemas_for_role(
        "quality",
        "coder_safe",
        (_schema("apply_source_edit"),),
    )
    assert tools == ()


def test_source_edit_reports_the_same_skills_as_source_patch() -> None:
    edit_skills = skills_for_tool("generation", "apply_source_edit", model_role="coder")
    patch_skills = skills_for_tool("generation", "apply_source_patch", model_role="coder")
    assert edit_skills
    assert edit_skills == patch_skills


def test_capability_context_lists_alias_as_model_tool_not_host_permission() -> None:
    prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
    rendered = build_agent_capability_context(
        "generation",
        (_schema("apply_source_edit"),),
        model_role="coder",
    )
    assert rendered.startswith(prefix)
    payload = json.loads(rendered[len(prefix) :])
    matching = [
        skill
        for skill in payload["eligible_skills"]
        if "apply_source_edit" in skill.get("model_tools", ())
    ]
    assert matching
    assert all("apply_source_patch" not in skill.get("host_owned_tools", ()) for skill in matching)
