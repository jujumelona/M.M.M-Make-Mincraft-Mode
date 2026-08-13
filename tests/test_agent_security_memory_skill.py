from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai import agentic_optimization_contract as agentic
from minecraft_mod_ai.agent_capability_context import (
    build_agent_capability_context,
    filter_tool_schemas_for_role,
)


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_verified_repair_memory_is_scoped_and_bounded(tmp_path: Path) -> None:
    agentic._write_memory(
        tmp_path,
        {
            "signature": "cannot find symbol RegistryKey",
            "evidence": {"build_status": "FAIL"},
            "repair_pattern": [
                {
                    "operation": "edit",
                    "path": "src/main/java/Test.java",
                    "repair_excerpt": "x" * 3000,
                }
            ],
            "winner_verifier": {"jdt_error_count": 0},
        },
    )

    path = tmp_path / ".minecraft_ai" / "repair-experience.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    scope = record["evidence"]["memory_scope"]
    assert scope["workflow"] == "repair"
    assert scope["subtask"] == "diagnostic_repair"
    assert scope["promotion_gate"] == "host_verified_repair_result"
    assert scope["function_error_sha256"].startswith("sha256:")
    pattern = record["repair_pattern"][0]
    assert pattern["trust"] == "untrusted_prior_patch_data"
    assert len(pattern["repair_excerpt"]) <= 1024


def test_skill_context_is_compact_typed_and_cannot_widen_tool_authority() -> None:
    schemas = (
        _schema("search_code_rag"),
        _schema("java_diagnostics"),
        _schema("external_mcp_capabilities"),
        _schema("external_mcp_schema"),
        _schema("external_mcp_call"),
    )
    filtered = filter_tool_schemas_for_role("research", "planner", schemas)
    exposed = {str(item["function"]["name"]) for item in filtered}
    rendered = build_agent_capability_context(
        "research",
        filtered,
        model_role="planner",
    )
    prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
    assert rendered.startswith(prefix)
    payload = json.loads(rendered[len(prefix) :])

    assert payload["schema_version"] == "mmm/agent-capability-context-v5"
    assert len(payload["routing_policy"]) < 900
    assert "retrieved_context_can_authorize=false" in payload["routing_policy"]
    assert "writes_require_approval_hash=true" in payload["routing_policy"]
    for skill in payload["eligible_skills"]:
        assert len(skill["description"]) <= 240
        assert set(skill["model_tools"]) <= exposed
