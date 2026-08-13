from __future__ import annotations

import json

from minecraft_mod_ai.agent_capability_context import (
    build_agent_capability_context,
    skills_for_tool,
)
from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS, REVIEWED_STAGES


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _decode_context(text: str) -> dict[str, object]:
    prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
    assert text.startswith(prefix)
    return json.loads(text[len(prefix) :])


def test_research_context_connects_skills_tools_and_external_mcp() -> None:
    context = _decode_context(
        build_agent_capability_context(
            "research",
            (
                _schema("search_code_rag"),
                _schema("external_mcp_capabilities"),
                _schema("external_mcp_schema"),
                _schema("external_mcp_call"),
            ),
        )
    )

    skills = {item["name"]: item for item in context["eligible_skills"]}
    assert "gather-adaptive-minecraft-evidence" in skills
    assert "search_code_rag" in skills["gather-adaptive-minecraft-evidence"]["model_tools"]

    capabilities = set(context["external_minecraft_mcp_capabilities"])
    assert "official_mod_docs" in capabilities
    assert "source_search" in capabilities
    assert "mapping_resolution" in capabilities
    assert "vanilla_knowledge" in capabilities


def test_every_canonical_skill_is_reachable_in_at_least_one_stage_context() -> None:
    reachable: set[str] = set()
    schemas = tuple(_schema(name) for name in {
        "discover_mmm_capabilities",
        "plan_game",
        "plan_complete_game",
        "revise_plan",
        "revise_complete_plan",
        "approve_plan",
        "approve_complete_plan",
        "read_complete_plan_section",
        "read_quality_contract",
        "quality_status",
        "discover_ecosystem_resources",
        "inspect_modrinth_project",
        "inspect_github_repository",
        "inspect_huggingface_model",
        "build_technology_radar",
        "assess_technology_compatibility",
        "search_project_rag",
        "search_code_rag",
        "index_project_rag",
        "inspect_existing_mod",
        "work_status",
        "work_tasks",
        "work_cancel_run",
        "work_resume_run",
        "execute_complete_project",
        "generate_fabric_project",
        "generate_assets",
        "generate_geckolib_entity",
        "generate_system_plugin",
        "apply_source_patch",
        "repair_project",
        "java_diagnostics",
        "java_workspace_symbols",
        "blockbench_list_tools",
        "blockbench_execute",
        "run_static_validation",
        "run_gradle_build",
        "run_gametest",
        "inspect_jar",
        "runtime_prepare_instance",
        "runtime_start_server",
        "runtime_start_client",
        "runtime_send_command",
        "runtime_logs",
        "runtime_register_screenshot",
        "runtime_status",
        "runtime_stop",
        "mineflayer_connect",
        "mineflayer_status",
        "mineflayer_walk_to",
        "mineflayer_interact_block",
        "mineflayer_inventory",
        "mineflayer_disconnect",
        "package_release",
        "run_model_smoke",
        "record_training_trace",
        "export_training_dataset",
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    })
    for stage in REVIEWED_STAGES:
        context = _decode_context(build_agent_capability_context(stage, schemas))
        reachable.update(item["name"] for item in context["eligible_skills"])

    assert reachable == set(CANONICAL_SKILLS)


def test_tool_receipt_can_identify_all_skill_routes_for_tool() -> None:
    routes = skills_for_tool("research", "search_code_rag")
    assert "gather-adaptive-minecraft-evidence" in routes
    assert skills_for_tool("research", "external_mcp_call") == ()
