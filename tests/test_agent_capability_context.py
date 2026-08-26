from __future__ import annotations

import json

from minecraft_mod_ai.agent_capability_context import (
    build_agent_capability_context,
    filter_tool_schemas_for_role,
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


def _tool_names(schemas) -> set[str]:
    return {str(item["function"]["name"]) for item in schemas}


def _external_proxy_schemas():
    return (
        _schema("external_mcp_capabilities"),
        _schema("external_mcp_schema"),
        _schema("external_mcp_call"),
    )


def test_research_context_connects_role_skills_tools_and_external_mcp() -> None:
    context = _decode_context(
        build_agent_capability_context(
            "research",
            (
                _schema("search_code_rag"),
                *_external_proxy_schemas(),
            ),
            model_role="researcher",
        )
    )

    assert context["agent_roles"] == ["ResearchAgent"]
    assert "minecraft-dev" in context["reviewed_mcp_servers"]
    assert "minecraft-wiki" in context["reviewed_mcp_servers"]

    skills = {item["name"]: item for item in context["eligible_skills"]}
    evidence = skills["gather-adaptive-minecraft-evidence"]
    assert "search_code_rag" in evidence["model_tools"]
    assert evidence["activate_when"]
    assert evidence["validators"]
    assert set(evidence["retry"]) >= {
        "max_attempts",
        "strategy",
        "stop_on_repeated_error_signature",
        "require_fresh_evidence",
    }
    assert "writes" in evidence["approvals"]
    assert evidence["forbidden_actions"]
    assert set(evidence["exit"]) == {"success", "blocked", "failed"}

    capabilities = context["external_minecraft_mcp_capabilities"]
    access = context["external_minecraft_mcp_access"]
    assert capabilities["official_mod_docs"] == ["mcmodding-docs"]
    assert capabilities["source_search"] == ["minecraft-dev"]
    assert "minecraft-dev" in capabilities["mapping_resolution"]
    assert capabilities["vanilla_knowledge"] == ["minecraft-wiki"]
    assert access["source_search"]["minecraft-dev"] == "read"
    assert "runtime_command" not in capabilities
    assert "server_rcon" not in capabilities
    assert "player_e2e_interact" not in capabilities


def test_resident_planner_research_turn_uses_research_agent_policy() -> None:
    schemas = (
        _schema("search_code_rag"),
        _schema("java_diagnostics"),
        *_external_proxy_schemas(),
    )
    filtered = filter_tool_schemas_for_role("research", "planner", schemas)
    names = _tool_names(filtered)
    assert "search_code_rag" in names
    assert "java_diagnostics" not in names

    context = _decode_context(
        build_agent_capability_context(
            "research",
            filtered,
            model_role="planner",
        )
    )
    assert context["execution_model_role"] == "planner"
    assert context["model_role"] == "researcher"
    assert context["agent_roles"] == ["ResearchAgent"]

    skills = {item["name"]: item for item in context["eligible_skills"]}
    assert "gather-adaptive-minecraft-evidence" in skills
    for skill in skills.values():
        assert "java_diagnostics" not in skill["model_tools"]
        assert "java_diagnostics" not in skill["host_owned_tools"]
    assert "gather-adaptive-minecraft-evidence" in skills_for_tool(
        "research",
        "search_code_rag",
        model_role="planner",
    )


def test_runtime_context_discovers_gated_write_and_admin_minecraft_mcp_routes() -> None:
    context = _decode_context(
        build_agent_capability_context(
            "runtime",
            _external_proxy_schemas(),
            model_role="coder_safe",
        )
    )

    assert "RuntimeTester" in context["agent_roles"]
    capabilities = context["external_minecraft_mcp_capabilities"]
    access = context["external_minecraft_mcp_access"]

    assert "minecraft-player-agent" in capabilities["player_e2e_interact"]
    assert access["player_e2e_interact"]["minecraft-player-agent"] == "write"
    assert "fabric-game-runtime" in capabilities["runtime_command"]
    assert access["runtime_command"]["fabric-game-runtime"] == "admin"
    assert "minecraft-rcon" in capabilities["server_rcon"]
    assert access["server_rcon"]["minecraft-rcon"] == "admin"
    assert "disposable_runtime=true" in context["routing_policy"]


def test_role_filter_removes_unassigned_tools_but_keeps_external_bridge() -> None:
    schemas = (
        _schema("search_code_rag"),
        _schema("java_diagnostics"),
        *_external_proxy_schemas(),
    )
    filtered = filter_tool_schemas_for_role("research", "researcher", schemas)
    names = _tool_names(filtered)
    assert "search_code_rag" in names
    assert "java_diagnostics" not in names
    assert {
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    } <= names

    assert filter_tool_schemas_for_role("research", "unknown-role", schemas) == schemas


def test_every_canonical_skill_is_reachable_in_at_least_one_stage_context() -> None:
    reachable: set[str] = set()
    schemas = tuple(_schema(name) for name in (
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
    ))
    for stage in REVIEWED_STAGES:
        context = _decode_context(build_agent_capability_context(stage, schemas))
        reachable.update(item["name"] for item in context["eligible_skills"])

    assert reachable == set(CANONICAL_SKILLS)


def test_tool_receipt_is_role_scoped() -> None:
    routes = skills_for_tool(
        "research",
        "search_code_rag",
        model_role="researcher",
    )
    assert "gather-adaptive-minecraft-evidence" in routes
    assert skills_for_tool(
        "research",
        "java_diagnostics",
        model_role="researcher",
    ) == ()
    assert skills_for_tool(
        "research",
        "external_mcp_call",
        model_role="researcher",
    ) == ()
