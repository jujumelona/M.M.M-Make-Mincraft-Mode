import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _handshake(workspace: Path, stage: str) -> set[str]:
    env = os.environ.copy()
    env["MMM_WORKSPACE"] = str(workspace)
    env["MMM_MODEL_PROFILE"] = "t4_local"
    env["MMM_MCP_STAGE"] = stage
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "minecraft_mod_ai.mcp_server"],
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            tool_names = {tool.name for tool in tools.tools}
            resource_uris = {str(item.uri) for item in resources.resources}
            assert "mmm://model-registry" in resource_uris
            assert "mmm://plugins" in resource_uris
            return tool_names


def test_real_stdio_frontdoor_hides_specialist_mutators(tmp_path: Path) -> None:
    tools = asyncio.run(_handshake(tmp_path / "workspace", "frontdoor"))
    assert "discover_mmm_capabilities" in tools
    assert "plan_complete_game" in tools
    assert "revise_complete_plan" in tools
    assert "read_complete_plan_section" not in tools
    assert "plan_game" not in tools
    assert "work_status" in tools
    assert "quality_status" in tools
    assert "work_cancel_run" in tools
    assert "work_resume_run" in tools
    assert "execute_complete_project" not in tools
    assert "build_technology_radar" in tools
    assert "inspect_huggingface_model" not in tools
    assert "assess_technology_compatibility" not in tools
    assert "run_gametest" not in tools
    assert "runtime_start_server" not in tools


def test_real_stdio_quality_stage_is_narrow(tmp_path: Path) -> None:
    tools = asyncio.run(_handshake(tmp_path / "workspace", "quality"))
    assert "discover_mmm_capabilities" in tools
    assert "run_gametest" in tools
    assert "quality_status" in tools
    assert "read_quality_contract" in tools
    assert "java_diagnostics" in tools
    assert "plan_game" not in tools
    assert "runtime_start_server" not in tools
    assert "build_technology_radar" not in tools
    assert "inspect_huggingface_model" not in tools
    assert "assess_technology_compatibility" not in tools


def test_real_stdio_planning_stage_exposes_paged_complete_plan(
    tmp_path: Path,
) -> None:
    tools = asyncio.run(_handshake(tmp_path / "workspace", "planning"))
    assert "plan_complete_game" in tools
    assert "read_complete_plan_section" in tools
    assert "read_quality_contract" in tools
    assert "approve_complete_plan" in tools
    assert "execute_complete_project" not in tools
    assert "build_technology_radar" in tools
    assert "inspect_huggingface_model" in tools
    assert "assess_technology_compatibility" in tools


def test_real_stdio_generation_stage_exposes_coder_evidence_tools(tmp_path: Path) -> None:
    tools = asyncio.run(_handshake(tmp_path / "workspace", "generation"))
    assert "inspect_existing_mod" in tools
    assert "search_project_rag" in tools
    assert "search_code_rag" in tools
    assert "runtime_start_server" not in tools
    assert "package_release" not in tools
    assert "plan_complete_game" not in tools
