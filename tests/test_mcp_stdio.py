import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _handshake(workspace: Path) -> None:
    env = os.environ.copy()
    env["MMM_WORKSPACE"] = str(workspace)
    env["MMM_MODEL_PROFILE"] = "t4_local"
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
            assert "plan_game" in tool_names
            assert "run_gametest" in tool_names
            assert "mmm://model-registry" in resource_uris
            assert "mmm://plugins" in resource_uris


def test_real_stdio_initialize_tools_and_resources(tmp_path: Path) -> None:
    asyncio.run(_handshake(tmp_path / "workspace"))
