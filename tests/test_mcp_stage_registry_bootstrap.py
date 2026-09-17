from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import anyio

from minecraft_mod_ai.mcp_child_trace_contract import traced_stdio_session


ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "minecraft_mod_ai" / "mcp_server.py"


def _registry_and_decorated_tool_names() -> tuple[set[str], set[str]]:
    tree = ast.parse(MCP_SERVER.read_text(encoding="utf-8"), filename=str(MCP_SERVER))
    registry: set[str] | None = None
    decorated: set[str] = set()

    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_TOOL_STAGES"
            and isinstance(node.value, ast.Dict)
        ):
            registry = {
                str(ast.literal_eval(key))
                for key in node.value.keys
                if key is not None
            }

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "_stage_tool"
                ):
                    decorated.add(node.name)

    assert registry is not None, "mcp_server.py must define _TOOL_STAGES as a dict literal"
    return registry, decorated


def test_mcp_stage_registry_exactly_matches_stage_tool_decorators() -> None:
    registry, decorated = _registry_and_decorated_tool_names()
    assert decorated == registry, (
        "MCP stage registry/decorator mismatch: "
        f"missing decorators={sorted(registry - decorated)}; "
        f"unexpected decorators={sorted(decorated - registry)}"
    )


def test_generation_stage_mcp_server_imports_in_clean_child_process() -> None:
    env = os.environ.copy()
    env["MMM_MCP_STAGE"] = "generation"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import minecraft_mod_ai.mcp_server as server; "
                "assert 'search_project_rag' in server._TOOL_STAGES; "
                "assert '_normalized_minecraft_version' not in server._TOOL_STAGES"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        "generation-stage MCP child import failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_generation_stage_mcp_stdio_initializes_and_lists_tools(tmp_path: Path) -> None:
    async def exercise() -> None:
        env = os.environ.copy()
        env["MMM_MCP_STAGE"] = "generation"
        env["MMM_WORKSPACE"] = str(tmp_path)
        async with traced_stdio_session(
            "generation",
            env,
            timeout_seconds=30.0,
        ) as session:
            listed = await session.list_tools()
            names = {
                str(getattr(tool, "name", ""))
                for tool in (getattr(listed, "tools", ()) or ())
            }
            assert "search_project_rag" in names
            assert "apply_source_patch" in names
            assert "java_diagnostics" in names
            assert "_normalized_minecraft_version" not in names

    anyio.run(exercise)
