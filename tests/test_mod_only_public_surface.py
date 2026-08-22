from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai import mcp_server
from minecraft_mod_ai.mcp_tools import MMMToolService
from minecraft_mod_ai.skill_catalog import MUTATING_TOOLS, REVIEWED_TOOL_STAGES


ROOT = Path(__file__).resolve().parents[1]
REMOVED = {"generate_world_ir", "compile_world_ir"}


def test_primary_mcp_has_no_standalone_map_tools() -> None:
    assert REMOVED.isdisjoint(mcp_server._TOOL_STAGES)
    assert REMOVED.isdisjoint(mcp_server._tool_names_for_stage("all"))


def test_primary_service_has_no_removed_map_compiler_surface() -> None:
    assert not hasattr(MMMToolService, "generate_world_ir")


def test_obsolete_compatibility_entrypoints_are_physically_removed() -> None:
    assert not (ROOT / "mcp_gateway.py").exists()
    assert not (ROOT / "colab_app.py").exists()


def test_skill_policy_physically_excludes_removed_tools() -> None:
    assert REMOVED.isdisjoint(REVIEWED_TOOL_STAGES)
    assert REMOVED.isdisjoint(MUTATING_TOOLS)
    packaged = json.loads(
        (ROOT / "minecraft_mod_ai/packaged_skills.json").read_text(
            encoding="utf-8"
        )
    )
    selected = "\n".join(
        packaged["skills"][name]
        for name in ("plan-game-design", "generate-worldgen")
    )
    assert "generate_world_ir" not in selected
    assert "compile_world_ir" not in selected


def test_all_generation_registries_use_mod_only_server() -> None:
    root_config = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    plugin_config = json.loads(
        (ROOT / "plugins/mmm-minecraft-mod-ai/.mcp.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [
        "-m",
        "minecraft_mod_ai.mcp_stdio_entrypoint",
        "minecraft_mod_ai.mod_generation_mcp_server",
    ]
    for config in (root_config, plugin_config):
        args = config["mcpServers"]["mmm-generation"]["args"]
        assert args == expected
    registry = (
        ROOT / "minecraft_mod_ai/config/external_mcp_registry.yaml"
    ).read_text(encoding="utf-8")
    assert "minecraft_mod_ai.mod_generation_mcp_server" in registry
    assert "env: {MMM_MCP_STAGE: generation}" not in registry
