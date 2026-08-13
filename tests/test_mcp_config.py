import json
from pathlib import Path


def test_mcp_config_has_real_local_dev_and_research_servers() -> None:
    config = json.loads(Path(".mcp.json").read_text(encoding="utf-8"))
    servers = config["mcpServers"]
    staged_servers = {
        "mmm-frontdoor": "frontdoor",
        "mmm-research": "research",
        "mmm-quality": "quality",
        "mmm-runtime": "runtime",
        "mmm-release": "release",
        "mmm-training": "training",
    }
    for server_name, stage in staged_servers.items():
        assert servers[server_name]["args"] == [
            "-m",
            "minecraft_mod_ai.mcp_server",
        ]
        assert servers[server_name]["env"]["MMM_MCP_STAGE"] == stage

    generation = servers["mmm-generation"]
    assert generation["args"] == [
        "-m",
        "minecraft_mod_ai.mod_generation_mcp_server",
    ]
    assert generation["env"]["MMM_MCP_STAGE"] == "generation"

    assert servers["minecraft-dev"]["args"] == [
        "-y",
        "@mcdxai/minecraft-dev-mcp@1.2.4",
    ]
    assert servers["playwright"]["args"][1] == "@playwright/mcp@0.0.78"
    assert all("@latest" not in str(server) for server in servers.values())
