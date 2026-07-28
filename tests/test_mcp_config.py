import json
from pathlib import Path


def test_mcp_config_has_real_servers() -> None:
    config = json.loads(Path(".mcp.json").read_text(encoding="utf-8"))
    servers = config["mcpServers"]
    assert servers["mmm-local"]["args"] == ["-m", "minecraft_mod_ai.mcp_server"]
    assert servers["minecraft-dev"]["args"] == ["-y", "@mcdxai/minecraft-dev-mcp"]
