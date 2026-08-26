import json
from pathlib import Path

import yaml

_STDIO_ENTRYPOINT = "minecraft_mod_ai.mcp_stdio_entrypoint"
_STAGE_MODULE = "minecraft_mod_ai.mcp_server"
_GENERATION_MODULE = "minecraft_mod_ai.mod_generation_mcp_server"
_STAGED_SERVERS = {
    "mmm-frontdoor": "frontdoor",
    "mmm-research": "research",
    "mmm-quality": "quality",
    "mmm-runtime": "runtime",
    "mmm-release": "release",
    "mmm-training": "training",
}


def _safe_module_args(target: str) -> list[str]:
    return ["-m", _STDIO_ENTRYPOINT, target]


def _assert_first_party_config(config: dict) -> None:
    servers = config["mcpServers"]
    for server_name, stage in _STAGED_SERVERS.items():
        assert servers[server_name]["args"] == _safe_module_args(_STAGE_MODULE)
        assert servers[server_name]["env"]["MMM_MCP_STAGE"] == stage

    generation = servers["mmm-generation"]
    assert generation["args"] == _safe_module_args(_GENERATION_MODULE)
    assert generation["env"]["MMM_MCP_STAGE"] == "generation"


def test_mcp_config_has_real_local_dev_and_research_servers() -> None:
    config = json.loads(Path(".mcp.json").read_text(encoding="utf-8"))
    _assert_first_party_config(config)
    servers = config["mcpServers"]

    assert servers["minecraft-dev"]["args"] == [
        "-y",
        "@mcdxai/minecraft-dev-mcp@1.2.4",
    ]
    assert servers["playwright"]["args"][1] == "@playwright/mcp@0.0.78"
    assert all("@latest" not in str(server) for server in servers.values())


def test_packaged_plugin_uses_same_protocol_safe_first_party_launchers() -> None:
    config = json.loads(
        Path("plugins/mmm-minecraft-mod-ai/.mcp.json").read_text(encoding="utf-8")
    )
    _assert_first_party_config(config)


def test_external_registry_cannot_reintroduce_direct_first_party_stdio() -> None:
    registry = yaml.safe_load(
        Path("minecraft_mod_ai/config/external_mcp_registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    servers = registry["servers"]
    safe_prefix = ["python", "-m", _STDIO_ENTRYPOINT]

    for server_name in _STAGED_SERVERS:
        assert servers[server_name]["command"] == [*safe_prefix, _STAGE_MODULE]
        assert servers[server_name]["federated"] is False

    assert servers["mmm-generation"]["command"] == [*safe_prefix, _GENERATION_MODULE]
    assert servers["mmm-generation"]["federated"] is False
