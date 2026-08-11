from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config_paths import config_path
from .platform_catalog import supported_minecraft_versions


class ExternalMCPRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else config_path("external_mcp_registry.yaml")
        )
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != "mmm/external-mcp-registry-v1":
            raise ValueError("Unsupported external MCP registry.")
        servers = raw.get("servers")
        if not isinstance(servers, dict) or not servers:
            raise ValueError("External MCP registry contains no servers.")
        self.servers = servers
        self.validate()

    def validate(self) -> None:
        statuses = {
            "enabled",
            "optional",
            "configuration_required",
            "incompatible_by_default",
        }
        reviewed_versions = set(supported_minecraft_versions(loader="fabric"))
        for name, entry in self.servers.items():
            if not isinstance(entry, dict) or entry.get("status") not in statuses:
                raise ValueError(f"Invalid MCP registry entry: {name}")
            versions = entry.get("target_versions", [])
            if versions:
                if not isinstance(versions, list) or any(
                    not isinstance(value, str) or not value.strip() for value in versions
                ):
                    raise ValueError(f"Invalid target_versions for MCP server {name}.")
                unknown = sorted(set(versions) - reviewed_versions)
                if unknown and entry.get("status") != "incompatible_by_default":
                    raise ValueError(
                        f"MCP server {name} advertises unreviewed Minecraft targets: {unknown}"
                    )
            if name.startswith("minecraft-runtime-") or name.startswith("mineflayer-"):
                if entry.get("status") != "incompatible_by_default" and not versions:
                    raise ValueError(f"{name} must declare an exact Minecraft target.")
            if entry.get("status") == "enabled" and not (
                entry.get("command") or entry.get("default_url")
            ):
                raise ValueError(f"Enabled MCP server {name} has no launch target.")

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/external-mcp-registry-public-v1",
            "servers": self.servers,
        }
