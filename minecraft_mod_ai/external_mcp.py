from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config_paths import config_path
from .strict_yaml import safe_load_unique_keys


_ALLOWED_STATUSES = frozenset(
    {
        "enabled",
        "optional",
        "configuration_required",
        "incompatible_by_default",
        "disabled",
    }
)
_ALLOWED_TRANSPORTS = frozenset(
    {"stdio", "streamable_http", "first_party_process", "first_party_jsonl", "stdio_lsp"}
)
_ALLOWED_ACCESS = frozenset({"read", "write", "admin"})
_ALLOWED_VERSION_POLICIES = frozenset(
    {"dynamic", "provider_reported", "agnostic", "exact"}
)
_RESEARCH_STAGES = frozenset({"planning", "research", "migration", "generation", "quality", "runtime"})


class ExternalMCPRegistry:
    """Reviewed registry of external MCP providers and capability routes.

    Minecraft versions are deliberately not a global allow-list. A provider may
    discover/validate targets at runtime (``dynamic`` or ``provider_reported``),
    while a genuinely version-pinned helper may use ``exact`` with
    ``target_versions``. The approved MMM PlatformLock remains authoritative.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else config_path("external_mcp_registry.yaml")
        )
        raw = safe_load_unique_keys(
            self.path.read_text(encoding="utf-8"),
            source="external MCP registry",
        )
        if not isinstance(raw, dict):
            raise ValueError("External MCP registry must be an object.")
        schema = raw.get("schema_version")
        if schema not in {"mmm/external-mcp-registry-v1", "mmm/external-mcp-registry-v2"}:
            raise ValueError("Unsupported external MCP registry.")
        servers = raw.get("servers")
        if not isinstance(servers, dict) or not servers:
            raise ValueError("External MCP registry contains no servers.")
        self.schema_version = str(schema)
        self.servers: dict[str, dict[str, Any]] = deepcopy(servers)
        self.validate()

    def validate(self) -> None:
        for name, entry in self.servers.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
                raise ValueError(f"Invalid MCP registry entry: {name!r}")
            status = entry.get("status")
            if status not in _ALLOWED_STATUSES:
                raise ValueError(f"Invalid MCP registry status for {name}: {status!r}")
            transport = entry.get("transport")
            if transport is not None and transport not in _ALLOWED_TRANSPORTS:
                raise ValueError(f"Invalid MCP transport for {name}: {transport!r}")
            if status == "enabled" and not (
                entry.get("command") or entry.get("default_url") or entry.get("url_env")
            ):
                raise ValueError(f"Enabled MCP server {name} has no launch target.")

            version_policy = entry.get("version_policy")
            if version_policy is None:
                version_policy = "exact" if entry.get("target_versions") else "agnostic"
                entry["version_policy"] = version_policy
            if version_policy not in _ALLOWED_VERSION_POLICIES:
                raise ValueError(
                    f"Invalid MCP version_policy for {name}: {version_policy!r}"
                )
            versions = entry.get("target_versions", [])
            if not isinstance(versions, list) or any(
                not isinstance(value, str) or not value.strip() for value in versions
            ):
                raise ValueError(f"Invalid target_versions for MCP server {name}.")
            if version_policy == "exact" and status != "incompatible_by_default" and not versions:
                raise ValueError(f"Exact-version MCP server {name} must declare target_versions.")

            loaders = entry.get("loaders", ["*"])
            if not isinstance(loaders, list) or not loaders or any(
                not isinstance(value, str) or not value.strip() for value in loaders
            ):
                raise ValueError(f"Invalid loaders for MCP server {name}.")
            entry["loaders"] = [str(value).strip().lower() for value in loaders]

            required_env = entry.get("required_env", [])
            if not isinstance(required_env, list) or any(
                not isinstance(value, str) or not value.strip() for value in required_env
            ):
                raise ValueError(f"Invalid required_env for MCP server {name}.")

            capabilities = entry.get("capabilities", {})
            if capabilities is None:
                capabilities = {}
                entry["capabilities"] = capabilities
            if not isinstance(capabilities, dict):
                raise ValueError(f"MCP capabilities for {name} must be an object.")
            for capability, route in capabilities.items():
                self._validate_capability(name, capability, route)

    @staticmethod
    def _validate_capability(name: str, capability: Any, route: Any) -> None:
        if not isinstance(capability, str) or not capability.strip():
            raise ValueError(f"Invalid capability name for {name}.")
        if isinstance(route, str):
            route = {"tool": route}
        if not isinstance(route, dict):
            raise ValueError(f"Capability {name}/{capability} must be an object.")
        tool = route.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError(f"Capability {name}/{capability} requires a tool name.")
        access = route.get("access", "read")
        if access not in _ALLOWED_ACCESS:
            raise ValueError(f"Invalid access for {name}/{capability}: {access!r}")
        stages = route.get("stages", sorted(_RESEARCH_STAGES))
        if not isinstance(stages, list) or not stages or any(
            not isinstance(value, str) or not value.strip() for value in stages
        ):
            raise ValueError(f"Invalid stages for {name}/{capability}.")
        target_args = route.get("target_args", {})
        if not isinstance(target_args, dict) or any(
            key not in {"minecraft_version", "loader", "mapping", "mappings"}
            or not isinstance(value, str)
            or not value.strip()
            for key, value in target_args.items()
        ):
            raise ValueError(f"Invalid target_args for {name}/{capability}.")
        response_target_fields = route.get("response_target_fields", [])
        if not isinstance(response_target_fields, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in response_target_fields
        ):
            raise ValueError(
                f"Invalid response_target_fields for {name}/{capability}."
            )
        priority = route.get("priority", 100)
        if type(priority) is not int or not 0 <= priority <= 10_000:
            raise ValueError(f"Invalid priority for {name}/{capability}.")

    def server(self, name: str) -> dict[str, Any]:
        try:
            return deepcopy(self.servers[name])
        except KeyError as exc:
            raise KeyError(f"Unknown external MCP server: {name}") from exc

    def routes(
        self,
        capability: str,
        *,
        stage: str,
        minecraft_version: str = "",
        loader: str = "fabric",
        max_access: str = "read",
    ) -> tuple[dict[str, Any], ...]:
        if max_access not in _ALLOWED_ACCESS:
            raise ValueError(f"Invalid max_access: {max_access!r}")
        access_rank = {"read": 0, "write": 1, "admin": 2}
        selected_loader = str(loader or "fabric").strip().lower()
        rows: list[dict[str, Any]] = []
        for name, entry in self.servers.items():
            if entry.get("status") in {"disabled", "incompatible_by_default"}:
                continue
            if entry.get("federated", True) is False:
                continue
            route = entry.get("capabilities", {}).get(capability)
            if isinstance(route, str):
                route = {"tool": route}
            if not isinstance(route, dict):
                continue
            stages = route.get("stages", sorted(_RESEARCH_STAGES))
            if stage not in stages:
                continue
            if access_rank[route.get("access", "read")] > access_rank[max_access]:
                continue
            loaders = entry.get("loaders", ["*"])
            if "*" not in loaders and selected_loader not in loaders:
                continue
            policy = entry.get("version_policy", "agnostic")
            if policy == "exact" and minecraft_version:
                if minecraft_version not in entry.get("target_versions", []):
                    continue
            rows.append(
                {
                    "server": name,
                    "entry": deepcopy(entry),
                    "route": deepcopy(route),
                    "priority": int(route.get("priority", 100)),
                }
            )
        rows.sort(key=lambda item: (item["priority"], item["server"]))
        return tuple(rows)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/external-mcp-registry-public-v2",
            "servers": deepcopy(self.servers),
        }
