from __future__ import annotations

import json
import os
from typing import Any

import httpx


_ALLOWED = frozenset(
    {
        "open_project",
        "create_cube",
        "set_texture",
        "set_uv",
        "create_animation",
        "render_preview",
        "validate_uv",
        "export_bbmodel",
        "export_geckolib",
        "close_project",
    }
)


class BlockbenchMCPError(RuntimeError):
    pass


class BlockbenchMCPClient:
    """Restricted streamable-HTTP MCP client for the Blockbench plugin.

    This client deliberately exposes only reviewed modeling operations. It never
    forwards arbitrary script, shell or unrestricted file tools.
    """

    def __init__(self, url: str | None = None, timeout_seconds: int = 60) -> None:
        self.url = (
            url
            or os.environ.get("MMM_BLOCKBENCH_MCP_URL", "").strip()
            or "http://127.0.0.1:3000/bb-mcp"
        )
        if not self.url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise BlockbenchMCPError(
                "Blockbench MCP must be bound to localhost in this profile."
            )
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(timeout=timeout_seconds)
        self.session_id: str | None = None
        self._request_id = 1

    def initialize(self) -> dict[str, Any]:
        result, headers = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "mmm-blockbench-restricted", "version": "1"},
                },
            }
        )
        self.session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
        self._post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        if self.session_id is None:
            self.initialize()
        result, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            }
        )
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("name") in _ALLOWED
        ]

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation not in _ALLOWED:
            raise BlockbenchMCPError(f"Blockbench operation is not allowlisted: {operation}")
        if self.session_id is None:
            self.initialize()
        available = {tool["name"] for tool in self.list_tools()}
        if operation not in available:
            raise BlockbenchMCPError(
                f"Connected Blockbench server does not expose reviewed tool {operation!r}."
            )
        result, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": operation, "arguments": arguments},
            }
        )
        return {
            "schema_version": "mmm/blockbench-call-result-v1",
            "operation": operation,
            "result": result,
        }

    def close(self) -> None:
        self.client.close()

    def _next_id(self) -> int:
        value = self._request_id
        self._request_id += 1
        return value

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = self.client.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        parsed = _parse_response(response)
        if "error" in parsed:
            raise BlockbenchMCPError(json.dumps(parsed["error"], ensure_ascii=False))
        result = parsed.get("result", {})
        if not isinstance(result, dict):
            result = {"value": result}
        return result, dict(response.headers)


def _parse_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        value = response.json()
        if not isinstance(value, dict):
            raise BlockbenchMCPError("Blockbench MCP returned a non-object response.")
        return value
    messages: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw:
            continue
        value = json.loads(raw)
        if isinstance(value, dict):
            messages.append(value)
    if not messages:
        raise BlockbenchMCPError("Blockbench MCP returned an empty event stream.")
    return messages[-1]


def allowed_blockbench_operations() -> tuple[str, ...]:
    return tuple(sorted(_ALLOWED))
