from __future__ import annotations

import json

from minecraft_mod_ai import agent_capability_context as capability_context
from minecraft_mod_ai.agent_capability_context import build_agent_capability_context


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _decode_context(text: str) -> dict[str, object]:
    prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
    assert text.startswith(prefix)
    return json.loads(text[len(prefix) :])


def test_external_manifest_failure_is_explicit_but_remains_fail_closed(monkeypatch) -> None:
    class _BrokenManifestRouter:
        def capability_manifest(self, *, stage, target, max_access):
            del stage, target, max_access
            raise RuntimeError("provider token=must-not-leak")

    monkeypatch.setattr(
        capability_context,
        "_manifest_router",
        lambda: _BrokenManifestRouter(),
    )

    rendered = build_agent_capability_context(
        "research",
        (
            _schema("external_mcp_capabilities"),
            _schema("external_mcp_schema"),
            _schema("external_mcp_call"),
        ),
        model_role="researcher",
    )
    context = _decode_context(rendered)

    assert context["external_minecraft_mcp_capabilities"] == {}
    assert context["external_minecraft_mcp_access"] == {}
    assert context["external_minecraft_mcp_manifest"] == {
        "status": "UNAVAILABLE",
        "error_category": "MANIFEST_BUILD_FAILED",
        "exception_type": "RuntimeError",
    }
    assert "must-not-leak" not in rendered
    assert "provider token" not in rendered


def test_external_manifest_status_is_not_requested_without_proxy_tools() -> None:
    context = _decode_context(
        build_agent_capability_context(
            "research",
            (_schema("search_code_rag"),),
            model_role="researcher",
        )
    )

    assert context["external_minecraft_mcp_manifest"] == {"status": "NOT_REQUESTED"}
