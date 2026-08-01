from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "mmm-minecraft-mod-ai"


def test_plugin_bundles_frontdoor_scale_rag_resume_and_mcp_stages() -> None:
    manifest = json.loads(
        (PLUGIN / ".codex-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["name"] == "mmm-minecraft-mod-ai"
    assert manifest["license"] == "MIT"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"

    skill_names = {
        path.parent.name
        for path in (PLUGIN / "skills").glob("*/SKILL.md")
    }
    assert set(CANONICAL_SKILLS) <= skill_names
    assert "make-minecraft-mod" in skill_names

    mcp = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    assert {
        "mmm-frontdoor",
        "mmm-research",
        "mmm-generation",
        "mmm-quality",
        "mmm-runtime",
        "mmm-release",
    } <= set(mcp["mcpServers"])
