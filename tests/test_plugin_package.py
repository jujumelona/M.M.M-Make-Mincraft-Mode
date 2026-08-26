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

    # Canonical Skills are source of truth.  The Codex plugin must package the exact
    # same bytes rather than merely containing a file with the same directory name;
    # otherwise policy/validator updates silently disappear at plugin runtime.
    for skill_name in CANONICAL_SKILLS:
        canonical = ROOT / "skills" / skill_name / "SKILL.md"
        packaged = PLUGIN / "skills" / skill_name / "SKILL.md"
        assert packaged.read_bytes() == canonical.read_bytes(), skill_name

    mcp = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    expected_stages = {
        "mmm-frontdoor": "frontdoor",
        "mmm-research": "research",
        "mmm-generation": "generation",
        "mmm-quality": "quality",
        "mmm-runtime": "runtime",
        "mmm-release": "release",
        "mmm-training": "training",
    }
    assert set(expected_stages) <= set(mcp["mcpServers"])
    for server_name, stage in expected_stages.items():
        assert mcp["mcpServers"][server_name]["env"]["MMM_MCP_STAGE"] == stage
