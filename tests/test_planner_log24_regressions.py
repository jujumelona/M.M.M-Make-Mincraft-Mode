from __future__ import annotations

from pathlib import Path


def test_removed_github_source_reuse_is_not_restored() -> None:
    root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (root / "pre_design_external_source_contract.py").exists()
    assert not (root / "reuse_template_pipeline.py").exists()

    from minecraft_mod_ai.pre_design_research_pipeline import _pre_design_brief
    brief = _pre_design_brief("minecraft seasonal crop planting")
    assert brief["domains"][0]["domain_id"] == "request"
    assert {"github", "curseforge", "modrinth"}.isdisjoint(
        brief["domains"][0]["providers"]
    )
