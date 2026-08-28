from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai import skill_catalog
from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS

ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "skills"
OUTPUT = ROOT / "minecraft_mod_ai" / "packaged_skills.json"
PLUGIN_SKILLS_ROOT = ROOT / "plugins" / "mmm-minecraft-mod-ai" / "skills"


def _compile_raw_skill_catalog():
    """Compile the checked-in Skill snapshot without runtime platform overlays."""

    current = skill_catalog._parse_skill
    raw = getattr(skill_catalog, "_mmm_raw_parse_skill", current)
    skill_catalog._parse_skill = raw
    try:
        return skill_catalog.compile_skill_catalog(SKILLS_ROOT)
    finally:
        skill_catalog._parse_skill = current


def _read_skill_source(name: str) -> str:
    """Canonicalize EOF so packaging is stable across equivalent text writes."""

    text = (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
    return text if text.endswith("\n") else text + "\n"


def build_payload() -> dict[str, object]:
    missing = [
        name
        for name in CANONICAL_SKILLS
        if not (SKILLS_ROOT / name / "SKILL.md").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing canonical Skill sources: {missing}")

    skills = {name: _read_skill_source(name) for name in CANONICAL_SKILLS}
    contracts = {
        name: contract.to_dict()
        for name, contract in _compile_raw_skill_catalog().items()
    }
    return {
        "schema_version": "mmm/packaged-skills-v3",
        "skills": skills,
        "contracts": contracts,
    }


def _sync_plugin_skills() -> None:
    """Keep the Codex plugin copy byte-identical to canonical Skill sources."""

    for name in CANONICAL_SKILLS:
        destination = PLUGIN_SKILLS_ROOT / name / "SKILL.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_read_skill_source(name), encoding="utf-8")


def main() -> None:
    OUTPUT.write_text(
        json.dumps(
            build_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _sync_plugin_skills()


if __name__ == "__main__":
    main()
