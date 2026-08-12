from __future__ import annotations

import json
from pathlib import Path

import minecraft_mod_ai.skill_catalog as skill_catalog
from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS


ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "skills"
OUTPUT = ROOT / "minecraft_mod_ai" / "packaged_skills.json"


def _compile_raw_skill_catalog():
    """Compile the checked-in Skill snapshot without runtime platform overlays."""

    current = skill_catalog._parse_skill
    raw = getattr(skill_catalog, "_mmm_raw_parse_skill", current)
    skill_catalog._parse_skill = raw
    try:
        return skill_catalog.compile_skill_catalog(SKILLS_ROOT)
    finally:
        skill_catalog._parse_skill = current


def build_payload() -> dict[str, object]:
    missing = [
        name
        for name in CANONICAL_SKILLS
        if not (SKILLS_ROOT / name / "SKILL.md").is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing canonical Skill sources: {missing}")

    skills = {
        name: (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")
        for name in CANONICAL_SKILLS
    }
    contracts = {
        name: contract.to_dict()
        for name, contract in _compile_raw_skill_catalog().items()
    }
    return {
        "schema_version": "mmm/packaged-skills-v3",
        "skills": skills,
        "contracts": contracts,
    }


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


if __name__ == "__main__":
    main()
