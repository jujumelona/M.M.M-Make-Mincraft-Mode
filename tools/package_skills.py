from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.skill_catalog import (
    CANONICAL_SKILLS,
    compile_skill_catalog,
)


ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = ROOT / "skills"
OUTPUT = ROOT / "minecraft_mod_ai" / "packaged_skills.json"


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
        for name, contract in compile_skill_catalog(SKILLS_ROOT).items()
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
