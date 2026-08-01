from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_mod_ai.skill_catalog import (
    CANONICAL_SKILLS,
    compile_skill_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPOSITORY_ROOT / "minecraft_mod_ai" / "packaged_skills.json"


def build_payload() -> dict[str, object]:
    skill_root = REPOSITORY_ROOT / "skills"
    contracts = compile_skill_catalog(skill_root)
    texts = {
        name: (skill_root / name / "SKILL.md").read_text(encoding="utf-8")
        for name in CANONICAL_SKILLS
    }
    return {
        "schema_version": "mmm/packaged-skills-v3",
        "skills": texts,
        "contracts": {
            name: contracts[name].to_dict()
            for name in CANONICAL_SKILLS
        },
    }


def rendered_payload() -> str:
    return json.dumps(build_payload(), ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic in-wheel Skill text and policy contracts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when packaged_skills.json is not current.",
    )
    args = parser.parse_args()

    rendered = rendered_payload()
    if args.check:
        current = (
            OUTPUT_PATH.read_text(encoding="utf-8")
            if OUTPUT_PATH.is_file()
            else ""
        )
        if current != rendered:
            print(f"stale: {OUTPUT_PATH}")
            return 1
        print(f"current: {OUTPUT_PATH}")
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
