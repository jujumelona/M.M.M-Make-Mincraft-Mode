from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CANONICAL_SKILLS = (
    "intake-mod-brief",
    "research-minecraft-evidence",
    "plan-game-design",
    "freeze-approved-spec",
    "inspect-existing-project",
    "generate-fabric-core",
    "generate-datagen",
    "generate-geckolib-entity",
    "generate-worldgen",
    "generate-quest-progression",
    "generate-gui-networking",
    "generate-textures",
    "model-with-blockbench",
    "compile-and-repair",
    "runtime-playtest",
    "visual-review",
    "release-security",
)
REQUIRED_SECTIONS = (
    "activate_when:",
    "inputs:",
    "required_rag:",
    "allowed_tools:",
    "output_schema:",
    "validators:",
    "retry_policy:",
    "approval_required:",
    "forbidden_actions:",
    "exit_conditions:",
)


def validate_skill_catalog(root: str | Path | None = None) -> dict[str, Any]:
    texts = _skill_texts(root)
    findings: list[str] = []
    for skill in CANONICAL_SKILLS:
        text = texts.get(skill)
        if text is None:
            findings.append(f"missing:{skill}")
            continue
        if "[TODO" in text or "TODO:" in text:
            findings.append(f"todo:{skill}")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                findings.append(f"missing-section:{skill}:{section}")
    return {
        "schema_version": "mmm/skill-catalog-validation-v1",
        "skills": list(CANONICAL_SKILLS),
        "findings": findings,
        "passed": not findings,
    }


def _skill_texts(root: str | Path | None) -> dict[str, str]:
    if root is not None:
        base = Path(root).expanduser().resolve()
    else:
        base = Path(__file__).resolve().parents[1] / "skills"
    if base.is_dir():
        return {
            skill: (base / skill / "SKILL.md").read_text(encoding="utf-8")
            for skill in CANONICAL_SKILLS
            if (base / skill / "SKILL.md").is_file()
        }
    packaged = Path(__file__).resolve().parent / "packaged_skills.json"
    if not packaged.is_file():
        return {}
    raw = json.loads(packaged.read_text(encoding="utf-8"))
    skills = raw.get("skills", {})
    return {
        str(name): str(text)
        for name, text in skills.items()
        if isinstance(name, str) and isinstance(text, str)
    }
