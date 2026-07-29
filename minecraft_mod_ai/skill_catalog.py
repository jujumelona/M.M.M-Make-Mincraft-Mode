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
    "execute-complete-production",
    "patch-existing-project",
    "generate-audio",
    "publish-release",
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
    """Load a complete catalog and overlay checked-out Skill files.

    Wheel installs contain ``packaged_skills.json`` while source checkouts expose
    ``skills/<name>/SKILL.md``.  A partial checkout must not hide packaged Skills,
    so both sources are merged and repository files take precedence.
    """

    texts: dict[str, str] = {}
    packaged = Path(__file__).resolve().parent / "packaged_skills.json"
    if packaged.is_file():
        raw = json.loads(packaged.read_text(encoding="utf-8"))
        skills = raw.get("skills", {})
        texts.update(
            {
                str(name): str(text)
                for name, text in skills.items()
                if isinstance(name, str) and isinstance(text, str)
            }
        )

    base = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[1] / "skills"
    )
    if base.is_dir():
        for skill in CANONICAL_SKILLS:
            path = base / skill / "SKILL.md"
            if path.is_file():
                texts[skill] = path.read_text(encoding="utf-8")
    return texts
