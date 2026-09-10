from __future__ import annotations

import json
from importlib.resources import files


class PackagedSkillRuntimeError(RuntimeError):
    pass


def packaged_skill_texts() -> dict[str, str]:
    """Load the checked-in Skill snapshot from the installed package."""

    resource = files("minecraft_mod_ai").joinpath("packaged_skills.json")
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagedSkillRuntimeError("Packaged Skill catalog is unreadable.") from exc
    if not isinstance(raw, dict):
        raise PackagedSkillRuntimeError("Packaged Skill catalog must be an object.")
    skills = raw.get("skills")
    if not isinstance(skills, dict):
        raise PackagedSkillRuntimeError("Packaged Skill catalog has no skills object.")
    result: dict[str, str] = {}
    for name, text in skills.items():
        if not isinstance(name, str) or not name or not isinstance(text, str) or not text:
            raise PackagedSkillRuntimeError("Packaged Skill catalog contains an invalid entry.")
        result[name] = text
    return result


__all__ = ["PackagedSkillRuntimeError", "packaged_skill_texts"]
