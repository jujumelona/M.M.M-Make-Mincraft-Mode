from __future__ import annotations

"""Keep model-facing tool aliases inside the canonical Skill permission namespace."""

from functools import wraps
from typing import Any

_MARKER = "_mmm_canonical_model_tool_skill_permissions_v1"


def install(capability_module: Any, aliases_module: Any) -> None:
    current = capability_module.skills_for_tool
    if getattr(current, _MARKER, False):
        return

    canonicalize = aliases_module.canonical_model_tool

    @wraps(current)
    def canonical_skills_for_tool(
        stage: str,
        tool: str,
        *,
        model_role: str = "",
    ) -> tuple[str, ...]:
        # Normalize before entering the existing permission stack. Capturing both
        # callables here prevents later module-level rebinding from making an alias
        # acquire a narrower or different authorization path than its canonical tool.
        canonical = canonicalize(str(tool).strip())
        return tuple(current(stage, canonical, model_role=model_role))

    setattr(canonical_skills_for_tool, _MARKER, True)
    canonical_skills_for_tool.__wrapped__ = current  # type: ignore[attr-defined]
    capability_module.skills_for_tool = canonical_skills_for_tool


__all__ = ["install"]
