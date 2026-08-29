from __future__ import annotations

"""Install single-pass constrained field generation on the planner's live section path.

The historical public installer name is retained for bootstrap compatibility only. The
implementation no longer imports, installs, or falls back to subtree/field repair code.
"""

import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from . import agentic_research_game_design as _design
from . import structured_unit_generation_contract as _units

_INSTALLED = False
_MARKER = "_mmm_single_pass_constrained_section_v1"


def install_structured_subtree_repair_dispatch_contract() -> None:
    """Bind the live planner directly to constrained field-unit generation."""

    global _INSTALLED
    if _INSTALLED:
        return

    original = _design._generate_section
    if getattr(original, _MARKER, False):
        _INSTALLED = True
        return

    @wraps(original)
    def generate_section(
        router: Any,
        *,
        prompt: str,
        section_id: str,
        fields: Sequence[str],
        properties: Mapping[str, Any],
        research: Mapping[str, Any],
        media_paths: Sequence[str | Path],
        trace_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return _units._generate_section_units(
            router,
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            properties=properties,
            research=research,
            media_paths=media_paths,
            trace_metadata=trace_metadata,
        )

    setattr(generate_section, _MARKER, True)
    generate_section.__wrapped__ = original  # type: ignore[attr-defined]
    _design._generate_section = generate_section

    # Repair stale import-by-value edges without installing any repair implementation.
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("minecraft_mod_ai.") or module is None:
            continue
        if getattr(module, "_generate_section", None) is original:
            setattr(module, "_generate_section", generate_section)

    _INSTALLED = True


__all__ = ["install_structured_subtree_repair_dispatch_contract"]
