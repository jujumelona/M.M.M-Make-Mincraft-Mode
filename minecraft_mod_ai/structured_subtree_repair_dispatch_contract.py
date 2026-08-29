from __future__ import annotations

"""Route structured planner sections through bounded constrained field generation."""

from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import agentic_research_game_design as _design
from . import structured_repair_contract as _base
from . import structured_unit_generation_contract as _units

_INSTALLED = False
_IN_UNIT_GENERATION: ContextVar[bool] = ContextVar(
    "mmm_structured_field_unit_generation",
    default=False,
)


def install_structured_subtree_repair_dispatch_contract() -> None:
    """Install field-unit generation while preserving the public bootstrap hook."""

    global _INSTALLED
    if _INSTALLED:
        return
    if not getattr(_design._generate_section, "_mmm_field_local_repair", False):
        raise RuntimeError(
            "field-unit structured generation requires installed field-local contract"
        )

    original = _base._generate_section_local
    if getattr(original, "_mmm_field_unit_dispatch", False):
        _INSTALLED = True
        return

    @wraps(original)
    def dispatch(*args: Any, **kwargs: Any):
        if _IN_UNIT_GENERATION.get():
            return original(*args, **kwargs)
        token = _IN_UNIT_GENERATION.set(True)
        try:
            return _units._generate_section_units(*args, **kwargs)
        finally:
            _IN_UNIT_GENERATION.reset(token)

    dispatch._mmm_field_unit_dispatch = True
    dispatch._mmm_exact_subtree_dispatch = True
    dispatch._mmm_field_local_fallback = original
    _base._generate_section_local = dispatch
    _INSTALLED = True


__all__ = ["install_structured_subtree_repair_dispatch_contract"]
