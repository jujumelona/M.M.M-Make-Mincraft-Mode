from __future__ import annotations

"""Install exact-path repair while preserving the original field-local fallback."""

from contextvars import ContextVar
from functools import wraps
from typing import Any

from . import agentic_research_game_design as _design
from . import structured_repair_contract as _base
from . import structured_subtree_repair_contract as _exact

_INSTALLED = False
_IN_EXACT: ContextVar[bool] = ContextVar("mmm_structured_exact_repair", default=False)


def install_structured_subtree_repair_dispatch_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if not getattr(_design._generate_section, "_mmm_field_local_repair", False):
        raise RuntimeError("exact subtree repair requires installed field-local repair")

    original = _base._generate_section_local
    if getattr(original, "_mmm_exact_subtree_dispatch", False):
        _INSTALLED = True
        return

    @wraps(original)
    def dispatch(*args: Any, **kwargs: Any):
        if _IN_EXACT.get():
            return original(*args, **kwargs)
        token = _IN_EXACT.set(True)
        try:
            return _exact._generate_section_exact(*args, **kwargs)
        finally:
            _IN_EXACT.reset(token)

    dispatch._mmm_exact_subtree_dispatch = True
    dispatch._mmm_field_local_fallback = original
    _base._generate_section_local = dispatch
    _INSTALLED = True


__all__ = ["install_structured_subtree_repair_dispatch_contract"]
