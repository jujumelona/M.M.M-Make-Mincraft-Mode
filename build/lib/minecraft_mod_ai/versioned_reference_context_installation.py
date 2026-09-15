from __future__ import annotations

"""Install the exact-version research context into the custom coder hot path."""

from threading import RLock

_INSTALL_LOCK = RLock()
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from . import custom_generation_research as generation_research
        from .versioned_research_context import VersionedResearchCodeContext

        generation_research.ResearchCodeContext = VersionedResearchCodeContext
        _INSTALLED = True


__all__ = ["install"]
