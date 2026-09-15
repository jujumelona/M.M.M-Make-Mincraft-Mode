from __future__ import annotations

"""Hot-path lifecycle fixes for the canonical repository-research owner.

Retrieval metrics, fusion weights, graph traversal, and ranking stay owned by
``research_code_context``.  This module only prevents duplicate draft evolution for an
identical generated state.  Keeping scoring in one owner avoids wrapper-dependent metric
vectors and redundant normalization work on every candidate.
"""

from functools import wraps
from typing import Any

_MARKER = "_mmm_research_code_context_performance_v1"


def _install_generation_fixed_point(module: Any) -> None:
    cls = module.ResearchCodeContext
    current = cls.evolve_from_generation
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def evolve_from_generation(self: Any, text: str):
        key = module._sha(str(text))
        seen = getattr(self, "_mmm_generation_evolution_seen", None)
        if not isinstance(seen, set):
            seen = set()
            self._mmm_generation_evolution_seen = seen
        if key in seen:
            violations = self.monitor.validate_model_output(text)
            return (self.bundle(), violations) if violations else (None, ())
        result = current(self, text)
        seen.add(key)
        return result

    setattr(evolve_from_generation, _MARKER, True)
    evolve_from_generation._mmm_generation_fixed_point_v1 = True  # type: ignore[attr-defined]
    evolve_from_generation.__wrapped__ = current  # type: ignore[attr-defined]
    cls.evolve_from_generation = evolve_from_generation


def harden(module: Any) -> None:
    _install_generation_fixed_point(module)


__all__ = ["harden"]
