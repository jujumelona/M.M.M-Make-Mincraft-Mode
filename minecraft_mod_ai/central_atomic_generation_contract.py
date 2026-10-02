from __future__ import annotations

"""Compatibility hook for central structured generation.

Central responses are no longer fragmented by arbitrary field/string/array limits.
The complete host-owned schema is passed through as one semantic contract; transport
and context budgets remain runtime concerns rather than correctness gates.
"""

from functools import wraps
from types import ModuleType
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_central_atomic_fixed_template_generation"


def install(central_module: ModuleType) -> None:
    current = central_module.generate_fixed_template_text
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def semantic_generate(
        router: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        return current(router, role, messages, **kwargs)

    setattr(semantic_generate, _MARKER, True)
    semantic_generate.__wrapped__ = current
    central_module.generate_fixed_template_text = semantic_generate


__all__ = ["install"]
