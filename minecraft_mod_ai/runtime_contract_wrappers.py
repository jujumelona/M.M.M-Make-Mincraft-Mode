from __future__ import annotations

"""Safe metadata helpers for runtime contract wrapper stacks.

Runtime contracts are composed by wrapping already-installed callables.  The default
``functools.wraps`` behavior updates the wrapper ``__dict__`` from the wrapped
callable.  That is unsafe for MMM because ``_mmm_*`` attributes are ownership markers:
a marker copied from an inner layer makes an outer layer look like it owns a contract
that it merely wraps.

Use :func:`contract_wraps` for runtime monkey-patch wrappers and
:func:`owns_contract_marker` whenever ownership (rather than inherited visibility)
matters.
"""

from functools import wraps
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def contract_wraps(wrapped: _F):
    """Preserve callable metadata without copying runtime ownership markers.

    ``updated=()`` keeps ``__wrapped__`` and the standard assigned metadata used by
    inspection/signature tooling, but deliberately does not merge the wrapped
    callable's ``__dict__`` into the new wrapper.
    """

    return wraps(wrapped, updated=())


def owns_contract_marker(value: Any, marker: str) -> bool:
    """Return whether ``marker`` is defined by this exact callable layer."""

    namespace = getattr(value, "__dict__", None)
    return bool(isinstance(namespace, dict) and namespace.get(marker, False))


def contract_markers(value: Any) -> frozenset[str]:
    """Return the exact layer's MMM runtime ownership markers."""

    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict):
        return frozenset()
    return frozenset(
        name
        for name, enabled in namespace.items()
        if name.startswith("_mmm_") and bool(enabled)
    )


__all__ = ["contract_markers", "contract_wraps", "owns_contract_marker"]
