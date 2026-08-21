from __future__ import annotations

"""Safe metadata helpers for runtime contract wrapper stacks.

Runtime contracts are composed by wrapping already-installed callables. The default
``functools.wraps`` behavior updates the wrapper ``__dict__`` from the wrapped
callable. That is unsafe for MMM because ``_mmm_*`` attributes are ownership markers:
a marker copied from an inner layer makes an outer layer look like it owns a contract
that it merely wraps.

There are two intentionally different questions:

* ``owns_contract_marker``: does this exact wrapper layer own the contract?
* ``has_contract_marker``: is the contract already installed anywhere in the wrapper
  chain?

Install idempotence should normally use ``has_contract_marker``. Code that chooses an
exact layer to unwrap or bypass must use ``owns_contract_marker``.
"""

from functools import wraps
from typing import Any, Callable, Iterator, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def contract_wraps(wrapped: _F):
    """Preserve callable metadata without copying runtime ownership markers.

    ``updated=()`` keeps ``__wrapped__`` and the standard assigned metadata used by
    inspection/signature tooling, but deliberately does not merge the wrapped
    callable's ``__dict__`` into the new wrapper.
    """

    return wraps(wrapped, updated=())


def wrapped_layers(value: Any) -> Iterator[Any]:
    """Yield a wrapper chain once per callable layer, guarding malformed cycles."""

    current = value
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "__wrapped__", None)


def owns_contract_marker(value: Any, marker: str) -> bool:
    """Return whether ``marker`` is defined by this exact callable layer."""

    namespace = getattr(value, "__dict__", None)
    return bool(isinstance(namespace, dict) and namespace.get(marker, False))


def has_contract_marker(value: Any, marker: str) -> bool:
    """Return whether any exact layer in ``value``'s wrapper chain owns ``marker``."""

    return any(owns_contract_marker(layer, marker) for layer in wrapped_layers(value))


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


__all__ = [
    "contract_markers",
    "contract_wraps",
    "has_contract_marker",
    "owns_contract_marker",
    "wrapped_layers",
]
