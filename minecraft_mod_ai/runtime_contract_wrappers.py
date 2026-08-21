from __future__ import annotations

"""Safe metadata helpers for runtime contract wrapper stacks.

Runtime contracts are composed by wrapping already-installed callables. The default
``functools.wraps`` behavior updates the wrapper ``__dict__`` from the wrapped
callable. That is unsafe for MMM because ``_mmm_*`` attributes are ownership markers:
a marker copied from an inner layer makes an outer layer appear to own a contract it
merely wraps.

There are two intentionally different questions:

* ``owns_contract_marker``: is this the deepest layer carrying the marker, and
  therefore its effective owner even when legacy ``wraps`` copied it outward?
* ``has_contract_marker``: is the contract installed anywhere in the wrapper chain?

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


def _raw_contract_marker(value: Any, marker: str) -> bool:
    namespace = getattr(value, "__dict__", None)
    return bool(isinstance(namespace, dict) and namespace.get(marker, False))


def owns_contract_marker(value: Any, marker: str) -> bool:
    """Return whether ``value`` is the effective owner of ``marker``.

    Legacy ``functools.wraps`` can copy an inner marker into every outer ``__dict__``.
    The marker therefore belongs to the *deepest* marked layer. This definition makes
    exact ownership correct before legacy wrappers have all been migrated.
    """

    if not _raw_contract_marker(value, marker):
        return False
    first = True
    for layer in wrapped_layers(value):
        if first:
            first = False
            continue
        if _raw_contract_marker(layer, marker):
            return False
    return True


def has_contract_marker(value: Any, marker: str) -> bool:
    """Return whether any layer in ``value``'s wrapper chain carries ``marker``."""

    return any(_raw_contract_marker(layer, marker) for layer in wrapped_layers(value))


def contract_markers(value: Any) -> frozenset[str]:
    """Return only markers effectively owned by this wrapper layer."""

    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict):
        return frozenset()
    return frozenset(
        name
        for name, enabled in namespace.items()
        if name.startswith("_mmm_")
        and bool(enabled)
        and owns_contract_marker(value, name)
    )


def copied_contract_markers(value: Any) -> frozenset[str]:
    """Return truthy MMM markers visible here but effectively owned deeper."""

    namespace = getattr(value, "__dict__", None)
    if not isinstance(namespace, dict):
        return frozenset()
    return frozenset(
        name
        for name, enabled in namespace.items()
        if name.startswith("_mmm_")
        and bool(enabled)
        and not owns_contract_marker(value, name)
    )


__all__ = [
    "contract_markers",
    "contract_wraps",
    "copied_contract_markers",
    "has_contract_marker",
    "owns_contract_marker",
    "wrapped_layers",
]
