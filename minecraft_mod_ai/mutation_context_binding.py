from __future__ import annotations

"""Keep retrieval evidence separate from host-owned mutation targeting."""

from typing import Any

from .mutation_authority import CURRENT_MUTATION_AUTHORITY, MutationAuthorityMode


def observed_context_may_bind(
    current_context: Any,
    observed_context: Any,
) -> bool:
    if observed_context is None:
        return False
    authority = CURRENT_MUTATION_AUTHORITY.get()
    if authority is None or authority.mode is not MutationAuthorityMode.BOUNDED_ROOTS:
        return True
    return bool(
        getattr(observed_context, "target_pinned", False)
        or getattr(current_context, "target_pinned", False)
    )


def context_is_host_pinned(context: Any) -> bool:
    return bool(context is not None and getattr(context, "target_pinned", False))
