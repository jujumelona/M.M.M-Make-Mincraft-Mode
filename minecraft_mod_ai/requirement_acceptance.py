from __future__ import annotations

"""Host-owned public acceptance normalization.

Acceptance is user-facing observable behavior. Task hashes, owned anchors, provider
receipts and gate bookkeeping are internal invariants and must never leak into the
public requirement contract.
"""

from collections.abc import Iterable
from typing import Any

_INTERNAL_MARKERS = (
    "owned anchors",
    "owned_anchor",
    "declared provides",
    "declared_provides",
    "required gates",
    "required_gates",
    "task_sha256",
    "done_predicate",
)


def is_public_acceptance(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    folded = text.casefold()
    if folded.startswith("task_") and ":" in folded:
        return False
    return not any(marker in folded for marker in _INTERNAL_MARKERS)


def requirement_acceptance(
    capability: str,
    candidates: Iterable[Any],
) -> tuple[str, ...]:
    claimed = tuple(
        dict.fromkeys(
            text
            for item in candidates
            if (text := str(item or "").strip()) and is_public_acceptance(text)
        )
    )
    if claimed:
        return claimed
    return (
        f"Verify the observable player-facing behavior for capability {capability}.",
    )


__all__ = ["is_public_acceptance", "requirement_acceptance"]
