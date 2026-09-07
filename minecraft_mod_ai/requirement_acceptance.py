from __future__ import annotations

"""Host-owned requirement acceptance projection.

Public-acceptance semantics are owned exclusively by acceptance_contracts.py. This
module only projects candidate checks onto a requirement and must not define a second
acceptance policy.
"""

from collections.abc import Iterable
from typing import Any

from .acceptance_contracts import is_public_acceptance


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
