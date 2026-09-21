from __future__ import annotations

"""Semantic validation for one bounded existing-source repair candidate."""

from collections.abc import Callable
from typing import Any

SemanticCheck = Callable[[str, str | None, Any], str | None]


def existing_source_repair_semantic_error(
    *,
    operation: str,
    supplied: str,
    pinned: str,
    is_new_file: bool,
    current_source: Any,
    old_text: Any,
    new_text: Any,
    identity_check: SemanticCheck,
    footprint_check: SemanticCheck,
) -> str | None:
    if operation != "replace_exact" or supplied != pinned or is_new_file:
        return None
    if not (
        isinstance(current_source, str)
        and isinstance(old_text, str)
        and old_text
        and isinstance(new_text, str)
        and current_source.count(old_text) == 1
    ):
        return None
    candidate_source = current_source.replace(old_text, new_text, 1)
    identity_error = identity_check(pinned, current_source, candidate_source)
    if identity_error is not None:
        return identity_error
    return footprint_check(pinned, current_source, candidate_source)
