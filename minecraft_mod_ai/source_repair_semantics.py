from __future__ import annotations

"""Semantic validation for one bounded existing-source repair candidate."""

import re
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


_IMPORT_LINE_RE = re.compile(r"^\s*import\s+(?:static\s+)?[\w.*]+;\s*$")
_PACKAGE_DECL_RE = re.compile(r"(?m)^\s*package\s+[A-Za-z_$][\w.$]*\s*;")


def atomic_repair_scope_error(
    *,
    old_text: Any,
    new_text: Any,
    max_chars: int,
) -> str | None:
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return None
    if len(new_text) > max_chars:
        return (
            "REPAIR_ATOMIC_REPLACEMENT_TOO_LARGE: replacement has "
            f"{len(new_text)} chars but this verifier-selected span allows at most "
            f"{max_chars}; repair only the selected span"
        )

    old_lines = tuple(line for line in old_text.splitlines() if line.strip())
    new_lines = tuple(line for line in new_text.splitlines() if line.strip())
    if old_lines and all(_IMPORT_LINE_RE.fullmatch(line) for line in old_lines):
        if any(not _IMPORT_LINE_RE.fullmatch(line) for line in new_lines):
            return (
                "REPAIR_ATOMIC_SCOPE_VIOLATION: an import-only verifier span may "
                "only be deleted or replaced by import declarations"
            )

    if _PACKAGE_DECL_RE.search(old_text) is None and _PACKAGE_DECL_RE.search(new_text):
        return (
            "REPAIR_ATOMIC_SCOPE_VIOLATION: replacement introduced a package "
            "declaration outside the verifier-selected span"
        )
    return None
