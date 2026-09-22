from __future__ import annotations

"""Semantic validation for one bounded existing-source repair candidate."""

import re
from collections.abc import Callable
from pathlib import PurePosixPath
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
    semantic_baseline_source: Any = None,
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
    semantic_baseline = (
        semantic_baseline_source
        if isinstance(semantic_baseline_source, str) and semantic_baseline_source
        else current_source
    )
    identity_error = identity_check(pinned, semantic_baseline, candidate_source)
    if identity_error is not None:
        return identity_error
    return footprint_check(pinned, semantic_baseline, candidate_source)


_IMPORT_LINE_RE = re.compile(r"^\s*import\s+(?:static\s+)?[\w.*]+;\s*$")
_PACKAGE_DECL_RE = re.compile(r"(?m)^\s*package\s+[A-Za-z_$][\w.$]*\s*;")


_JAVA_PACKAGE_RE = re.compile(
    r"(?m)^\s*package\s+([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*;"
)
_JAVA_IMPORT_RE = re.compile(
    r"(?m)^\s*import\s+((?:static\s+)?[A-Za-z_$][\w$]*(?:\.[A-Za-z_$*][\w$*]*)*)\s*;"
)
_JAVA_MEMBER_RE = re.compile(
    r"\b(?:public|protected|private)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default|strictfp|transient|volatile)\s+)*"
    r"(?:<[^>{};]+>\s+)?"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]+>)?(?:\[\])?\s+)+"
    r"([A-Za-z_$][\w$]*)\s*(?:\(|=|;|,)"
)


def existing_java_structurally_subsumes_candidate(
    path: str,
    current_source: Any,
    candidate_source: Any,
) -> bool:
    """Return true only for a structurally reductive rewrite of the same Java type.

    This does not claim arbitrary method bodies are equivalent. It identifies the
    cumulative-authored case where a later model call tries to replace an already
    materialized Java file with a structurally smaller version containing no new
    imports or member anchors. The caller must still verify the preserved source.
    """

    if (
        not str(path or "").casefold().endswith(".java")
        or not isinstance(current_source, str)
        or not current_source
        or not isinstance(candidate_source, str)
        or not candidate_source
    ):
        return False

    current_package = _JAVA_PACKAGE_RE.search(current_source)
    candidate_package = _JAVA_PACKAGE_RE.search(candidate_source)
    if (current_package.group(1) if current_package else "") != (
        candidate_package.group(1) if candidate_package else ""
    ):
        return False

    expected_type = PurePosixPath(str(path).replace("\\", "/")).stem
    type_pattern = re.compile(
        rf"\b(?:class|interface|enum|record)\s+{re.escape(expected_type)}\b"
    )
    if not type_pattern.search(current_source) or not type_pattern.search(candidate_source):
        return False

    current_imports = frozenset(_JAVA_IMPORT_RE.findall(current_source))
    candidate_imports = frozenset(_JAVA_IMPORT_RE.findall(candidate_source))
    if not candidate_imports.issubset(current_imports):
        return False

    current_members = frozenset(_JAVA_MEMBER_RE.findall(current_source))
    candidate_members = frozenset(_JAVA_MEMBER_RE.findall(candidate_source))
    if not candidate_members or not candidate_members.issubset(current_members):
        return False

    return (
        candidate_members != current_members
        or len(candidate_source.encode("utf-8"))
        < len(current_source.encode("utf-8"))
    )


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
