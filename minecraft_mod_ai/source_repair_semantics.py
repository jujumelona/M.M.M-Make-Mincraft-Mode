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



_JAVA_PUBLIC_TOP_LEVEL_TYPE_RE = re.compile(
    r"\bpublic\s+(?:(?:abstract|final|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record|@interface)\s+([A-Za-z_$][\w$]*)\b"
)
_JAVA_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_JAVA_VISIBLE_METHOD_RE = re.compile(
    r"\b(?:public|protected|private)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>{};]+>\s+)?"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]+>)?(?:\[\])?\s+)+"
    r"([A-Za-z_$][\w$]*)\s*\("
)
_JAVA_VISIBLE_FIELD_RE = re.compile(
    r"\b(?:public|protected|private)\s+"
    r"(?:(?:static|final|transient|volatile)\s+)*"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]+>)?(?:\[\])?\s+)+"
    r"([A-Za-z_$][\w$]*)\s*(?:=|;|,)"
)
_JAVA_STATIC_FINAL_FIELD_RE = re.compile(
    r"\bstatic\s+final\s+"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]+>)?(?:\[\])?\s+)+"
    r"([A-Za-z_$][\w$]*)\s*(?:=|;|,)"
)


def _java_declares_type(source: str, type_name: str) -> bool:
    return bool(
        re.search(
            rf"(?:\b(?:class|interface|enum|record)\s+|@interface\s+)"
            rf"{re.escape(type_name)}\b",
            source,
        )
    )


def _java_semantic_member_anchors(source: str) -> frozenset[str]:
    clean = _JAVA_COMMENT_RE.sub(" ", source)
    methods = {
        f"method:{name}"
        for name in _JAVA_VISIBLE_METHOD_RE.findall(clean)
    }
    fields = {
        f"field:{name}"
        for name in _JAVA_VISIBLE_FIELD_RE.findall(clean)
    }
    fields.update(
        f"field:{name}"
        for name in _JAVA_STATIC_FINAL_FIELD_RE.findall(clean)
    )
    return frozenset((*methods, *fields))


def java_semantic_footprint_error(
    path: str,
    current_source: str | None,
    new_source: Any,
) -> str | None:
    """Reject verifier repairs that compile by deleting existing Java behavior."""

    if (
        not path.casefold().endswith(".java")
        or not isinstance(current_source, str)
        or not isinstance(new_source, str)
    ):
        return None
    current_bytes = len(current_source.encode("utf-8"))
    new_bytes = len(new_source.encode("utf-8"))
    if current_bytes >= 800 and new_bytes * 100 < current_bytes * 55:
        return (
            "REPAIR_SEMANTIC_FOOTPRINT_VIOLATION: Java repair candidate "
            f"collapsed {path!r} from {current_bytes} to {new_bytes} bytes; "
            "repair must preserve the existing implementation footprint"
        )

    current_anchors = _java_semantic_member_anchors(current_source)
    new_anchors = _java_semantic_member_anchors(new_source)
    missing = sorted(current_anchors - new_anchors)
    if missing:
        return (
            "REPAIR_SEMANTIC_FOOTPRINT_VIOLATION: Java repair candidate "
            f"removed existing member anchors from {path!r}: {missing[:12]!r}"
        )
    return None


def java_path_package_error(path: str, source: Any) -> str | None:
    """Require a Java package declaration to match its repository source-set path."""

    if not str(path or "").casefold().endswith(".java") or not isinstance(source, str):
        return None
    normalized = re.sub(r"^(?:\./)+", "", str(path or "").strip().replace("\\", "/"))
    relative = ""
    for prefix in (
        "src/main/java/",
        "src/client/java/",
        "src/test/java/",
        "src/gametest/",
    ):
        if normalized.startswith(prefix):
            relative = normalized.removeprefix(prefix)
            break
    if not relative or "/" not in relative:
        return None
    expected_package = relative.rsplit("/", 1)[0].replace("/", ".")
    match = _JAVA_PACKAGE_RE.search(source)
    actual_package = match.group(1) if match is not None else ""
    if actual_package == expected_package:
        return None
    return (
        "MUTATION_JAVA_PACKAGE_MISMATCH: Java source package must match its host-authorized "
        f"path; expected {expected_package!r} for {normalized!r}, got "
        f"{actual_package or '<missing>'!r}"
    )


def java_source_identity_error(
    path: str,
    current_source: str | None,
    new_source: Any,
) -> str | None:
    """Reject Java repair candidates that change the host-selected source identity."""

    if not str(path or "").casefold().endswith(".java") or not isinstance(new_source, str):
        return None
    expected_type = PurePosixPath(str(path).replace("\\", "/")).stem
    current = current_source if isinstance(current_source, str) else ""

    current_package_match = _JAVA_PACKAGE_RE.search(current)
    new_package_match = _JAVA_PACKAGE_RE.search(new_source)
    if current_package_match is not None:
        current_package = current_package_match.group(1)
        new_package = new_package_match.group(1) if new_package_match is not None else ""
        if new_package != current_package:
            return (
                "REPAIR_SEMANTIC_IDENTITY_VIOLATION: Java repair candidate changed "
                f"package identity for {path!r}: expected {current_package!r}, got "
                f"{new_package or '<missing>'!r}"
            )
    elif _java_declares_type(current, expected_type) and new_package_match is not None:
        return (
            "REPAIR_SEMANTIC_IDENTITY_VIOLATION: Java repair candidate changed "
            f"package identity for {path!r}: expected the default package, got "
            f"{new_package_match.group(1)!r}"
        )

    if _java_declares_type(current, expected_type) and not _java_declares_type(
        new_source,
        expected_type,
    ):
        replacement_public_types = tuple(
            _JAVA_PUBLIC_TOP_LEVEL_TYPE_RE.findall(new_source)
        )
        return (
            "REPAIR_SEMANTIC_IDENTITY_VIOLATION: Java repair candidate removed "
            f"the existing primary type {expected_type!r} from {path!r}"
            + (
                f"; replacement public types={replacement_public_types!r}"
                if replacement_public_types
                else ""
            )
        )
    return None

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
