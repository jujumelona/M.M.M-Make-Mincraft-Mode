from __future__ import annotations

"""Host-owned localization for atomic verifier repair spans."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_REPAIR_WINDOW_CHARS = 4096
MIN_REPAIR_REPLACEMENT_CHARS = 256


def repair_replacement_max_chars(old_text: Any) -> int:
    if not isinstance(old_text, str) or not old_text:
        return MIN_REPAIR_REPLACEMENT_CHARS
    return min(
        MAX_REPAIR_WINDOW_CHARS,
        max(MIN_REPAIR_REPLACEMENT_CHARS, (len(old_text) * 2) + 128),
    )


def normalize_model_repair_replacement(
    current_source: Any,
    old_text: Any,
    model_new: Any,
) -> Any:
    """Down-project a provable whole-source candidate to the selected local span."""

    if not (
        isinstance(current_source, str)
        and isinstance(old_text, str)
        and old_text
        and isinstance(model_new, str)
        and current_source.count(old_text) == 1
    ):
        return model_new
    prefix, suffix = current_source.split(old_text, 1)
    if not model_new.startswith(prefix) or not model_new.endswith(suffix):
        return model_new
    suffix_length = len(suffix)
    end = len(model_new) - suffix_length if suffix_length else len(model_new)
    if end < len(prefix):
        return model_new
    return model_new[len(prefix):end]


def selected_repair_diagnostic(
    diagnostics: Sequence[Mapping[str, Any]],
    repair_window: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    usable = tuple(item for item in diagnostics if isinstance(item, Mapping))
    if not usable:
        return None
    raw_index = repair_window.get("diagnostic_index") if repair_window else 0
    index = raw_index if isinstance(raw_index, int) else 0
    if not 0 <= index < len(usable):
        index = 0
    return usable[index]


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_$][\w$]{2,}\b")
_QUALIFIED_NAME_RE = re.compile(r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b")
_IMPORT_DECL_RE = re.compile(
    r"(?m)(?P<full>import\s+(?:static\s+)?(?P<name>[\w.$*]+);[ \t]*(?:\r?\n|(?=\S)))"
)
_IDENTIFIER_STOPWORDS = frozenset({
    "cannot", "resolved", "resolve", "type", "variable", "method", "field",
    "constructor", "undefined", "unknown", "error", "java", "class",
    "interface", "find", "symbol",
})


def _diagnostic_line_index(
    diagnostic: Mapping[str, Any],
    line_count: int,
) -> int | None:
    raw_range = diagnostic.get("range")
    if isinstance(raw_range, Mapping):
        start = raw_range.get("start")
        if isinstance(start, Mapping):
            raw = start.get("line")
            if isinstance(raw, int) and 0 <= raw < line_count:
                return raw
    raw_line = diagnostic.get("line")
    if not isinstance(raw_line, int):
        return None
    for candidate in (raw_line - 1, raw_line):
        if 0 <= candidate < line_count:
            return candidate
    return None


def _unique_line_window(
    source: str,
    lines: Sequence[str],
    line_index: int,
) -> dict[str, Any] | None:
    for radius in (0, 1, 2, 3):
        start = max(0, line_index - radius)
        end = min(len(lines), line_index + radius + 1)
        old = "".join(lines[start:end])
        if old and old != source and len(old) <= MAX_REPAIR_WINDOW_CHARS and source.count(old) == 1:
            return {
                "start_line": start + 1,
                "end_line": end,
                "old": old,
                "old_chars": len(old),
            }
    return None


def _diagnostic_import_window(
    source: str,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    imports = tuple(_IMPORT_DECL_RE.finditer(source))
    if not imports:
        return None
    for diagnostic_index, diagnostic in enumerate(diagnostics):
        message = str(diagnostic.get("message") or "")
        qualified = tuple(_QUALIFIED_NAME_RE.findall(message))
        if not qualified:
            continue
        for match in imports:
            imported = match.group("name")
            if any(
                imported == candidate
                or imported.startswith(candidate + ".")
                or candidate.startswith(imported.rstrip(".*") + ".")
                for candidate in qualified
            ):
                old = match.group("full")
                if old and old != source and len(old) <= MAX_REPAIR_WINDOW_CHARS:
                    line = source.count("\n", 0, match.start()) + 1
                    return {
                        "start_line": line,
                        "end_line": line,
                        "old": old,
                        "old_chars": len(old),
                        "diagnostic_index": diagnostic_index,
                    }
    return None


def _identifier_window(
    source: str,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for diagnostic_index, diagnostic in enumerate(diagnostics):
        message = str(diagnostic.get("message") or "")
        for token in _IDENTIFIER_RE.findall(message):
            if token.casefold() in _IDENTIFIER_STOPWORDS:
                continue
            matches = list(re.finditer(rf"\b{re.escape(token)}\b", source))
            if len(matches) == 1:
                line = source.count("\n", 0, matches[0].start()) + 1
                return {
                    "start_line": line,
                    "end_line": line,
                    "old": token,
                    "old_chars": len(token),
                    "diagnostic_index": diagnostic_index,
                }
    return None


def _context_window(
    source: str,
    lines: Sequence[str],
    start_line: int | None,
    end_line: int | None,
) -> dict[str, Any] | None:
    for raw_line in (start_line, end_line):
        if not isinstance(raw_line, int):
            continue
        for line_index in (raw_line, raw_line - 1):
            if 0 <= line_index < len(lines):
                window = _unique_line_window(source, lines, line_index)
                if window is not None:
                    return window
    return None


def _fallback_repair_window(
    source: str,
    lines: Sequence[str],
    diagnostics: Sequence[Mapping[str, Any]],
    start_line: int | None,
    end_line: int | None,
) -> dict[str, Any] | None:
    window = _diagnostic_import_window(source, diagnostics)
    if window is not None:
        return window
    window = _identifier_window(source, diagnostics)
    if window is not None:
        return window
    window = _context_window(source, lines, start_line, end_line)
    if window is not None and diagnostics:
        return {**window, "diagnostic_index": 0}
    return window


def select_verifier_repair_window(
    source: str,
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any] | None:
    lines = source.splitlines(keepends=True)
    if not lines:
        return None
    usable = tuple(item for item in diagnostics if isinstance(item, Mapping))
    for diagnostic_index, diagnostic in enumerate(usable):
        line_index = _diagnostic_line_index(diagnostic, len(lines))
        if line_index is not None:
            window = _unique_line_window(source, lines, line_index)
            if window is not None:
                return {**window, "diagnostic_index": diagnostic_index}
    return _fallback_repair_window(
        source,
        lines,
        usable,
        start_line,
        end_line,
    )


def exact_rollback_arguments(
    *,
    path: str,
    previous_source: Any,
    context_path: Any,
    current_source: Any,
) -> dict[str, Any] | None:
    if (
        not path
        or not isinstance(previous_source, str)
        or not isinstance(current_source, str)
        or str(context_path or "").replace("\\", "/") != path
    ):
        return None
    return {
        "operation": "replace_exact",
        "path": path,
        "old": current_source,
        "new": previous_source,
        "count": 1,
    }
