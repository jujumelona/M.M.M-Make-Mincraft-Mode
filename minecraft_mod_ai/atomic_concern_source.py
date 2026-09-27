from __future__ import annotations

"""Host-owned concern regions for bounded small-model Java generation."""

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .custom_module_errors import CustomModuleGenerationError

MEMBERS_MARKER = "<<<MMM_CONCERN_MEMBERS>>>"
INITIALIZE_MARKER = "<<<MMM_CONCERN_INITIALIZE>>>"
END_MARKER = "<<<MMM_CONCERN_END>>>"
_DEFAULT_REGION_ATTEMPT_LIMIT = 4
_MAX_REGION_ATTEMPT_LIMIT = 32
_DEFAULT_COMPILE_REPAIR_LIMIT = 4
_MAX_COMPILE_REPAIR_LIMIT = 16


def _region_attempt_limit() -> int:
    """Return the host safety bound for one already-decomposed concern region."""

    raw = os.environ.get("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "").strip()
    if not raw:
        return _DEFAULT_REGION_ATTEMPT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_REGION_ATTEMPT_LIMIT
    return max(1, min(_MAX_REGION_ATTEMPT_LIMIT, value))


def _compile_repair_limit() -> int:
    """Bound compiler-driven repair rounds for one concern checkpoint."""

    raw = os.environ.get("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "").strip()
    if not raw:
        return _DEFAULT_COMPILE_REPAIR_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_COMPILE_REPAIR_LIMIT
    return max(1, min(_MAX_COMPILE_REPAIR_LIMIT, value))


def _trace_region_generation(
    event: str,
    *,
    result: str,
    concern: str,
    region: str,
    attempt: int,
    attempt_limit: int,
    reason: str = "",
    output_sha256: str = "",
    output_chars: int = 0,
) -> None:
    from .root_cause_trace import emit_root_cause

    emit_root_cause(
        event,
        stage="production",
        operation="atomic_concern_region",
        gate="bounded_region_generation",
        result=result,
        reason=reason,
        details={
            "concern": concern,
            "region": region,
            "attempt": attempt,
            "attempt_limit": attempt_limit,
            **({"output_sha256": output_sha256} if output_sha256 else {}),
            **({"output_chars": output_chars} if output_chars else {}),
        },
    )


_HOST_PREFIX = "MMM_ATOMIC_CONCERN"
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*;\s*$")
_FORBIDDEN = re.compile(
    r"\b(?:package|import)\s+"
    r"|\b(?:ModInitializer|ClientModInitializer|DedicatedServerModInitializer)\b"
    r"|\bonInitialize(?:Client|Server)?\b"
)
_TYPE_DECL = re.compile(
    r"(?m)^\s*(?P<modifiers>(?:(?:public|protected|private|static|final|abstract|sealed|non-sealed)\s+)*)"
    r"(?P<kind>class|interface|enum|record)\b"
)
_INITIALIZE_DECL = re.compile(r"\bpublic\s+static\s+void\s+initialize\s*\(")
_MODEL_PROSE_LINE = re.compile(
    r"(?mi)^\s*(?:"
    r"the user\b|i\s+(?:need|should|will|must|can|am|want)\b|"
    r"let me\b|looking at\b|since this\b|we\s+(?:need|should|will|must|can)\b|"
    r"here(?:'s| is)\b|(?:first|next|finally),?\b|"
    r"\d+[.)]\s+\S|[-*]\s+\*\*"
    r")"
)


def _contains_non_java_narrative(scan: str) -> bool:
    return (
        "`" in scan
        or "**" in scan
        or _MODEL_PROSE_LINE.search(scan) is not None
    )




def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().casefold()).strip("_")
    if not slug or re.fullmatch(r"[a-z][a-z0-9_]*", slug) is None:
        raise CustomModuleGenerationError(f"ATOMIC_CONCERN_INVALID_NAME: {value!r}")
    return slug


def _marker(concern: str, region: str, edge: str) -> str:
    return f"// {_HOST_PREFIX}_{_slug(concern).upper()}_{region}_{edge}"


_FENCE_LINE = re.compile(r"^\s*```(?:[A-Za-z0-9_+.\-]+)?\s*$", re.IGNORECASE)
_HOST_MARKER_LINE = re.compile(
    r"^\s*//\s*MMM_ATOMIC_CONCERN_[A-Z0-9_]+_(?:MEMBERS|INIT)_(?:START|END)\s*$",
    re.IGNORECASE,
)
_REGION_LABEL_LINE = re.compile(
    r"^\s*(?:members?|member code|initialize(?: body)?|initialization|java)\s*:?\s*$",
    re.IGNORECASE,
)

_VISIBILITY_STATIC_INITIALIZER = re.compile(
    r"(?m)^(?P<indent>[ \t]*)(?:public|protected|private)\s+static\s*\{"
)
_INVALID_VISIBILITY_INITIALIZER = re.compile(
    r"(?m)^\s*(?:public|protected|private)\s*\{"
)


def _split_response_regions(text: str) -> tuple[str, str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    marker_positions: dict[str, list[int]] = {
        marker: [index for index, line in enumerate(lines) if line.strip() == marker]
        for marker in (MEMBERS_MARKER, INITIALIZE_MARKER, END_MARKER)
    }
    if any(len(marker_positions[marker]) != 1 for marker in marker_positions):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: each host response marker must occur exactly once as a marker line."
        )
    members_at = marker_positions[MEMBERS_MARKER][0]
    init_at = marker_positions[INITIALIZE_MARKER][0]
    end_at = marker_positions[END_MARKER][0]
    if not members_at < init_at < end_at:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: host response markers are out of order."
        )
    return (
        "\n".join(lines[members_at + 1:init_at]).strip(),
        "\n".join(lines[init_at + 1:end_at]).strip(),
    )


def _normalize_region_text(value: str) -> str:
    rows: list[str] = []
    for line in str(value or "").splitlines():
        if _FENCE_LINE.fullmatch(line):
            continue
        if _HOST_MARKER_LINE.fullmatch(line):
            continue
        if _REGION_LABEL_LINE.fullmatch(line):
            continue
        rows.append(line)
    normalized = "\n".join(rows).strip()
    # Java class initializers cannot carry visibility. Small coders sometimes emit
    # "private static { ... }" when they mean a static initializer. Removing only
    # the impossible visibility modifier preserves the executable semantics.
    return _VISIBILITY_STATIC_INITIALIZER.sub(
        lambda match: f"{match.group('indent')}static {{",
        normalized,
    )


def _structure_scan(value: str) -> str:
    """Blank comments/literals so scope checks only inspect executable Java structure."""
    text = str(value or "")
    chars = list(text)
    length = len(text)
    index = 0

    def blank(start: int, end: int) -> None:
        for pos in range(start, min(end, length)):
            if chars[pos] != "\n":
                chars[pos] = " "

    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            if end < 0:
                end = length
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            blank(index, end)
            index = end
            continue
        if text.startswith('"""', index):
            end = text.find('"""', index + 3)
            end = length if end < 0 else end + 3
            blank(index, end)
            index = end
            continue
        if text[index] in {'"', "'"}:
            quote = text[index]
            end = index + 1
            escaped = False
            while end < length:
                char = text[end]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    end += 1
                    break
                end += 1
            blank(index, end)
            index = end
            continue
        index += 1
    return "".join(chars)


def _is_inert_empty_region(value: str) -> bool:
    stripped = str(value or "").strip()
    if not stripped:
        return True
    without_block_comments = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    without_comments = "\n".join(
        line for line in without_block_comments.splitlines()
        if not line.lstrip().startswith("//")
    ).strip()
    token = without_comments.casefold().rstrip(";").strip()
    return token in {
        "",
        "none",
        "n/a",
        "na",
        "empty",
        "<empty>",
        "not applicable",
        "no initialization",
        "no initialization needed",
    }


def _brace_balanced_region(scan: str) -> bool:
    depth = 0
    for char in scan:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_forbidden_type_declaration(scan: str, *, initialize_region: bool) -> bool:
    for match in _TYPE_DECL.finditer(scan):
        if initialize_region:
            return True
        modifiers = set(str(match.group("modifiers") or "").split())
        if "private" not in modifiers:
            return True
    return False


_JAVA_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_JAVA_MODIFIERS = frozenset(
    {
        "public", "protected", "private", "static", "final", "abstract",
        "synchronized", "native", "strictfp", "transient", "volatile",
        "default", "sealed", "non-sealed",
    }
)


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    paren = bracket = brace = angle = 0
    for index, char in enumerate(value):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)
        elif char == "<" and paren == 0 and brace == 0:
            angle += 1
        elif char == ">" and paren == 0 and brace == 0 and angle:
            angle -= 1
        elif (
            char == delimiter
            and paren == 0
            and bracket == 0
            and brace == 0
            and angle == 0
        ):
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts



def _top_level_member_chunks(value: str) -> tuple[str, ...]:
    source = str(value or "")
    scan = _structure_scan(source)
    chunks: list[str] = []
    start = 0
    brace = paren = bracket = 0
    for index, char in enumerate(scan):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            if brace:
                brace -= 1
                if brace == 0 and paren == 0 and bracket == 0:
                    chunk = source[start:index + 1].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = index + 1
        elif char == ";" and brace == 0 and paren == 0 and bracket == 0:
            chunk = source[start:index + 1].strip()
            if chunk:
                chunks.append(chunk)
            start = index + 1
    tail = source[start:].strip()
    if tail:
        chunks.append(tail)
    return tuple(chunks)

def _top_level_assignment_index(value: str) -> int:
    paren = bracket = brace = angle = 0
    for index, char in enumerate(value):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)
        elif char == "<" and paren == 0 and brace == 0:
            angle += 1
        elif char == ">" and paren == 0 and brace == 0 and angle:
            angle -= 1
        elif (
            char == "="
            and paren == 0
            and bracket == 0
            and brace == 0
            and angle == 0
        ):
            prev = value[index - 1] if index else ""
            nxt = value[index + 1] if index + 1 < len(value) else ""
            if prev not in "=!<>" and nxt != "=":
                return index
    return -1


def _erase_generic_arguments(value: str) -> str:
    result: list[str] = []
    depth = 0
    for char in str(value or ""):
        if char == "<":
            depth += 1
            continue
        if char == ">" and depth:
            depth -= 1
            continue
        if depth == 0:
            result.append(char)
    return "".join(result)


def _parameter_type_signature(raw: str) -> str:
    text = re.sub(r"@\w+(?:\s*\([^)]*\))?\s*", " ", str(raw or ""))
    tokens = [token for token in text.strip().split() if token and token != "final"]
    if not tokens:
        return ""
    # The final identifier is the parameter name. Preserve array suffixes attached to it.
    name = tokens[-1]
    suffix = ""
    while name.endswith("[]"):
        suffix += "[]"
        name = name[:-2]
    if _JAVA_IDENTIFIER.fullmatch(name):
        tokens = tokens[:-1]
    normalized = re.sub(r"\s+", "", " ".join(tokens)) + suffix
    return _erase_generic_arguments(normalized)


def _member_declaration_symbols(value: str) -> dict[str, str]:
    """Return compiler-relevant class-body declaration keys for one concern region."""

    symbols: dict[str, str] = {}
    for chunk in _top_level_member_chunks(value):
        flat = re.sub(r"\s+", " ", _structure_scan(chunk)).strip()
        if not flat:
            continue

        type_match = re.search(
            r"\b(?:class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b",
            flat,
        )
        if type_match:
            name = type_match.group(1)
            symbols[f"type:{name}"] = name
            continue

        header = flat
        brace_at = header.find("{")
        if brace_at >= 0:
            header = header[:brace_at].strip()
        header = header.rstrip(";").strip()
        if header in {"", "static"}:
            continue

        method_matches = list(
            re.finditer(
                r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^()]*)\)\s*(?:throws\s+[^{};]+)?$",
                header,
            )
        )
        if method_matches:
            method = method_matches[-1]
            assignment = _top_level_assignment_index(header)
            if assignment < 0 or assignment > method.start():
                name = method.group(1)
                params = ",".join(
                    _parameter_type_signature(part)
                    for part in _split_top_level(method.group(2), ",")
                    if part.strip()
                )
                display = f"{name}({params})"
                symbols[f"method:{display}"] = display
                continue

        declaration = chunk.rstrip().rstrip(";").strip()
        for declarator in _split_top_level(declaration, ","):
            left = declarator
            assignment = _top_level_assignment_index(left)
            if assignment >= 0:
                left = left[:assignment]
            identifiers = _JAVA_IDENTIFIER.findall(left)
            while identifiers and identifiers[0] in _JAVA_MODIFIERS:
                identifiers.pop(0)
            if not identifiers:
                continue
            name = identifiers[-1]
            if name in {"return", "throw", "new", "this", "super"}:
                continue
            symbols[f"field:{name}"] = name
    return symbols



def _member_type_kinds(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in _top_level_member_chunks(value):
        flat = re.sub(r"\s+", " ", _structure_scan(chunk)).strip()
        match = re.search(
            r"\b(class|interface|enum|record)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b",
            flat,
        )
        if match:
            result[match.group(2)] = match.group(1)
    return result


def _repair_type_structure_error(previous: str, candidate: str) -> str:
    before = _member_type_kinds(previous)
    after = _member_type_kinds(candidate)
    added = sorted(set(after) - set(before))
    changed = sorted(
        name
        for name in set(before) & set(after)
        if before[name] != after[name]
    )
    if not added and not changed:
        return ""
    parts: list[str] = []
    if added:
        parts.append("new nested types=" + ",".join(added))
    if changed:
        parts.append(
            "changed nested type kinds="
            + ",".join(
                f"{name}:{before[name]}->{after[name]}" for name in changed
            )
        )
    return "; ".join(parts)

_JAVA_LANG_SIMPLE_TYPES = frozenset(
    {
        "Appendable",
        "AutoCloseable",
        "Boolean",
        "Byte",
        "Character",
        "CharSequence",
        "Class",
        "ClassLoader",
        "Cloneable",
        "Comparable",
        "Deprecated",
        "Double",
        "Enum",
        "Error",
        "Exception",
        "Float",
        "FunctionalInterface",
        "IllegalArgumentException",
        "IllegalStateException",
        "Integer",
        "Iterable",
        "Long",
        "Math",
        "Number",
        "Object",
        "Override",
        "Record",
        "Runnable",
        "RuntimeException",
        "Short",
        "String",
        "StringBuffer",
        "StringBuilder",
        "SuppressWarnings",
        "System",
        "Thread",
        "Throwable",
        "Void",
    }
)


def _declared_type_names(value: str) -> set[str]:
    return {
        display
        for key, display in _member_declaration_symbols(value).items()
        if key.startswith("type:")
    }


def _unresolved_simple_type_names(
    value: str,
    *,
    allowed: Sequence[str] = (),
) -> tuple[str, ...]:
    scan = _structure_scan(value)
    local_symbols = _member_declaration_symbols(value)
    local_identifiers = {
        display
        for key, display in local_symbols.items()
        if key.startswith(("type:", "field:"))
    }
    allowed_names = set(allowed) | _JAVA_LANG_SIMPLE_TYPES | local_identifiers
    unresolved: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_$]*\b", scan):
        name = match.group(0)
        if name in allowed_names:
            continue
        if len(name) == 1 and name.isupper():
            continue
        if name.isupper() and "_" in name:
            continue
        before = scan[: match.start()].rstrip()
        if before.endswith("."):
            continue
        unresolved.add(name)
    return tuple(sorted(unresolved))


def _symbol_owner_payload(owners: Mapping[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, owner in sorted(owners.items()):
        kind, _, display = key.partition(":")
        rows.append({"kind": kind, "symbol": display, "owner_concern": owner})
    return rows


def _validate_region_text(value: str, *, initialize_region: bool) -> None:
    scan = _structure_scan(value)
    if _HOST_PREFIX in scan or "```" in scan:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: executable region contains host-marker syntax or Markdown fences."
        )
    if (
        not _brace_balanced_region(scan)
        or _contains_non_java_narrative(scan)
        or _INVALID_VISIBILITY_INITIALIZER.search(scan)
        or _FORBIDDEN.search(scan)
        or _INITIALIZE_DECL.search(scan)
        or _has_forbidden_type_declaration(scan, initialize_region=initialize_region)
    ):
        region = "initialize body" if initialize_region else "concern members"
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_SCOPE_ESCAPE: {region} attempted to change host-owned type/lifecycle structure."
        )



def parse_concern_content(text: str, *, section: str) -> tuple[str, str]:
    """Normalize inert model wrappers, then validate only executable concern content."""
    members, initialize = _split_response_regions(text)
    members = _normalize_region_text(members)
    initialize = _normalize_region_text(initialize)
    if str(section or "").strip() != "integration" and _is_inert_empty_region(initialize):
        initialize = ""
    _validate_region_text(members, initialize_region=False)
    _validate_region_text(initialize, initialize_region=True)
    if str(section or "").strip() != "integration" and initialize:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_SCOPE_ESCAPE: only integration concerns may add initialize() statements."
        )
    return members, initialize


def _parse_region_content(text: str, *, response_region: str) -> str:
    """Parse one host-selected region without requiring model-authored protocol markers."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    exact_markers = {
        MEMBERS_MARKER,
        INITIALIZE_MARKER,
        END_MARKER,
    }
    rows = [line for line in raw.splitlines() if line.strip() not in exact_markers]
    value = _normalize_region_text("\n".join(rows))
    initialize_region = response_region == "initialize"
    if response_region not in {"members", "initialize"}:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_RESPONSE_REGION_INVALID: {response_region!r}"
        )
    if initialize_region and _is_inert_empty_region(value):
        return ""
    _validate_region_text(value, initialize_region=initialize_region)
    return value

def _validate_concerns(concerns: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for expected, raw in enumerate(concerns):
        if not isinstance(raw, Mapping):
            raise CustomModuleGenerationError("ATOMIC_CONCERN_CONTRACT_INVALID: concern is not an object.")
        item = dict(raw)
        name = _slug(item.get("concern"))
        if name in names:
            raise CustomModuleGenerationError(f"ATOMIC_CONCERN_CONTRACT_INVALID: duplicate concern {name}.")
        if int(item.get("sequence", -1)) != expected:
            raise CustomModuleGenerationError(
                f"ATOMIC_CONCERN_CONTRACT_INVALID: unstable sequence for {name}."
            )
        if not str(item.get("identifier") or "").strip() or not str(item.get("task") or "").strip():
            raise CustomModuleGenerationError(
                f"ATOMIC_CONCERN_CONTRACT_INVALID: {name} lacks identifier/task."
            )
        names.add(name)
        result.append(item)
    if not result:
        raise CustomModuleGenerationError("ATOMIC_CONCERN_CONTRACT_MISSING")
    return tuple(result)


def build_concern_scaffold(
    original: str,
    *,
    symbol: str,
    concerns: Sequence[Mapping[str, Any]],
    require_initialize: bool,
) -> str:
    """Replace only a fresh host scaffold with host-owned immutable concern regions."""
    if "MMM_AUTHORED_FEATURE_BODY" not in original and _HOST_PREFIX not in original:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_FRESH_SCAFFOLD_REQUIRED: atomic concern generation may not overwrite arbitrary existing source."
        )
    package = _PACKAGE.search(original)
    package_line = f"package {package.group(1)};\n\n" if package else ""
    rows = [package_line + f"public final class {symbol} {{", f"    private {symbol}() {{}}", ""]
    for item in _validate_concerns(concerns):
        name = _slug(item["concern"])
        rows.extend([
            "    " + _marker(name, "MEMBERS", "START"),
            "    " + _marker(name, "MEMBERS", "END"),
            "",
        ])
    if require_initialize:
        rows.append("    public static void initialize() {")
        for item in concerns:
            name = _slug(item["concern"])
            rows.extend([
                "        " + _marker(name, "INIT", "START"),
                "        " + _marker(name, "INIT", "END"),
            ])
        rows.extend(["    }", ""])
    rows.append("}")
    return "\n".join(rows) + "\n"


def _replace_region(source: str, *, concern: str, region: str, content: str) -> str:
    start = _marker(concern, region, "START")
    end = _marker(concern, region, "END")
    pattern = re.compile(
        rf"(?m)^(?P<indent>[ \t]*){re.escape(start)}[ \t]*$.*?^(?P=indent){re.escape(end)}[ \t]*$",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise CustomModuleGenerationError(f"ATOMIC_CONCERN_MARKER_MISSING: {concern}:{region}")
    indent = match.group("indent")
    body = "\n" + indent
    if content:
        rendered = "\n".join(indent + line if line.strip() else "" for line in content.splitlines())
        body = "\n" + rendered + "\n" + indent
    replacement = indent + start + body + end
    return source[:match.start()] + replacement + source[match.end():]


def _concern_at_line(source: str, line_number: int) -> str:
    if line_number < 1:
        return ""
    active = ""
    for index, line in enumerate(source.splitlines(), start=1):
        match = re.search(
            r"MMM_ATOMIC_CONCERN_([A-Z0-9_]+)_(?:MEMBERS|INIT)_(START|END)",
            line,
        )
        if match:
            name = match.group(1).casefold()
            if match.group(2) == "START":
                active = name
            elif active == name:
                if index == line_number:
                    return active
                active = ""
        if index == line_number:
            return active
    return ""


def _failure_concern(source: str, *, log: str, relative: str) -> str:
    filename = re.escape(PurePosixPath(relative).name)
    for match in re.finditer(rf"(?:^|[\\/]){filename}:(\d+)(?::\d+)?", str(log or "")):
        concern = _concern_at_line(source, int(match.group(1)))
        if concern:
            return concern
    return ""


def _compact_compiler_failure(
    log: str,
    *,
    source: str,
    relative: str,
    concern: str,
    max_blocks: int = 8,
    max_chars: int = 6000,
) -> str:
    rows = str(log or "").splitlines()
    filename = re.escape(PurePosixPath(relative).name)
    blocks: list[str] = []
    index = 0
    while index < len(rows):
        match = re.search(
            rf"(?:^|[\\/]){filename}:(\d+)(?::\d+)?:\s*error:\s*(.*)$",
            rows[index],
        )
        if match is None:
            index += 1
            continue
        line_number = int(match.group(1))
        owner = _concern_at_line(source, line_number)
        start = index
        index += 1
        while index < len(rows):
            if re.search(
                rf"(?:^|[\\/]){filename}:\d+(?::\d+)?:\s*error:",
                rows[index],
            ):
                break
            if not rows[index].strip() and index > start + 1:
                index += 1
                break
            if rows[index].startswith("FAILURE:") or rows[index].startswith("* What went wrong:"):
                break
            index += 1
        if owner != concern:
            continue
        block = "\n".join(rows[start:index]).strip()
        if block and block not in blocks:
            blocks.append(block)
        if len(blocks) >= max_blocks:
            break

    if blocks:
        compact = (
            f"Compiler diagnostics localized to concern {concern}:\n"
            + "\n\n".join(blocks)
        )
    else:
        diagnostic_rows = [
            row
            for row in rows
            if (
                " error:" in row
                or row.strip().startswith(
                    ("symbol:", "location:", "required:", "found:", "reason:")
                )
            )
        ]
        compact = "\n".join(diagnostic_rows[: max_blocks * 5]).strip()
        if not compact:
            compact = "\n".join(rows[:80]).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "\n... compiler diagnostics truncated by host ..."
    return compact


def _compiler_diagnostic_fingerprint(log: str) -> str:
    diagnostics: list[str] = []
    for raw in str(log or "").splitlines():
        stripped = raw.strip()
        if " error:" in raw:
            diagnostics.append("error:" + raw.split(" error:", 1)[1].strip())
            continue
        if stripped.startswith(
            ("symbol:", "location:", "required:", "found:", "reason:")
        ):
            diagnostics.append(stripped)
    payload = "\n".join(diagnostics) if diagnostics else str(log or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _failure_measure(log: str) -> int:
    errors = {
        line.strip()
        for line in str(log or "").splitlines()
        if "error" in line.casefold() and line.strip()
    }
    return len(errors) or 1



def _requirement_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    key = str(item[0] or "")
    match = re.fullmatch(r"R(\d+)", key)
    return (int(match.group(1)) if match else 10**9, key)


def _requirement_concern_label(value: str) -> str:
    text = str(value or "")
    if not text.startswith("- ") or ":" not in text:
        return ""
    label = text[2:].split(":", 1)[0].strip()
    try:
        return _slug(label)
    except CustomModuleGenerationError:
        return ""


def _concern_source_requirements(
    raw: Mapping[str, Any],
    *,
    concern: str,
) -> dict[str, str]:
    ordered = [
        (str(key), str(value))
        for key, value in sorted(dict(raw or {}).items(), key=_requirement_sort_key)
    ]
    if not ordered:
        return {}

    target = _slug(concern)
    anchor = -1
    for index, (_key, value) in enumerate(ordered):
        if _requirement_concern_label(value) == target:
            anchor = index
            break

    headings = [
        (key, value)
        for key, value in ordered[: anchor if anchor >= 0 else len(ordered)]
        if value.lstrip().startswith("## ")
    ]
    selected: list[tuple[str, str]] = headings[-1:] if headings else []

    if anchor < 0:
        # Never leak the whole sibling section to a small coder when localization
        # fails. The concern task/rules remain authoritative.
        target_words = target.replace("_", " ")
        for key, value in ordered:
            lowered = value.casefold()
            if target in lowered or target_words in lowered:
                selected.append((key, value))
        return dict(selected)

    selected.append(ordered[anchor])
    for key, value in ordered[anchor + 1:]:
        if value.startswith("## "):
            break
        sibling_label = _requirement_concern_label(value)
        if sibling_label:
            if sibling_label == target:
                selected.append((key, value))
                continue
            break
        selected.append((key, value))
    return dict(selected)

def _concern_authority(
    task: Mapping[str, Any], concern: Mapping[str, Any]
) -> dict[str, Any]:
    name = _slug(concern.get("concern"))
    requirement_payload: dict[str, Any] = {}
    raw_obligations = task.get("implementation_obligations")
    if isinstance(raw_obligations, Sequence) and not isinstance(
        raw_obligations, (str, bytes, bytearray)
    ):
        for raw in raw_obligations:
            try:
                outer = json.loads(str(raw))
                instruction = json.loads(str(outer.get("instruction") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if str(instruction.get("concern") or "").strip() == name:
                requirement_payload = {
                    "source_requirements": _concern_source_requirements(
                        dict(outer.get("source_requirements") or {}),
                        concern=name,
                    ),
                }
                break
    return {
        "task_id": str(task.get("task_id") or ""),
        "concern": name,
        **requirement_payload,
    }


_STATE_VARIABLE_ATTRIBUTE = re.compile(
    r"\b(?P<key>[A-Za-z_][A-Za-z0-9_]*)\((?P<value>[^()]*)\)"
)
_STATE_JAVA_TYPES = {
    "bool": "boolean",
    "boolean": "boolean",
    "byte": "byte",
    "char": "char",
    "double": "double",
    "float": "float",
    "int": "int",
    "integer": "int",
    "long": "long",
    "short": "short",
    "string": "String",
}
_JAVA_RESERVED_WORDS = frozenset(
    {
        "_", "abstract", "assert", "boolean", "break", "byte", "case", "catch",
        "char", "class", "const", "continue", "default", "do", "double", "else",
        "enum", "exports", "extends", "false", "final", "finally", "float", "for",
        "goto", "if", "implements", "import", "instanceof", "int", "interface",
        "long", "module", "native", "new", "non-sealed", "null", "open", "opens",
        "package", "permits", "private", "protected", "provides", "public", "record",
        "requires", "return", "sealed", "short", "static", "strictfp", "super",
        "switch", "synchronized", "this", "throw", "throws", "to", "transient",
        "transitive", "true", "try", "uses", "var", "void", "volatile", "when",
        "while", "with", "yield",
    }
)


def _host_java_identifier(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    value = re.sub(r"[^A-Za-z0-9_$]", "_", text)
    if not value or value[0].isdigit():
        value = "$mmm$" + value
    if value in _JAVA_RESERVED_WORDS:
        value = "$mmm$" + value.replace("-", "_")
    return value


def _state_default_literal(java_type: str, raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    lowered = value.casefold()
    if java_type == "String":
        if lowered in {"empty", "blank"}:
            return '""'
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value
        return json.dumps(value, ensure_ascii=False)
    if java_type == "boolean":
        return lowered if lowered in {"true", "false"} else ""
    if java_type == "char":
        if len(value) == 3 and value.startswith("'") and value.endswith("'"):
            return value
        return ""
    if java_type in {"byte", "short", "int", "long"}:
        if re.fullmatch(r"[-+]?\d+[lL]?", value):
            return value
        return ""
    if java_type in {"float", "double"}:
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[fFdD]?",
            value,
        ):
            return value
        return ""
    return "null" if lowered == "null" else ""


def _state_variable_contract(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    authority = _concern_authority(task, concern)
    source_requirements = authority.get("source_requirements")
    if not isinstance(source_requirements, Mapping):
        return ()

    contracts: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for value in source_requirements.values():
        text = str(value or "")
        attributes = {
            match.group("key").casefold(): match.group("value").strip()
            for match in _STATE_VARIABLE_ATTRIBUTE.finditer(text)
        }
        if "name" not in attributes or "type" not in attributes:
            continue
        java_type = _STATE_JAVA_TYPES.get(attributes["type"].casefold())
        name = _host_java_identifier(attributes["name"])
        if not java_type or not name or name in seen_names:
            continue
        seen_names.add(name)
        contracts.append(
            {
                "name": name,
                "java_type": java_type,
                "default_literal": _state_default_literal(
                    java_type,
                    attributes.get("default", ""),
                ),
                "owner": attributes.get("owner", ""),
                "unit": attributes.get("unit", ""),
                "domain": attributes.get("domain", ""),
            }
        )
    return tuple(contracts)


def _deterministic_state_variable_members(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> str:
    contracts = _state_variable_contract(task, concern)
    if not contracts:
        return ""
    rows: list[str] = []
    for item in contracts:
        initializer = (
            f" = {item['default_literal']}" if item["default_literal"] else ""
        )
        rows.append(
            f"private static {item['java_type']} {item['name']}{initializer};"
        )
    return "\n".join(rows)

def _bounded_grounding(grounding: Mapping[str, Any]) -> dict[str, Any]:
    direct = grounding.get("direct_host_context")
    direct_payload = dict(direct) if isinstance(direct, Mapping) else {}
    return {
        "schema_version": grounding.get("schema_version"),
        "artifact_kind": grounding.get("artifact_kind"),
        "facts": grounding.get("facts") or [],
        "policy": dict(grounding.get("policy") or {}),
        "platform": dict(direct_payload.get("platform") or {}),
        "host_version_facts": dict(direct_payload.get("host_version_facts") or {}),
    }


def _dependency_context_rows(raw: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return tuple(rows)


def _dependency_api_context(raw: str, *, max_chars: int = 12000) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used = 0
    for row in _dependency_context_rows(raw):
        compact = {
            "symbol": str(row.get("symbol") or ""),
            "path": str(row.get("path") or ""),
            "responsibility": str(row.get("responsibility") or ""),
            "public_api": list(row.get("public_api") or []),
        }
        encoded = json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
        )
        if used + len(encoded) > max_chars:
            break
        result.append(compact)
        used += len(encoded)
    return result


def _dependency_declared_identifiers(raw: str) -> tuple[str, ...]:
    names: set[str] = set()
    for row in _dependency_context_rows(raw):
        symbol = str(row.get("symbol") or "").strip()
        if symbol:
            names.add(symbol)
    return tuple(sorted(names))


def _region_content(source: str, *, concern: str, region: str) -> str:
    start = _marker(concern, region, "START")
    end = _marker(concern, region, "END")
    pattern = re.compile(
        rf"(?ms)^\s*{re.escape(start)}[ \t]*$"
        rf"(?P<body>.*?)"
        rf"^\s*{re.escape(end)}[ \t]*$"
    )
    match = pattern.search(str(source or ""))
    return str(match.group("body") if match else "").strip()


def _member_declaration_summary(chunk: str) -> str:
    source = str(chunk or "").strip()
    if not source:
        return ""
    scan = _structure_scan(source)
    brace = scan.find("{")
    if brace >= 0:
        header = re.sub(r"\s+", " ", source[:brace]).strip()
        return (header + " { ... }").strip()
    return re.sub(r"\s+", " ", source).strip()


def _sibling_symbol_inventory(
    source: str,
    *,
    sibling_concerns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concern in sibling_concerns:
        members = _region_content(source, concern=concern, region="MEMBERS")
        for chunk in _top_level_member_chunks(members):
            declaration = _member_declaration_summary(chunk)
            symbols = _member_declaration_symbols(chunk)
            for key, display in sorted(symbols.items()):
                kind, _, _ = key.partition(":")
                row: dict[str, Any] = {
                    "owner_concern": concern,
                    "kind": kind,
                    "symbol": display,
                    "declaration": declaration,
                }
                if kind == "field":
                    row["mutable"] = not bool(
                        re.search(r"\bfinal\b", declaration)
                    )
                rows.append(row)
    return rows

def _messages(
    *,
    section: str,
    concern: Mapping[str, Any],
    task: Mapping[str, Any],
    grounding: Mapping[str, Any],
    dependency_source: str,
    current_source: str,
    response_region: str,
    failure: str = "",
    sibling_concerns: Sequence[str] = (),
    host_symbol: str = "",
) -> list[dict[str, str]]:
    name = _slug(concern.get("concern"))
    if response_region == "members":
        response_contract = (
            "Call emit_java_structure exactly once. Do not write a Java region string. "
            "Use only the structured categories exposed by the required tool schema for this "
            "concern. The host owns Java syntax and renders those parts. Logic concerns are "
            "intentionally denied nested-type categories; do not work around that restriction. "
            "Every concern-owned domain type you reference must be declared in records/enums/classes "
            "in the same tool call unless it already exists in available_sibling_api or dependency_api. "
            "If this concern needs state not present in available_sibling_api, declare the minimal "
            "concern-local backing field instead of referencing an undeclared symbol. "
            "For JDK collection/concurrency types you may use simple names such as List/Map/Set; "
            "the host qualifies them. Keep each method body short and concern-local."
        )
    elif response_region == "initialize":
        response_contract = (
            "Call emit_java_structure exactly once with only its statements array. "
            "Each entry is one statement or one complete block that belongs inside the host-owned "
            "initialize() body. The host owns initialize() syntax. Use an empty statements array "
            "when no initialization is required."
        )
    else:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_RESPONSE_REGION_INVALID: {response_region!r}"
        )
    system = (
        "Implement exactly one host-selected concern inside one already-selected Java class. "
        "Planning record schemas, planning task labels, and concern cardinality are host-owned "
        "metadata and are deliberately not exposed as Java source shapes. Source requirement labels "
        "such as owner/type/unit/default/domain/from_state/trigger/guard describe semantics; they "
        "are not a request to create a Java metadata record with those labels as components. "
        "Create a record/class only when the runtime gameplay implementation itself needs that data object. "
        "You do not choose files, classes, dependencies, architecture, tools, search routes, APIs, or sibling work. "
        + response_contract + " "
        "The task_authority source requirements are already host-sliced to this concern; "
        "current_selected_region_source is the only region you may replace. "
        "available_sibling_api contains authoritative compiled Java declarations from "
        "earlier concerns: use their exact symbol spelling, declared type, signature, "
        "and mutability. Never treat an object/record field as a primitive, never assign "
        "to a field declared final, and never invent a sibling symbol that is not listed. "
        "Do not implement sibling concerns. "
        "Use fully-qualified external API names when needed. "
        "Use only supplied host grounding and dependency_api; never invent a Minecraft/Fabric API."
    )
    payload = {
        "phase": "implement_atomic_concern_region",
        "section": section,
        "response_region": response_region,
        "host_selected_class": str(host_symbol or "").strip(),
        "concern": {
            "sequence": concern.get("sequence"),
            "identifier": concern.get("identifier"),
            "name": name,
            "implementation_goal": (
                f"Implement only the {name} semantics stated in "
                "task_authority.source_requirements inside the selected Java class."
            ),
        },
        "task_authority": _concern_authority(task, concern),
        "state_variable_contract": (
            list(_state_variable_contract(task, concern))
            if section == "state_model" and name == "variables"
            else []
        ),
        "host_grounding": _bounded_grounding(grounding),
        "dependency_api": _dependency_api_context(dependency_source),
        "current_selected_region_source": _region_content(
            current_source,
            concern=name,
            region="INIT" if response_region == "initialize" else "MEMBERS",
        ),
        "available_sibling_api": _sibling_symbol_inventory(
            current_source,
            sibling_concerns=sibling_concerns,
        ),
        "repair_failure": failure or None,
        "generation_recipe": {
            "first_pass_goal": (
                "produce the smallest compile-ready structured Java components in one call "
                "and finish the required tool call well inside the finite output page"
            ),
            "declare_local_domain_types_first": True,
            "jdk_simple_names_host_qualified": [
                "List",
                "Map",
                "Set",
                "Optional",
                "UUID",
                "ArrayList",
                "HashMap",
                "HashSet",
                "ConcurrentHashMap",
                "AtomicBoolean",
                "AtomicInteger",
                "AtomicLong",
            ],
            "no_raw_top_level_java": True,
            "sibling_api_is_authoritative": True,
            "never_mutate_final_sibling_fields": True,
            "declare_missing_concern_local_state": (
                "When this concern reads or writes state absent from available_sibling_api, "
                "declare a private static non-final backing field with a compatible runtime "
                "value type instead of referencing an undeclared symbol or inventing a metadata DTO."
            ),
            "preferred_shape": (
                "fields_and_local_types"
                if name in {"variables", "inputs", "outputs", "stored_state", "payloads"}
                else "methods_and_constants"
                if name in {
                    "transitions",
                    "invariants",
                    "initialization",
                    "updates",
                    "cleanup",
                    "concurrency",
                    "preconditions",
                    "success_postconditions",
                    "rejection_postconditions",
                    "security_checks",
                    "synchronization",
                    "bounds",
                }
                else "smallest_components_that_satisfy_this_concern"
            ),
        },
        "scope": {
            "selected_region": _marker(
                name,
                "INIT" if response_region == "initialize" else "MEMBERS",
                "START",
            ),
            "sibling_regions_immutable": True,
            "required_output_tool": "emit_java_structure",
            "model_tool_choice": False,
            "sibling_concerns_out_of_scope": list(sibling_concerns),
            "scope_rule": (
                "Implement only the selected concern and only the lines in "
                "task_authority.source_requirements. Do not pre-implement sibling concerns. "
                "The host owns declaration deduplication and sibling bookkeeping."
            ),
            "repair_structure_rule": (
                "Compiler repair may remove or edit existing nested types but must not add, "
                "rename, or change the kind of nested types."
                if failure
                else None
            ),
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]

@dataclass
class AtomicConcernExecutor:
    root: Path
    target: Path
    relative: str
    symbol: str
    original: str
    task: Mapping[str, Any]
    section: str
    concerns: Sequence[Mapping[str, Any]]
    grounding: Mapping[str, Any]
    dependency_source: str
    require_initialize: bool
    call_coder: Callable[[Sequence[Mapping[str, str]]], str]
    compile_java: Callable[[Path], Any]
    compile_log: Callable[[Any], str]
    write_source: Callable[[Path, str], None]
    ordered: tuple[dict[str, Any], ...] = field(init=False)
    source: str = field(init=False)
    state: dict[str, tuple[str, str]] = field(default_factory=dict, init=False)
    summaries: list[str] = field(default_factory=list, init=False)
    seen_failures: set[str] = field(default_factory=set, init=False)
    repairs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.ordered = _validate_concerns(self.concerns)
        self.source = build_concern_scaffold(
            self.original,
            symbol=self.symbol,
            concerns=self.ordered,
            require_initialize=self.require_initialize,
        )

    def _concern(self, name: str) -> dict[str, Any]:
        for item in self.ordered:
            if _slug(item["concern"]) == name:
                return item
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_UNKNOWN_REPAIR_SCOPE: {name}"
        )

    def _sibling_symbol_owners(self, *, exclude: str) -> dict[str, str]:
        owners: dict[str, str] = {}
        for concern_name, (members, _initialize) in self.state.items():
            if concern_name == exclude:
                continue
            for key in _member_declaration_symbols(members):
                owners.setdefault(key, concern_name)
        return owners


    def _remove_owned_symbols(
        self,
        *,
        members: str,
        keys: set[str],
    ) -> tuple[str, set[str], set[str]]:
        kept: list[str] = []
        removed: set[str] = set()
        unresolved: set[str] = set()
        for chunk in _top_level_member_chunks(members):
            declared = set(_member_declaration_symbols(chunk))
            overlap = declared & keys
            if not overlap:
                kept.append(chunk)
                continue
            if declared and declared <= keys:
                removed.update(declared)
                continue
            kept.append(chunk)
            unresolved.update(overlap)
        return "\n".join(kept).strip(), removed, unresolved

    def _rehome_symbol_collisions(self, *, concern: str, members: str) -> None:
        current = set(_member_declaration_symbols(members))
        if not current:
            return

        owners = self._sibling_symbol_owners(exclude=concern)
        by_owner: dict[str, set[str]] = {}
        for key in current:
            owner = owners.get(key)
            if owner:
                by_owner.setdefault(owner, set()).add(key)
        if not by_owner:
            return

        plans: list[tuple[str, str, str, set[str]]] = []
        for owner, keys in sorted(by_owner.items()):
            owner_members, owner_initialize = self.state[owner]
            rewritten, removed, unresolved = self._remove_owned_symbols(
                members=owner_members,
                keys=keys,
            )
            missing = keys - removed
            if unresolved or missing:
                details = []
                for key in sorted(unresolved | missing):
                    kind, _, display = key.partition(":")
                    details.append(f"{kind} {display} is entangled in concern {owner}")
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_SYMBOL_COLLISION: " + "; ".join(details)
                )
            plans.append((owner, rewritten, owner_initialize, removed))

        for owner, rewritten, owner_initialize, removed in plans:
            self.source = _replace_region(
                self.source,
                concern=owner,
                region="MEMBERS",
                content=rewritten,
            )
            self.state[owner] = (rewritten, owner_initialize)

            from .root_cause_trace import emit_root_cause

            emit_root_cause(
                "atomic_concern_symbols_rehomed",
                stage="production",
                operation="atomic_concern_region",
                gate="host_symbol_ownership",
                result="PASS",
                details={
                    "from_concern": owner,
                    "to_concern": concern,
                    "symbols": [
                        key.partition(":")[2]
                        for key in sorted(removed)
                    ],
                },
            )

    def _known_simple_types(self, *, exclude: str) -> tuple[str, ...]:
        names: set[str] = {self.symbol}
        for concern_name, (members, _initialize) in self.state.items():
            if concern_name == exclude:
                continue
            for key, display in _member_declaration_symbols(members).items():
                if key.startswith(("type:", "field:")):
                    names.add(display)
        names.update(_dependency_declared_identifiers(self.dependency_source))
        return tuple(sorted(names))

    def _generate_region(
        self,
        concern: Mapping[str, Any],
        *,
        response_region: str,
        failure: str = "",
    ) -> str:
        name = _slug(concern["concern"])
        seen_violations: set[tuple[str, str]] = set()
        repair_failure = failure
        attempt_limit = _region_attempt_limit()

        if (
            not failure
            and response_region == "members"
            and str(self.section or "").strip() == "state_model"
            and name == "variables"
        ):
            host_members = _deterministic_state_variable_members(
                self.task,
                concern,
            )
            if host_members:
                output_sha = hashlib.sha256(
                    host_members.encode("utf-8")
                ).hexdigest()
                _trace_region_generation(
                    "atomic_concern_region_host_lowered",
                    result="PASS",
                    concern=name,
                    region=response_region,
                    attempt=1,
                    attempt_limit=1,
                    output_sha256=output_sha,
                    output_chars=len(host_members),
                )
                return host_members

        for attempt in range(1, attempt_limit + 1):
            _trace_region_generation(
                "atomic_concern_region_attempt",
                result="START",
                concern=name,
                region=response_region,
                attempt=attempt,
                attempt_limit=attempt_limit,
            )
            output_text = ""
            output_sha = hashlib.sha256(b"").hexdigest()
            try:
                output = self.call_coder(_messages(
                    section=self.section,
                    concern=concern,
                    task=self.task,
                    grounding=self.grounding,
                    dependency_source=self.dependency_source,
                    current_source=self.source,
                    response_region=response_region,
                    failure=repair_failure,
                    sibling_concerns=tuple(
                        _slug(item["concern"])
                        for item in self.ordered
                        if _slug(item["concern"]) != name
                    ),
                    host_symbol=self.symbol,
                ))
                output_text = str(output or "")
                output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                parsed = _parse_region_content(
                    output,
                    response_region=response_region,
                )
                if failure and response_region == "members":
                    previous_region = _region_content(
                        self.source,
                        concern=name,
                        region="MEMBERS",
                    )
                    structure_error = _repair_type_structure_error(
                        previous_region,
                        parsed,
                    )
                    if structure_error:
                        raise CustomModuleGenerationError(
                            "ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE: "
                            + structure_error
                        )
            except CustomModuleGenerationError as exc:
                reason = str(exc).split("\n", 1)[0]
                recoverable = reason.startswith(
                    (
                        "ATOMIC_CONCERN_RESPONSE_INVALID:",
                        "ATOMIC_CONCERN_SCOPE_ESCAPE:",
                        "ATOMIC_CONCERN_REPAIR_STRUCTURE_ESCAPE:",
                        "ATOMIC_CONCERN_SYMBOL_COLLISION:",
                        "ATOMIC_CONCERN_OUTPUT_EXHAUSTED:",
                    )
                )
                if not recoverable:
                    _trace_region_generation(
                        "atomic_concern_region_rejected",
                        result="FAIL",
                        concern=name,
                        region=response_region,
                        attempt=attempt,
                        attempt_limit=attempt_limit,
                        reason=reason,
                        output_sha256=output_sha,
                        output_chars=len(output_text),
                    )
                    raise

                violation = (reason, output_sha)
                if violation in seen_violations:
                    _trace_region_generation(
                        "atomic_concern_region_no_progress",
                        result="FAIL",
                        concern=name,
                        region=response_region,
                        attempt=attempt,
                        attempt_limit=attempt_limit,
                        reason=reason,
                        output_sha256=output_sha,
                        output_chars=len(output_text),
                    )
                    raise CustomModuleGenerationError(
                        f"ATOMIC_CONCERN_RESPONSE_NO_PROGRESS: {name}:{response_region} "
                        f"repeated identical invalid output: {reason}"
                    ) from exc

                seen_violations.add(violation)
                _trace_region_generation(
                    "atomic_concern_region_rejected",
                    result="RETRY" if attempt < attempt_limit else "FAIL",
                    concern=name,
                    region=response_region,
                    attempt=attempt,
                    attempt_limit=attempt_limit,
                    reason=reason,
                    output_sha256=output_sha,
                    output_chars=len(output_text),
                )
                if attempt >= attempt_limit:
                    raise CustomModuleGenerationError(
                        "ATOMIC_CONCERN_RESPONSE_RETRY_EXHAUSTED: "
                        f"{name}:{response_region} remained invalid after "
                        f"{attempt_limit} bounded attempts. Last failure: {reason}"
                    ) from exc

                validation_failure = (
                    "HOST REGION VALIDATION FAILED BEFORE COMPILATION:\n"
                    + reason
                    + f"\nRegenerate only the {response_region} region. "
                    "Do not emit response markers, prose, package/import/top-level/lifecycle declarations. "
                    "Do not introduce, rename, or change the kind of nested types during compiler repair. "
                    "Fix fields, method signatures, modifiers, expressions, and method bodies in place. "
                    "Implement only this concern; do not add declarations for sibling concerns. "
                    "The host will reconcile declarations emitted earlier by another concern. "
                    "Emit executable Java only; no analysis, reasoning, plans, or Markdown commentary."
                )
                repair_failure = "\n\n".join(
                    item for item in (failure, validation_failure) if item
                )
                continue

            _trace_region_generation(
                "atomic_concern_region_accepted",
                result="PASS",
                concern=name,
                region=response_region,
                attempt=attempt,
                attempt_limit=attempt_limit,
                output_sha256=output_sha,
                output_chars=len(output_text),
            )
            return parsed

        raise AssertionError("atomic concern region attempt loop terminated unexpectedly")

    def _apply(self, concern: Mapping[str, Any], *, failure: str = "") -> None:
        name = _slug(concern["concern"])
        members = self._generate_region(
            concern,
            response_region="members",
            failure=failure,
        )
        initialize = ""
        if self.require_initialize and str(self.section or "").strip() == "integration":
            initialize = self._generate_region(
                concern,
                response_region="initialize",
                failure=failure,
            )
        if failure and self.state.get(name) == (members, initialize):
            raise CustomModuleGenerationError(
                f"ATOMIC_CONCERN_REPAIR_NO_PROGRESS: {name} repeated the same bounded source."
            )
        self._rehome_symbol_collisions(concern=name, members=members)
        self.source = _replace_region(
            self.source, concern=name, region="MEMBERS", content=members
        )
        if self.require_initialize:
            self.source = _replace_region(
                self.source, concern=name, region="INIT", content=initialize
            )
        self.state[name] = (members, initialize)
        label = f"{name} repair" if failure else name
        self.summaries.append(label)

    def _compile(self) -> Any:
        self.write_source(self.target, self.source)
        return self.compile_java(self.root)

    def _repair_once(self, report: Any) -> Any:
        failure = self.compile_log(report) or str(
            getattr(report, "error", "") or "Gradle compileJava failed."
        )
        fingerprint = _compiler_diagnostic_fingerprint(failure)
        if fingerprint in self.seen_failures:
            repeated_name = _failure_concern(
                self.source,
                log=failure,
                relative=self.relative,
            )
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_NO_PROGRESS: compiler diagnostics repeated.\n"
                + _compact_compiler_failure(
                    failure,
                    source=self.source,
                    relative=self.relative,
                    concern=repeated_name,
                )
            )
        self.seen_failures.add(fingerprint)
        name = _failure_concern(self.source, log=failure, relative=self.relative)
        if not name:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_UNLOCALIZED: failure is outside every active concern region.\n"
                + _compact_compiler_failure(
                    failure,
                    source=self.source,
                    relative=self.relative,
                    concern="",
                )
            )
        compact_failure = _compact_compiler_failure(
            failure,
            source=self.source,
            relative=self.relative,
            concern=name,
        )
        self._apply(self._concern(name), failure=compact_failure)
        self.repairs += 1
        return self._compile()

    def run(self) -> dict[str, Any]:
        repair_limit = _compile_repair_limit()
        for concern in self.ordered:
            name = _slug(concern["concern"])
            self.seen_failures.clear()
            self._apply(concern)
            report = self._compile()
            concern_repairs = 0
            while getattr(report, "status", "") != "PASS":
                if concern_repairs >= repair_limit:
                    failure = self.compile_log(report) or str(
                        getattr(report, "error", "") or "Gradle compileJava failed."
                    )
                    failing_name = _failure_concern(
                        self.source,
                        log=failure,
                        relative=self.relative,
                    ) or name
                    raise CustomModuleGenerationError(
                        "ATOMIC_CONCERN_COMPILE_RETRY_EXHAUSTED: "
                        f"{failing_name} remained uncompilable after {repair_limit} "
                        "bounded compiler-driven repairs.\n"
                        + _compact_compiler_failure(
                            failure,
                            source=self.source,
                            relative=self.relative,
                            concern=failing_name,
                        )
                    )
                report = self._repair_once(report)
                concern_repairs += 1
            self.seen_failures.clear()
        return {
            "source": self.source,
            "summary": " | ".join(self.summaries),
            "concern_count": len(self.ordered),
            "repair_count": self.repairs,
        }

__all__ = [
    "END_MARKER",
    "INITIALIZE_MARKER",
    "MEMBERS_MARKER",
    "build_concern_scaffold",
    "AtomicConcernExecutor",
    "parse_concern_content",
]
