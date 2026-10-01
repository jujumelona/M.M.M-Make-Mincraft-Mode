from __future__ import annotations

"""Host-owned concern regions for bounded small-model Java generation."""

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .authored_ir_parser import section_slug, slice_concern_requirements
from .custom_module_errors import AtomicJavaDecisionError, CustomModuleGenerationError
from .authored_execution_schema import section_spec
from .generation_implementation_grounding import render_generation_implementation_authority_prompt
from .java_generation_policy import (
    DEFAULT_COMPILE_REPAIR_LIMIT as _DEFAULT_COMPILE_REPAIR_LIMIT,
    DEFAULT_REGION_ATTEMPT_LIMIT as _DEFAULT_REGION_ATTEMPT_LIMIT,
    MAX_COMPILE_REPAIR_LIMIT as _MAX_COMPILE_REPAIR_LIMIT,
    MAX_REGION_ATTEMPT_LIMIT as _MAX_REGION_ATTEMPT_LIMIT,
    PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS,
    atomic_error_recoverable,
    atomic_error_terminal_after_normalization,
    production_java_generation_recipe_policy,
)
from .java_region_parser import (
    JavaRegionParseError,
    admit_initialize_region,
    admit_member_region,
    class_body_chunks,
    class_body_assignment_targets,
    class_body_direct_return_calls,
    class_body_member_contracts,
    class_body_member_kinds,
    class_body_method_invocation_details,
    class_body_method_invocations,
    class_body_object_creations,
    class_body_simple_type_occurrences,
    public_source_member_contracts,
    strict_initialize_statements,
    strict_member_chunks,
)

MEMBERS_MARKER = "<<<MMM_CONCERN_MEMBERS>>>"
INITIALIZE_MARKER = "<<<MMM_CONCERN_INITIALIZE>>>"
END_MARKER = "<<<MMM_CONCERN_END>>>"
_DECLARATION_ONLY_CONCERNS = frozenset({"stored_state"})
_DECLARATION_ONLY_MEMBER_KINDS = frozenset(
    {
        "field_declaration",
        "annotation_type_declaration",
        "class_declaration",
        "enum_declaration",
        "interface_declaration",
        "record_declaration",
    }
)


def _region_attempt_limit() -> int:
    """Read an optional override; production supplies its own correction budget."""

    raw = os.environ.get("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "").strip()
    if not raw:
        return _DEFAULT_REGION_ATTEMPT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_REGION_ATTEMPT_LIMIT
    return max(1, min(_MAX_REGION_ATTEMPT_LIMIT, value))


def _compile_repair_limit() -> int:
    """Disable model compiler-repair in production unless explicitly opted in."""

    raw = os.environ.get("MMM_ATOMIC_CONCERN_COMPILE_REPAIRS", "").strip()
    if not raw:
        return _DEFAULT_COMPILE_REPAIR_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_COMPILE_REPAIR_LIMIT
    return max(0, min(_MAX_COMPILE_REPAIR_LIMIT, value))


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
    rejected_response: str | None = None,
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
            **({"rejected_response": rejected_response} if rejected_response is not None else {}),
        },
    )


_HOST_PREFIX = "MMM_ATOMIC_CONCERN"
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*;\s*$")
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
    """Return Java class-body members from the Tree-sitter Java AST."""
    try:
        return class_body_chunks(value)
    except JavaRegionParseError as exc:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_JAVA_PARSE_INVALID: {exc}"
        ) from exc

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



def _drop_authoritative_sibling_redeclarations(
    value: str,
    *,
    owners: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Drop whole redundant declarations already owned by accepted sibling concerns.

    Earlier concern declarations are authoritative. A later concern may reference
    them, but a small coder can still copy those declarations into its own region.
    When an entire top-level member chunk declares only already-owned symbols, the
    host can deterministically discard that redundant chunk without changing the
    accepted sibling source. Mixed chunks remain untouched so the ownership gate
    can reject ambiguous partial collisions.
    """

    if not owners:
        return str(value or "").strip(), ()
    kept: list[str] = []
    dropped: set[str] = set()
    owner_keys = set(owners)
    for chunk in _top_level_member_chunks(value):
        symbols = set(_member_declaration_symbols(chunk))
        collisions = symbols & owner_keys
        if symbols and collisions == symbols:
            dropped.update(collisions)
            continue
        kept.append(chunk.strip())
    return "\n\n".join(chunk for chunk in kept if chunk).strip(), tuple(sorted(dropped))


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


_JDK_CANONICAL_SIMPLE_TYPES = {
    "Object": "java.lang.Object",
    "String": "java.lang.String",
    "Collection": "java.util.Collection",
    "List": "java.util.List",
    "Map": "java.util.Map",
    "Set": "java.util.Set",
    "Queue": "java.util.Queue",
    "Deque": "java.util.Deque",
    "ArrayList": "java.util.ArrayList",
    "HashMap": "java.util.HashMap",
    "LinkedHashMap": "java.util.LinkedHashMap",
    "HashSet": "java.util.HashSet",
    "LinkedHashSet": "java.util.LinkedHashSet",
    "ArrayDeque": "java.util.ArrayDeque",
    "ConcurrentHashMap": "java.util.concurrent.ConcurrentHashMap",
    "Lock": "java.util.concurrent.locks.Lock",
    "ReentrantLock": "java.util.concurrent.locks.ReentrantLock",
    "ReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "ReentrantReadWriteLock": "java.util.concurrent.locks.ReentrantReadWriteLock",
    "AtomicBoolean": "java.util.concurrent.atomic.AtomicBoolean",
    "AtomicInteger": "java.util.concurrent.atomic.AtomicInteger",
    "AtomicLong": "java.util.concurrent.atomic.AtomicLong",
}
_JDK_FQCN_ALIASES = {
    "java.util.ConcurrentHashMap": "java.util.concurrent.ConcurrentHashMap",
    "java.util.Lock": "java.util.concurrent.locks.Lock",
    "java.util.ReentrantLock": "java.util.concurrent.locks.ReentrantLock",
    "java.util.ReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "java.util.ReentrantReadWriteLock": "java.util.concurrent.locks.ReentrantReadWriteLock",
    "java.util.AtomicBoolean": "java.util.concurrent.atomic.AtomicBoolean",
    "java.util.AtomicInteger": "java.util.concurrent.atomic.AtomicInteger",
    "java.util.AtomicLong": "java.util.concurrent.atomic.AtomicLong",
    "java.util.concurrent.Lock": "java.util.concurrent.locks.Lock",
    "java.util.concurrent.ReentrantLock": "java.util.concurrent.locks.ReentrantLock",
    "java.util.concurrent.ReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "java.util.concurrent.ReentrantReadWriteLock": "java.util.concurrent.locks.ReentrantReadWriteLock",
}
_JDK_ASSIGNABLE_DECLARATIONS = {
    "java.util.ArrayList": frozenset(
        {"java.util.ArrayList", "java.util.List", "java.util.Collection", "java.lang.Object"}
    ),
    "java.util.HashMap": frozenset(
        {"java.util.HashMap", "java.util.Map", "java.lang.Object"}
    ),
    "java.util.LinkedHashMap": frozenset(
        {"java.util.LinkedHashMap", "java.util.Map", "java.lang.Object"}
    ),
    "java.util.concurrent.ConcurrentHashMap": frozenset(
        {"java.util.concurrent.ConcurrentHashMap", "java.util.Map", "java.lang.Object"}
    ),
    "java.util.HashSet": frozenset(
        {"java.util.HashSet", "java.util.Set", "java.util.Collection", "java.lang.Object"}
    ),
    "java.util.LinkedHashSet": frozenset(
        {"java.util.LinkedHashSet", "java.util.Set", "java.util.Collection", "java.lang.Object"}
    ),
    "java.util.ArrayDeque": frozenset(
        {
            "java.util.ArrayDeque",
            "java.util.Deque",
            "java.util.Queue",
            "java.util.Collection",
            "java.lang.Object",
        }
    ),
    "java.util.concurrent.locks.ReentrantLock": frozenset(
        {
            "java.util.concurrent.locks.ReentrantLock",
            "java.util.concurrent.locks.Lock",
            "java.lang.Object",
        }
    ),
    "java.util.concurrent.locks.ReentrantReadWriteLock": frozenset(
        {
            "java.util.concurrent.locks.ReentrantReadWriteLock",
            "java.util.concurrent.locks.ReadWriteLock",
            "java.lang.Object",
        }
    ),
    "java.util.concurrent.atomic.AtomicBoolean": frozenset(
        {"java.util.concurrent.atomic.AtomicBoolean", "java.lang.Object"}
    ),
    "java.util.concurrent.atomic.AtomicInteger": frozenset(
        {"java.util.concurrent.atomic.AtomicInteger", "java.lang.Object"}
    ),
    "java.util.concurrent.atomic.AtomicLong": frozenset(
        {"java.util.concurrent.atomic.AtomicLong", "java.lang.Object"}
    ),
}
_JDK_PREFERRED_DECLARATIONS = {
    "java.util.ArrayList": "java.util.List",
    "java.util.HashMap": "java.util.Map",
    "java.util.LinkedHashMap": "java.util.Map",
    "java.util.concurrent.ConcurrentHashMap": "java.util.Map",
    "java.util.HashSet": "java.util.Set",
    "java.util.LinkedHashSet": "java.util.Set",
    "java.util.ArrayDeque": "java.util.Deque",
    "java.util.concurrent.locks.ReentrantLock": "java.util.concurrent.locks.Lock",
    "java.util.concurrent.locks.ReentrantReadWriteLock": "java.util.concurrent.locks.ReadWriteLock",
    "java.util.concurrent.atomic.AtomicBoolean": "java.util.concurrent.atomic.AtomicBoolean",
    "java.util.concurrent.atomic.AtomicInteger": "java.util.concurrent.atomic.AtomicInteger",
    "java.util.concurrent.atomic.AtomicLong": "java.util.concurrent.atomic.AtomicLong",
}
_JDK_GENERIC_ARITY = {
    "java.util.Collection": 1,
    "java.util.List": 1,
    "java.util.Set": 1,
    "java.util.Queue": 1,
    "java.util.Deque": 1,
    "java.util.ArrayList": 1,
    "java.util.HashSet": 1,
    "java.util.LinkedHashSet": 1,
    "java.util.ArrayDeque": 1,
    "java.util.Map": 2,
    "java.util.HashMap": 2,
    "java.util.LinkedHashMap": 2,
    "java.util.concurrent.ConcurrentHashMap": 2,
}
_KNOWN_CANONICAL_JDK_TYPES = frozenset(
    set(_JDK_CANONICAL_SIMPLE_TYPES.values())
    | set(_JDK_FQCN_ALIASES.values())
    | set(_JDK_ASSIGNABLE_DECLARATIONS)
    | set().union(*_JDK_ASSIGNABLE_DECLARATIONS.values())
)

_JDK_LOCK_RECEIVER_METHODS = frozenset(
    {"lock", "unlock", "tryLock", "lockInterruptibly", "newCondition"}
)
_FIELD_MODIFIERS = frozenset(
    {"public", "protected", "private", "static", "final", "volatile", "transient"}
)


def _canonical_jdk_class_name(value: str) -> str:
    raw = re.sub(r"\s+", "", _erase_generic_arguments(str(value or ""))).strip()
    if raw in _JDK_FQCN_ALIASES:
        return _JDK_FQCN_ALIASES[raw]
    if raw in _JDK_CANONICAL_SIMPLE_TYPES:
        return _JDK_CANONICAL_SIMPLE_TYPES[raw]
    # Only consult the installed JDK image for a previously unknown simple name.
    # Exact FQCN validation is handled by the declaration-authority gate below.
    if "." in raw or "$" in raw:
        return raw
    try:
        from .jdk_type_index import canonical_public_jdk_type

        return canonical_public_jdk_type(raw)
    except (OSError, RuntimeError, ValueError):
        return raw


def _rewrite_known_jdk_fqcns(value: str) -> str:
    source = str(value or "")
    aliases = tuple(
        sorted(_JDK_FQCN_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    )
    out: list[str] = []
    index = 0
    quote = ""
    line_comment = False
    block_comment = False
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            out.append(char)
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            out.append(char)
            if char == "*" and nxt == "/":
                out.append(nxt)
                index += 2
                block_comment = False
            else:
                index += 1
            continue
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < len(source):
                out.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and nxt == "/":
            out.extend((char, nxt))
            index += 2
            line_comment = True
            continue
        if char == "/" and nxt == "*":
            out.extend((char, nxt))
            index += 2
            block_comment = True
            continue
        matched = False
        for alias, canonical in aliases:
            if not source.startswith(alias, index):
                continue
            end = index + len(alias)
            before = source[index - 1] if index else ""
            after = source[end] if end < len(source) else ""
            if (
                (before and (before.isalnum() or before in "_$"))
                or (after and (after.isalnum() or after in "_$"))
            ):
                continue
            out.append(canonical)
            index = end
            matched = True
            break
        if matched:
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _single_initialized_field_parts(
    chunk: str,
) -> tuple[str, str, str] | None:
    declaration = str(chunk or "").strip()
    if not declaration.endswith(";"):
        return None
    body = declaration[:-1].strip()
    assignment = _top_level_assignment_index(body)
    if assignment < 0:
        return None
    left = body[:assignment].strip()
    initializer = body[assignment + 1 :].strip()
    identifiers = _JAVA_IDENTIFIER.findall(_structure_scan(left))
    if not identifiers:
        return None
    field_name = identifiers[-1]
    name_match = re.search(rf"\b{re.escape(field_name)}\s*$", left)
    if name_match is None:
        return None
    prefix = left[: name_match.start()].strip()
    tokens = prefix.split()
    while tokens and tokens[0] in _FIELD_MODIFIERS:
        tokens.pop(0)
    declared_type = " ".join(tokens).strip()
    if not declared_type or "@" in declared_type:
        return None
    return declared_type, field_name, initializer


def _initializer_constructor_type(
    initializer: str,
) -> tuple[str, str]:
    match = re.match(
        r"^new\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*(?:<[^()]*>)?\s*\(",
        str(initializer or "").strip(),
    )
    if match is None:
        return "", ""
    raw = match.group(1)
    return raw, _canonical_jdk_class_name(raw)


def _receiver_method_names(source: str, field_name: str) -> frozenset[str]:
    return frozenset(
        match.group(1)
        for match in re.finditer(
            rf"\b{re.escape(field_name)}\s*\.\s*"
            r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
            _structure_scan(source),
        )
    )


def _generic_arguments(value: str) -> tuple[str, ...]:
    text = str(value or "").strip()
    start = text.find("<")
    if start < 0:
        return ()
    depth = 0
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        return ()
    inner = text[start + 1 : end]
    return tuple(
        part.strip()
        for part in _split_top_level(inner, ",")
        if part.strip()
    )


def _render_canonical_declared_type(target: str, original: str) -> str:
    arity = _JDK_GENERIC_ARITY.get(target)
    if not arity:
        return target
    arguments = _generic_arguments(original)
    if len(arguments) != arity:
        return target
    return target + "<" + ", ".join(arguments) + ">"


def _canonicalize_tree_sitter_jdk_types(
    value: str,
    *,
    protected_simple_types: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Rewrite only AST-proven simple JDK type identifiers to canonical FQCNs."""

    source = str(value or "")
    if not source.strip():
        return source, ()
    protected = {
        str(name or "").strip()
        for name in protected_simple_types
        if str(name or "").strip()
    }
    try:
        contracts = class_body_member_contracts(source)
    except JavaRegionParseError:
        return source, ()
    protected.update(
        str(item.get("symbol") or "").strip()
        for item in contracts
        if item.get("kind") == "type" and str(item.get("symbol") or "").strip()
    )

    replacements: list[tuple[int, int, str, str]] = []
    for occurrence in class_body_simple_type_occurrences(source):
        name = str(occurrence.get("name") or "").strip()
        if not name or name in protected:
            continue
        canonical = _canonical_jdk_class_name(name)
        if (
            canonical == name
            or canonical.startswith("java.lang.")
            or "." not in canonical
        ):
            continue
        replacements.append(
            (
                int(occurrence["start_byte"]),
                int(occurrence["end_byte"]),
                name,
                canonical,
            )
        )

    if not replacements:
        return source, ()

    raw = source.encode("utf-8")
    changes: list[str] = []
    for start, end, name, canonical in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        raw = raw[:start] + canonical.encode("utf-8") + raw[end:]
        changes.append(f"type:{name}->{canonical}")
    return raw.decode("utf-8"), tuple(reversed(changes))


def _canonicalize_local_final_rebindings(
    value: str,
) -> tuple[str, tuple[str, ...]]:
    """Remove final only from concern-local initialized fields that this region rebinds.

    The model can emit an initialized private static final backing field together
    with a setter that reassigns the field. Because both declaration and assignment
    are owned by the same atomic region, the host preserves the authored rebinding
    semantics by dropping only that local final modifier. Sibling/dependency fields
    are never rewritten here.
    """

    source = str(value or "").strip()
    if not source:
        return source, ()

    assigned = set(class_body_assignment_targets(source))
    if not assigned:
        return source, ()

    chunks = class_body_chunks(source)
    kinds = class_body_member_kinds(source)
    if len(chunks) != len(kinds):
        return source, ()

    changes: list[str] = []
    rendered: list[str] = []
    for chunk, kind in zip(chunks, kinds, strict=True):
        if kind != "field_declaration":
            rendered.append(chunk)
            continue

        try:
            fields = tuple(
                row
                for row in class_body_member_contracts(chunk)
                if row.get("kind") == "field"
            )
        except JavaRegionParseError:
            rendered.append(chunk)
            continue

        if len(fields) != 1:
            rendered.append(chunk)
            continue

        field = fields[0]
        symbol = str(field.get("symbol") or "").strip()
        if (
            not symbol
            or symbol not in assigned
            or field.get("mutable") is not False
            or field.get("initialized") is not True
        ):
            rendered.append(chunk)
            continue

        normalized = re.sub(r"\bfinal\s+", "", chunk, count=1)
        if normalized == chunk:
            rendered.append(chunk)
            continue
        rendered.append(normalized)
        changes.append(f"{symbol}:final->mutable")

    normalized = "\n\n".join(
        item.strip() for item in rendered if item.strip()
    ).strip()
    if changes:
        try:
            class_body_member_contracts(normalized)
        except JavaRegionParseError:
            return source, ()
    return normalized, tuple(changes)


def _canonicalize_jdk_construction_semantics(
    value: str,
) -> tuple[str, tuple[str, ...]]:
    """Lower impossible JDK constructor calls to unambiguous same-type factories.

    Authority comes from the installed JDK via javap. Rewriting occurs only when
    the requested constructor is not public/valid and exactly one public static
    factory method returning the same type accepts the same argument count.
    Ambiguous cases remain untouched for the semantic validator.
    """

    source = str(value or "").strip()
    if not source:
        return source, ()

    try:
        from .jdk_type_index import (
            is_public_jdk_type,
            public_jdk_constructor_shapes,
            public_jdk_static_factory_shapes,
        )
    except ImportError:
        return source, ()

    replacements: list[tuple[int, int, str, str]] = []
    for creation in class_body_object_creations(source):
        raw_type = str(creation.get("type") or "").strip()
        fqcn = _canonical_jdk_class_name(raw_type)
        if not fqcn.startswith(("java.", "javax.")):
            continue
        try:
            if not is_public_jdk_type(fqcn):
                continue
            constructors = public_jdk_constructor_shapes(fqcn)
            factories = public_jdk_static_factory_shapes(fqcn)
        except (OSError, RuntimeError, ValueError):
            continue

        argument_count = int(creation.get("argument_count") or 0)
        if constructors and _jdk_constructor_accepts_arity(
            constructors, argument_count
        ):
            continue

        matching = {
            str(factory.get("name") or "").strip()
            for factory in factories
            if str(factory.get("name") or "").strip()
            and _jdk_constructor_accepts_arity((factory,), argument_count)
        }
        if len(matching) != 1:
            continue

        start = int(creation.get("start_byte", -1))
        end = int(creation.get("end_byte", -1))
        if start < 0 or end <= start:
            continue
        method = next(iter(matching))
        arguments = ", ".join(
            str(item) for item in creation.get("arguments") or ()
        )
        replacement = f"{fqcn}.{method}({arguments})"
        replacements.append(
            (
                start,
                end,
                replacement,
                f"{fqcn}:constructor->{method}",
            )
        )

    if not replacements:
        return source, ()

    raw = source.encode("utf-8")
    changes: list[str] = []
    for start, end, replacement, change in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        raw = raw[:start] + replacement.encode("utf-8") + raw[end:]
        changes.append(change)
    normalized = raw.decode("utf-8")
    try:
        class_body_member_contracts(normalized)
    except JavaRegionParseError:
        return source, ()
    return normalized, tuple(reversed(changes))


def _canonicalize_generated_jdk_semantics(
    value: str,
    *,
    authoritative_field_types: Mapping[str, str] | None = None,
    protected_simple_types: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    """Canonicalize known JDK runtime types before the first compiler invocation.

    The small coder still authors complete Java regions, but the host owns canonical
    JDK spellings and obvious constructor/declared-type compatibility. This keeps
    the first compile from failing on deterministic standard-library facts.
    """

    source = _rewrite_known_jdk_fqcns(value)
    source, type_changes = _canonicalize_tree_sitter_jdk_types(
        source,
        protected_simple_types=protected_simple_types,
    )
    source, construction_changes = _canonicalize_jdk_construction_semantics(source)
    chunks = class_body_chunks(source)
    kinds = class_body_member_kinds(source)
    if len(chunks) != len(kinds):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_JAVA_PARSE_INVALID: member chunk/kind cardinality drift"
        )

    changes: list[str] = [*type_changes, *construction_changes]
    rendered: list[str] = []
    for chunk, kind in zip(chunks, kinds, strict=True):
        if kind != "field_declaration":
            rendered.append(chunk)
            continue
        parts = _single_initialized_field_parts(chunk)
        if parts is None:
            rendered.append(chunk)
            continue
        declared_type, field_name, initializer = parts
        constructor_raw, constructor_type = _initializer_constructor_type(initializer)
        if not constructor_type:
            rendered.append(chunk)
            continue
        assignable = _JDK_ASSIGNABLE_DECLARATIONS.get(constructor_type)
        if not assignable:
            rendered.append(chunk)
            continue

        declared_raw = re.sub(
            r"\s+",
            "",
            _erase_generic_arguments(declared_type),
        ).strip()
        declared_class = _canonical_jdk_class_name(declared_type)
        receiver_methods = _receiver_method_names(source, field_name)
        target_type = (
            declared_class
            if declared_raw != declared_class
            and declared_class in set().union(*_JDK_ASSIGNABLE_DECLARATIONS.values())
            else ""
        )
        if (
            receiver_methods & _JDK_LOCK_RECEIVER_METHODS
            and constructor_type
            in {
                "java.util.concurrent.locks.ReentrantLock",
                "java.util.concurrent.locks.ReentrantReadWriteLock",
            }
        ):
            target_type = _JDK_PREFERRED_DECLARATIONS[constructor_type]
        elif declared_class not in assignable:
            target_type = _JDK_PREFERRED_DECLARATIONS.get(
                constructor_type,
                constructor_type,
            )
        elif declared_class == "java.lang.Object" and receiver_methods:
            target_type = _JDK_PREFERRED_DECLARATIONS.get(
                constructor_type,
                constructor_type,
            )

        normalized_chunk = chunk
        if constructor_raw and constructor_raw != constructor_type:
            normalized_chunk = re.sub(
                rf"\bnew\s+{re.escape(constructor_raw)}(?=\s*(?:<[^()]*>)?\s*\()",
                "new " + constructor_type,
                normalized_chunk,
                count=1,
            )
            changes.append(
                f"{field_name}:initializer_type:{constructor_raw}->{constructor_type}"
            )
        if target_type:
            rendered_type = _render_canonical_declared_type(
                target_type,
                declared_type,
            )
            normalized_chunk = normalized_chunk.replace(
                declared_type,
                rendered_type,
                1,
            )
            changes.append(
                f"{field_name}:declared_type:{declared_type}->{rendered_type}"
            )
        if normalized_chunk != chunk:
            normalized_chunk = _rewrite_known_jdk_fqcns(normalized_chunk)
        rendered.append(normalized_chunk)

    normalized = "\n\n".join(item.strip() for item in rendered if item.strip()).strip()
    pre_projection = normalized
    projected, projection_changes = _rewrite_map_entry_projection_types(
        normalized,
        authoritative_field_types=dict(authoritative_field_types or {}),
    )
    try:
        class_body_chunks(projected)
        class_body_member_kinds(projected)
    except JavaRegionParseError:
        normalized = pre_projection
        projection_changes = ()
    else:
        normalized = projected
    changes.extend(projection_changes)
    if normalized != value and not changes:
        changes.append("canonical_jdk_fqcn")
    return normalized, tuple(changes)


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
    """Validate an already-admitted region using Java AST structure only.

    Model-envelope recovery belongs to java_region_parser. Once that parser has
    admitted a region, lexical words and identifier spellings are never treated
    as protocol/prose signals. This prevents valid Java identifiers such as
    next from being rejected by English-language heuristics.
    """

    region = "initialize body" if initialize_region else "concern members"
    try:
        if initialize_region:
            strict_initialize_statements(value)
        else:
            strict_member_chunks(value)
    except JavaRegionParseError as exc:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_SCOPE_ESCAPE: {region} is not admissible Java: {exc}"
        ) from exc


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


def _project_declaration_only_members(value: str) -> tuple[str, tuple[str, ...]]:
    """Keep only Tree-sitter-proven declarations allowed by a data-only concern."""

    chunks = class_body_chunks(value)
    kinds = class_body_member_kinds(value)
    if len(chunks) != len(kinds):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_JAVA_PARSE_INVALID: member chunk/kind cardinality drift"
        )
    kept = [
        chunk
        for chunk, kind in zip(chunks, kinds, strict=True)
        if kind in _DECLARATION_ONLY_MEMBER_KINDS
    ]
    dropped = tuple(
        kind for kind in kinds if kind not in _DECLARATION_ONLY_MEMBER_KINDS
    )
    projected = "\n\n".join(kept).strip()
    if projected:
        _validate_region_text(projected, initialize_region=False)
    return projected, dropped


def _parse_region_content(
    text: str,
    *,
    response_region: str,
    allow_inert_empty: bool = False,
    allow_host_initialize_only_empty: bool = False,
) -> str:
    """Admit one host-selected region through Markdown and Java parsers in order."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    exact_markers = {
        MEMBERS_MARKER,
        INITIALIZE_MARKER,
        END_MARKER,
    }
    raw_value = "\n".join(
        line for line in raw.splitlines() if line.strip() not in exact_markers
    ).strip()
    initialize_region = response_region == "initialize"
    if response_region not in {"members", "initialize"}:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_RESPONSE_REGION_INVALID: {response_region!r}"
        )

    # Java envelope handling is centralized in java_region_parser for both
    # member and initialize regions. Integration members may be intentionally
    # empty because their executable lifecycle work is generated separately.
    if (initialize_region or allow_inert_empty) and _is_inert_empty_region(
        _normalize_region_text(raw_value)
    ):
        return ""
    try:
        value = (
            admit_initialize_region(raw_value)
            if initialize_region
            else admit_member_region(
                raw_value,
                allow_host_initialize_only_empty=allow_host_initialize_only_empty,
            )
        )
    except JavaRegionParseError as exc:
        region = "initialize body" if initialize_region else "concern members"
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_SCOPE_ESCAPE: {region} could not be admitted by the Java parser: {exc}"
        ) from exc
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
    rendered_log = str(log or "")
    for match in re.finditer(rf"(?:^|[\\/]){filename}:(\d+)(?::\d+)?", rendered_log):
        concern = _concern_at_line(source, int(match.group(1)))
        if concern:
            return concern

    # javac reports some declaration-owned failures at a use site outside the
    # declaration region. The common case is definite assignment of a blank
    # final, which is reported at the constructor even though the field belongs
    # to one atomic concern. Route those diagnostics by the named symbol before
    # declaring the failure unlocalized.
    symbols: list[str] = []
    for pattern in (
        r"variable\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+might not have been initialized",
        r"cannot assign a value to (?:static )?final variable\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    ):
        for match in re.finditer(pattern, rendered_log, re.IGNORECASE):
            symbol = match.group(1)
            if symbol not in symbols:
                symbols.append(symbol)
    for symbol in symbols:
        owners: list[str] = []
        declaration = re.compile(rf"\b{re.escape(symbol)}\b")
        active = ""
        for line in source.splitlines():
            marker = re.search(
                r"MMM_ATOMIC_CONCERN_([A-Z0-9_]+)_(?:MEMBERS|INIT)_(START|END)",
                line,
            )
            if marker:
                name = marker.group(1).casefold()
                if marker.group(2) == "START":
                    active = name
                elif active == name:
                    active = ""
                continue
            if active and declaration.search(line) and active not in owners:
                owners.append(active)
        if len(owners) == 1:
            return owners[0]
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


def _compiler_failure_line_numbers(log: str, *, relative: str) -> tuple[int, ...]:
    filename = re.escape(PurePosixPath(relative).name)
    numbers: list[int] = []
    for match in re.finditer(
        rf"(?:^|[\\/]){filename}:(\d+)(?::\d+)?(?::|\s)",
        str(log or ""),
        re.MULTILINE,
    ):
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    return tuple(numbers)


def _compiler_failure_source_excerpt(
    log: str,
    *,
    source: str,
    relative: str,
    concern: str,
    context_lines: int = 2,
    max_chars: int = 5000,
) -> str:
    source_rows = str(source or "").splitlines()
    selected: set[int] = set()
    failing = set(_compiler_failure_line_numbers(log, relative=relative))
    for number in failing:
        if concern and _concern_at_line(source, number) != concern:
            continue
        for current in range(
            max(1, number - context_lines),
            min(len(source_rows), number + context_lines) + 1,
        ):
            selected.add(current)
    if not selected:
        return ""

    rows: list[str] = []
    previous = 0
    for number in sorted(selected):
        if previous and number > previous + 1:
            rows.append("    ...")
        marker = ">>" if number in failing else "  "
        rows.append(f"{marker} {number:5d} | {source_rows[number - 1]}")
        previous = number
    excerpt = "\n".join(rows)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n... failing source excerpt truncated by host ..."
    return excerpt


def _compiler_repair_hints(log: str) -> tuple[str, ...]:
    text = str(log or "")
    lowered = text.casefold()
    hints: list[str] = []

    if "might not have been initialized" in lowered:
        hints.append(
            "DEFINITE_ASSIGNMENT: initialize each reported final field at its declaration "
            "or assign a blank final exactly once on every static-initialization path before any read."
        )
    if "cannot assign a value to static final variable" in lowered:
        hints.append(
            "FINAL_REBINDING: do not assign a new reference/value to the reported final field; "
            "mutate the referenced mutable object when that preserves the contract, otherwise make "
            "the concern-owned binding non-final only when rebinding is semantically required."
        )
    if (
        "incompatible types:" in lowered
        and "object cannot be converted to map" in lowered
    ):
        hints.append(
            "OBJECT_TO_MAP: the producer returns Object. Narrow with instanceof Map<?, ?>, "
            "copy/validate entries into the declared parameterized Map<String, Object>, and return "
            "a type-compatible fallback when the runtime value is not a map; do not use a raw or unchecked cast."
        )
    if (
        "location: package java.util.concurrent" in lowered
        and ("class lock" in lowered or "class reentrantlock" in lowered)
    ):
        hints.append(
            "JDK_LOCK_PACKAGE: use java.util.concurrent.locks.Lock and "
            "java.util.concurrent.locks.ReentrantLock; neither type is in java.util.concurrent."
        )
    if "uses unchecked or unsafe operations" in lowered:
        hints.append(
            "UNCHECKED_TYPES: remove raw collection use and unchecked casts in the selected region; "
            "preserve concrete generic types and validate runtime values before narrowing."
        )
    return tuple(hints)


def _compiler_repair_context(
    log: str,
    *,
    source: str,
    relative: str,
    concern: str,
    max_chars: int = 12000,
) -> str:
    diagnostics = _compact_compiler_failure(
        log,
        source=source,
        relative=relative,
        concern=concern,
        max_chars=5200,
    )
    excerpt = _compiler_failure_source_excerpt(
        log,
        source=source,
        relative=relative,
        concern=concern,
        max_chars=2800,
    )
    parts = [
        "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE "
        "(authoritative; preserve diagnostic text exactly):\n" + diagnostics,
    ]
    if excerpt:
        parts.append(
            "CURRENT COMPILED SOURCE AROUND THE REPORTED LINES "
            "(>> marks a compiler-reported line):\n" + excerpt
        )
    hints = _compiler_repair_hints(log)
    if hints:
        parts.append(
            "COMPILER-DERIVED REPAIR CHECKLIST:\n"
            + "\n".join(f"- {hint}" for hint in hints)
        )
    parts.append(
        "REPAIR CONTRACT:\n"
        "- Fix every compiler error above that belongs to this selected concern.\n"
        "- Use current_selected_region_source as the complete replacement scope.\n"
        "- Preserve already-correct behavior and sibling APIs; do not rewrite unrelated code.\n"
        "- Re-check JDK/package names, declared types, generics, final/mutability, method signatures, and initialization.\n"
        "- Return only the complete corrected selected Java region; the host will compile it again."
    )
    payload = "\n\n".join(parts)
    if len(payload) > max_chars:
        # Diagnostics/excerpts are already independently bounded so the compiler-derived
        # checklist and repair contract stay present. This final guard is defensive only.
        overflow = len(payload) - max_chars
        if overflow > 0 and diagnostics:
            keep = max(1200, len(diagnostics) - overflow - 64)
            diagnostics = diagnostics[:keep].rstrip() + "\n... compiler diagnostics truncated by host ..."
            parts[0] = (
                "ACTUAL COMPILER FAILURE FROM THE JUST-COMPILED CANDIDATE "
                "(authoritative; preserve diagnostic text exactly):\n" + diagnostics
            )
            payload = "\n\n".join(parts)
        if len(payload) > max_chars:
            payload = payload[:max_chars].rstrip() + "\n... repair context truncated by host ..."
    return payload


def _compiler_repair_fingerprint(
    log: str,
    *,
    source: str,
    relative: str,
    concern: str,
) -> str:
    diagnostic = _compiler_diagnostic_fingerprint(log)
    members = _region_content(source, concern=concern, region="MEMBERS")
    initialize = _region_content(source, concern=concern, region="INIT")
    payload = "\n".join((diagnostic, concern, members, initialize))
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
    return slice_concern_requirements(raw, concern=concern)

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
                    "structured_records": deepcopy(
                        outer.get("structured_records") or []
                    ),
                }
                break
    return {
        "task_id": str(task.get("task_id") or ""),
        "concern": name,
        **requirement_payload,
    }



def _behavior_actor_records(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Recover the canonical actor records without asking the coder to invent Java types."""
    authority = _concern_authority(task, concern)
    rows: list[dict[str, str]] = []
    structured = authority.get("structured_records")
    if isinstance(structured, Sequence) and not isinstance(
        structured, (str, bytes, bytearray)
    ):
        for raw in structured:
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name") or "").strip()
            role = str(raw.get("role") or "").strip()
            actor_authority = str(raw.get("authority") or "").strip()
            if name:
                rows.append(
                    {
                        "name": name,
                        "role": role or name,
                        "authority": actor_authority,
                    }
                )
    if rows:
        return tuple(rows)

    # Older/legacy saved designs can arrive with structured_sections={} even
    # though the exact authored source requirement is preserved. Recover only
    # this canonical concern from that exact requirement; never ask the model
    # to infer a missing local Actor type.
    sources = authority.get("source_requirements")
    if not isinstance(sources, Mapping):
        return ()

    def source_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
        match = re.fullmatch(r"R(\d+)", str(item[0] or ""))
        if match:
            return (0, int(match.group(1)))
        return (1, str(item[0] or ""))

    lines: list[str] = []
    for _key, source in sorted(sources.items(), key=source_key):
        lines.extend(
            str(source or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .splitlines()
        )

    def strip_markdown_wrapper(value: str) -> str:
        text = str(value or "").strip()
        wrappers = ("**", "__", "`", "*", "_")
        changed = True
        while changed and text:
            changed = False
            for wrapper in wrappers:
                if (
                    len(text) > len(wrapper) * 2
                    and text.startswith(wrapper)
                    and text.endswith(wrapper)
                ):
                    text = text[len(wrapper):-len(wrapper)].strip()
                    changed = True
                    break
        return text

    def add_row(name: str, role: str = "", actor_authority: str = "") -> None:
        actor_name = strip_markdown_wrapper(name)
        if not actor_name:
            return
        clean_role = strip_markdown_wrapper(role)
        clean_authority = str(actor_authority or "").strip()
        if any(row["name"] == actor_name for row in rows):
            return
        rows.append(
            {
                "name": actor_name,
                "role": clean_role or actor_name,
                "authority": clean_authority,
            }
        )

    def add_inline(raw: str) -> None:
        for item in _split_balanced_commas(raw):
            value = item.strip().rstrip(".;")
            if not value:
                continue
            wrapped = re.fullmatch(
                r"(?P<name>.+?)\s*\((?P<authority>.*)\)",
                value,
            )
            if wrapped is not None:
                add_row(
                    wrapped.group("name"),
                    wrapped.group("name"),
                    wrapped.group("authority"),
                )
                continue
            if ":" in value:
                name, role = value.split(":", 1)
                add_row(name, role)
                continue
            add_row(value, value)

    actor_indent: int | None = None
    for line in lines:
        anchor = re.match(
            r"^(?P<indent>\s*)[-*+]\s*(?P<label>[^:]+?)\s*:\s*(?P<tail>.*?)\s*$",
            line,
        )
        if (
            anchor is not None
            and section_slug(anchor.group("label")) == "actors"
        ):
            actor_indent = len(anchor.group("indent"))
            tail = anchor.group("tail").strip()
            if tail:
                add_inline(tail)
            continue

        if actor_indent is None or not line.strip():
            continue

        child = re.match(
            r"^(?P<indent>\s*)[-*+]\s*(?P<body>.+?)\s*$",
            line,
        )
        if child is not None:
            indent = len(child.group("indent"))
            if indent <= actor_indent:
                break
            body = child.group("body").strip()
            if ":" in body:
                name, role = body.split(":", 1)
                add_row(name, role)
            else:
                add_row(body, body)
            continue

        indentation = len(line) - len(line.lstrip())
        if indentation <= actor_indent:
            break

    return tuple(rows)


def _java_string_literal(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=True)


def _deterministic_behavior_actor_members(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> str:
    rows = _behavior_actor_records(task, concern)
    if not rows:
        raise CustomModuleGenerationError(
            "STRUCTURED_ACTORS_CONTRACT_REQUIRED: behavior_contract.actors must "
            "have canonical structured records or an exact actors source requirement; "
            "free-form Java fallback is disabled."
        )

    values = ",\n        ".join(
        "new Actor("
        + ", ".join(
            (
                _java_string_literal(row["name"]),
                _java_string_literal(row["role"]),
                _java_string_literal(row["authority"]),
            )
        )
        + ")"
        for row in rows
    )
    return (
        "private static final class Actor {\n"
        "    private final String name;\n"
        "    private final String role;\n"
        "    private final String authority;\n\n"
        "    private Actor(String name, String role, String authority) {\n"
        "        this.name = java.util.Objects.requireNonNull(name, \"name\");\n"
        "        this.role = java.util.Objects.requireNonNull(role, \"role\");\n"
        "        this.authority = java.util.Objects.requireNonNull(authority, \"authority\");\n"
        "    }\n"
        "}\n\n"
        "private static final java.util.List<Actor> ACTORS = java.util.List.of(\n"
        f"        {values}\n"
        ");\n\n"
        "private static Actor actorByName(String name) {\n"
        "    for (Actor actor : ACTORS) {\n"
        "        if (actor.name.equals(name)) {\n"
        "            return actor;\n"
        "        }\n"
        "    }\n"
        "    return null;\n"
        "}"
    )


def _flatten_contract_record(record: Mapping[str, Any]) -> str:
    parts: list[str] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, value[key])
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                visit(f"{prefix}[{index}]", item)
            return
        parts.append(f"{prefix}={value}")

    visit("", record)
    return "; ".join(part for part in parts if part)


def _behavior_contract_values(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[str, ...]:
    authority = _concern_authority(task, concern)
    structured = authority.get("structured_records")
    values: list[str] = []
    if isinstance(structured, Sequence) and not isinstance(
        structured, (str, bytes, bytearray)
    ):
        for raw in structured:
            if isinstance(raw, Mapping):
                value = _flatten_contract_record(raw)
                if value:
                    values.append(value)
    if values:
        return tuple(values)

    sources = authority.get("source_requirements")
    if isinstance(sources, Mapping):
        for key in sorted(sources):
            value = " ".join(str(sources[key] or "").split())
            if value and value not in values:
                values.append(value)
    if values:
        return tuple(values)

    raise CustomModuleGenerationError(
        "STRUCTURED_BEHAVIOR_CONTRACT_REQUIRED: "
        f"{_slug(concern.get('concern'))} has neither structured records nor exact "
        "source requirements; free-form Java fallback is disabled."
    )


def _deterministic_behavior_contract_members(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> str:
    name = _slug(concern.get("concern"))
    if name == "actors":
        return _deterministic_behavior_actor_members(task, concern)

    values = _behavior_contract_values(task, concern)
    constant = "CONTRACT_" + _host_java_identifier(name).upper()
    literals = ",\n        ".join(_java_string_literal(value) for value in values)
    return (
        f"private static final java.util.List<String> {constant} = java.util.List.of(\n"
        f"        {literals}\n"
        ");"
    )


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
    encoded: list[str] = []
    for char in text:
        if re.fullmatch(r"[A-Za-z0-9_$]", char):
            encoded.append(char)
        elif ord(char) > 127 and (char.isalpha() or char.isdigit()):
            encoded.append(f"u{ord(char):04X}")
        else:
            encoded.append("_")
    value = "".join(encoded)
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
            return value.rstrip("lL") + ("L" if java_type == "long" else "")
        return ""
    if java_type in {"float", "double"}:
        if re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[fFdD]?",
            value,
        ):
            return value.rstrip("fFdD") + "f" if java_type == "float" else value
        return ""
    return "null" if lowered == "null" else ""


_STATE_BOXED_TYPES = {
    "boolean": "Boolean", "byte": "Byte", "short": "Short", "int": "Integer",
    "long": "Long", "float": "Float", "double": "Double", "char": "Character",
}
_STATE_COLLECTION_TYPES = {
    "list": ("java.util.List", "java.util.ArrayList", 1),
    "collection": ("java.util.Collection", "java.util.ArrayList", 1),
    "set": ("java.util.Set", "java.util.HashSet", 1),
    "map": ("java.util.Map", "java.util.HashMap", 2),
    "queue": ("java.util.Queue", "java.util.ArrayDeque", 1),
    "deque": ("java.util.Deque", "java.util.ArrayDeque", 1),
}
_STATE_REFERENCE_BUILTINS = frozenset({
    "Object", "String", "Boolean", "Byte", "Short", "Integer", "Long",
    "Float", "Double", "Character", "Number", "UUID",
})


def _split_balanced_commas(raw: Any) -> tuple[str, ...]:
    text = str(raw or "")
    rows: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "<": 0, "[": 0, "{": 0}
    closing = {")": "(", ">": "<", "]": "[", "}": "{"}
    quote = ""
    escaped = False
    for char in text:
        if quote:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'", "`"}:
            quote = char
            current.append(char)
            continue
        if char in depths:
            depths[char] += 1
            current.append(char)
            continue
        opener = closing.get(char)
        if opener is not None:
            if depths[opener] > 0:
                depths[opener] -= 1
            current.append(char)
            continue
        if char == "," and not any(depths.values()):
            value = "".join(current).strip()
            if value:
                rows.append(value)
            current = []
            continue
        current.append(char)
    value = "".join(current).strip()
    if value:
        rows.append(value)
    return tuple(rows)


def _normalize_state_java_type(raw_type: str, *, reference: bool = False) -> str:
    source_type = str(raw_type or "").strip()
    if not source_type:
        return ""
    generic = re.fullmatch(
        r"(?P<base>(?:java\.util\.)?[A-Za-z_$][A-Za-z0-9_$.]*)"
        r"\s*<\s*(?P<args>.+)\s*>",
        source_type,
    )
    if generic:
        base = generic.group("base").rsplit(".", 1)[-1].casefold()
        args = _split_balanced_commas(generic.group("args"))
        if base == "enumset":
            if len(args) != 1:
                return ""
            element = _normalize_state_java_type(args[0], reference=True)
            return f"java.util.EnumSet<{element}>" if element else ""
        collection = _STATE_COLLECTION_TYPES.get(base)
        if collection is None or len(args) != collection[2]:
            return ""
        normalized = [_normalize_state_java_type(value, reference=True) for value in args]
        if any(not value for value in normalized):
            return ""
        return f"{collection[0]}<{', '.join(normalized)}>"

    base = source_type.rsplit(".", 1)[-1].casefold()
    primitive = _STATE_JAVA_TYPES.get(base)
    if primitive:
        if reference and primitive in _STATE_BOXED_TYPES:
            return _STATE_BOXED_TYPES[primitive]
        return primitive
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", source_type):
        return source_type
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+", source_type):
        return source_type
    return ""


def _state_java_contract(raw_type: str, raw_default: str) -> tuple[str, str]:
    source_type = str(raw_type or "").strip()
    generic = re.fullmatch(
        r"(?P<base>(?:java\.util\.)?[A-Za-z_$][A-Za-z0-9_$.]*)"
        r"\s*<\s*(?P<args>.+)\s*>",
        source_type,
    )
    base_type = generic.group("base") if generic else source_type
    base = base_type.rsplit(".", 1)[-1].casefold()
    default = str(raw_default or "").strip().casefold()
    empty = default in {"[]", "{}", "empty", "empty_list", "empty_set", "empty_map"}

    if base == "enumset":
        if generic:
            java_type = _normalize_state_java_type(source_type)
            if not java_type:
                return "", ""
            element_type = java_type[java_type.find("<") + 1:-1].strip()
            if (
                not element_type or "," in element_type or "?" in element_type
                or re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$.]*", element_type) is None
            ):
                return "", ""
            if default == "null":
                return java_type, "null"
            if empty or not default:
                return java_type, f"java.util.EnumSet.noneOf({element_type}.class)"
            return java_type, ""
        java_type = "java.util.Set<java.lang.Enum<?>>"
        if default == "null":
            return java_type, "null"
        return ((java_type, "new java.util.HashSet<>()") if empty or not default else (java_type, ""))

    collection = _STATE_COLLECTION_TYPES.get(base)
    if collection is not None:
        if generic:
            java_type = _normalize_state_java_type(source_type)
        else:
            fallback = "Object, Object" if collection[2] == 2 else "Object"
            java_type = f"{collection[0]}<{fallback}>"
        if not java_type:
            return "", ""
        if default == "null":
            return java_type, "null"
        if empty or not default:
            return java_type, f"new {collection[1]}<>()"
        return java_type, ""

    java_type = _normalize_state_java_type(source_type)
    if not java_type:
        return "", ""
    primitive = _STATE_JAVA_TYPES.get(base)
    if primitive:
        return java_type, _state_default_literal(java_type, raw_default)
    return java_type, "null" if default == "null" else ""


def _state_support_types(raw_type: str) -> tuple[tuple[str, str], ...]:
    source_type = str(raw_type or "").strip()
    if not source_type:
        return ()
    generic = re.fullmatch(
        r"(?P<base>(?:java\.util\.)?[A-Za-z_$][A-Za-z0-9_$.]*)"
        r"\s*<\s*(?P<args>.+)\s*>",
        source_type,
    )
    if generic:
        base = generic.group("base").rsplit(".", 1)[-1].casefold()
        rows: list[tuple[str, str]] = []
        for value in _split_balanced_commas(generic.group("args")):
            normalized = _normalize_state_java_type(value, reference=True)
            if not normalized:
                continue
            if base == "enumset":
                if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", normalized) and normalized not in _STATE_REFERENCE_BUILTINS:
                    rows.append(("enum", normalized))
                continue
            rows.extend(_state_support_types(value))
        return tuple(rows)

    base = source_type.rsplit(".", 1)[-1].casefold()
    if base in _STATE_JAVA_TYPES or source_type in _STATE_REFERENCE_BUILTINS:
        return ()
    if "." in source_type:
        return ()
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", source_type):
        return (("class", source_type),)
    return ()


def _structured_requirement_records(
    source_requirements: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Parse Markdown records by the host-declared record_schema field order."""
    ordered = [
        (str(key), str(value))
        for key, value in sorted(
            dict(source_requirements or {}).items(), key=_requirement_sort_key
        )
    ]
    target = _slug(concern.get("concern"))
    anchor = next(
        (
            index
            for index, (_key, value) in enumerate(ordered)
            if _requirement_concern_label(value) == target
        ),
        -1,
    )
    if anchor < 0:
        return ()

    schema = concern.get("record_schema")
    properties = (
        list(schema.get("properties") or {})
        if isinstance(schema, Mapping)
        else []
    )
    required = (
        list(schema.get("required") or [])
        if isinstance(schema, Mapping)
        else []
    )
    header_fields = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        ordered[anchor][1].split(":", 1)[1] if ":" in ordered[anchor][1] else "",
    )
    fields = [field for field in header_fields if not properties or field in properties]
    if not fields:
        fields = properties
    if not fields or fields[0] != "name":
        return ()

    records: list[dict[str, str]] = []
    for _key, value in ordered[anchor + 1:]:
        if value.startswith("## "):
            break
        sibling = _requirement_concern_label(value)
        if sibling:
            break
        stripped = value.lstrip()
        if not stripped.startswith("- "):
            continue
        head, separator, tail = stripped[2:].partition(":")
        if not separator:
            continue
        name = head.strip().strip("*`_ ")
        values = re.findall(r"`([^`]*)`", tail)
        if not name or len(values) < len(fields) - 1:
            continue
        record = {"name": name}
        record.update({
            field: raw.strip()
            for field, raw in zip(fields[1:], values)
        })
        if required and any(not str(record.get(field) or "").strip() for field in required):
            continue
        records.append(record)
    return tuple(records)


def _inline_state_variable_records(
    source_requirements: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Parse compact 'name (Type, key=value)' state declarations."""
    target = _slug(concern.get("concern"))
    ordered = [
        (str(key), str(value))
        for key, value in sorted(dict(source_requirements or {}).items(), key=_requirement_sort_key)
    ]
    anchor = next((value for _key, value in ordered if _requirement_concern_label(value) == target), "")
    if not anchor or ":" not in anchor:
        return ()
    payload = anchor.split(":", 1)[1].strip()
    if not payload:
        return ()
    if re.match(r"name\s*\(", payload, re.IGNORECASE):
        # name(...), type(...), default(...) is the labeled record syntax,
        # not a compact variable literally named 'name' with a custom type.
        return ()

    records: list[dict[str, str]] = []
    for entry in _split_balanced_commas(payload):
        match = re.fullmatch(r"\s*(?P<name>.+?)\s*\((?P<body>.*)\)\s*", entry)
        if match is None:
            return ()
        parts = _split_balanced_commas(match.group("body"))
        if not parts or "=" in parts[0]:
            return ()
        record = {
            "name": match.group("name").strip().strip("*`_ "),
            "type": parts[0].strip(),
        }
        for attribute in parts[1:]:
            key, separator, value = attribute.partition("=")
            if not separator:
                continue
            normalized_key = key.strip().casefold()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized_key):
                record[normalized_key] = value.strip().strip("`")
        if not record["name"] or not record["type"]:
            return ()
        records.append(record)
    return tuple(records)


def _state_variable_contract(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    authority = _concern_authority(task, concern)
    source_requirements = authority.get("source_requirements")
    if not isinstance(source_requirements, Mapping):
        return ()

    structured = authority.get("structured_records")
    records = (
        tuple(dict(item) for item in structured if isinstance(item, Mapping))
        if isinstance(structured, list) and structured
        else _structured_requirement_records(source_requirements, concern)
        or _inline_state_variable_records(source_requirements, concern)
    )
    if records:
        contracts: list[dict[str, str]] = []
        seen_names: set[str] = set()
        for attributes in records:
            name = _host_java_identifier(attributes.get("name"))
            java_type, default_literal = _state_java_contract(
                attributes.get("type", ""),
                attributes.get("default", ""),
            )
            # Never partially lower a variables record set. Unknown types fall
            # back to the structured coder rather than silently dropping state.
            if not name or not java_type or name in seen_names:
                return ()
            seen_names.add(name)
            contracts.append({
                "name": name,
                "java_type": java_type,
                "default_literal": default_literal,
                "owner": attributes.get("owner", ""),
                "unit": attributes.get("unit", ""),
                "domain": attributes.get("domain", ""),
                "source_type": attributes.get("type", ""),
            })
        return tuple(contracts)

    # Backward-compatible support for the older name(...)/type(...) form.
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
        name = _host_java_identifier(attributes["name"])
        java_type, default_literal = _state_java_contract(
            attributes["type"],
            attributes.get("default", ""),
        )
        if not java_type or not name or name in seen_names:
            continue
        seen_names.add(name)
        contracts.append({
            "name": name,
            "java_type": java_type,
            "default_literal": default_literal,
            "owner": attributes.get("owner", ""),
            "unit": attributes.get("unit", ""),
            "domain": attributes.get("domain", ""),
            "source_type": attributes.get("type", ""),
        })
    return tuple(contracts)


def _deterministic_state_variable_members(
    task: Mapping[str, Any],
    concern: Mapping[str, Any],
) -> str:
    contracts = _state_variable_contract(task, concern)
    if not contracts:
        return ""

    support_kinds: dict[str, str] = {}
    support_order: list[str] = []
    for item in contracts:
        for kind, name in _state_support_types(item.get("source_type", "")):
            previous = support_kinds.get(name)
            if previous is not None and previous != kind:
                return ""
            if previous is None:
                support_kinds[name] = kind
                support_order.append(name)

    rows: list[str] = []
    for name in support_order:
        rows.append(
            f"private enum {name} {{}}"
            if support_kinds[name] == "enum"
            else f"private static final class {name} {{}}"
        )
    for item in contracts:
        initializer = f" = {item['default_literal']}" if item["default_literal"] else ""
        rows.append(f"private static {item['java_type']} {item['name']}{initializer};")
    return "\n".join(rows)


_PLATFORM_FQN = re.compile(
    r"\b(?:net\.minecraft|net\.fabricmc)(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+"
)


def _section_platform_api_policy(section: str) -> str:
    spec = section_spec(str(section or "").strip()) or {}
    policy = str(spec.get("platform_api_policy") or "").strip()
    return policy if policy in {"forbidden", "host_grounded_only"} else "forbidden"


def _bounded_grounding(
    grounding: Mapping[str, Any],
    *,
    section: str = "",
) -> dict[str, Any]:
    direct = grounding.get("direct_host_context")
    direct_payload = dict(direct) if isinstance(direct, Mapping) else {}
    host_version_facts = dict(direct_payload.get("host_version_facts") or {})
    if _section_platform_api_policy(section) == "forbidden":
        # Domain concerns do not own Fabric/Minecraft registration or lifecycle.
        # Hiding unrelated API symbols prevents a coder from "helpfully" adding
        # registry/Identifier plumbing that the authored concern never requested.
        host_version_facts.pop("api_symbols", None)
    implementation_ir = direct_payload.get("implementation_ir_node")
    implementation_contract = {}
    if isinstance(implementation_ir, Mapping):
        implementation_contract = {
            "symbol": str(implementation_ir.get("symbol") or ""),
            "public_api": [
                str(value)
                for value in implementation_ir.get("public_api") or ()
                if str(value).strip()
            ],
        }
    return {
        "schema_version": grounding.get("schema_version"),
        "artifact_kind": grounding.get("artifact_kind"),
        "facts": (
            grounding.get("facts") or []
            if _section_platform_api_policy(section) != "forbidden"
            else []
        ),
        "policy": dict(grounding.get("policy") or {}),
        "platform": dict(direct_payload.get("platform") or {}),
        "host_version_facts": (
            host_version_facts
            if _section_platform_api_policy(section) != "forbidden"
            else {}
        ),
        "implementation_contract": implementation_contract,
    }


def _approved_platform_owners(grounding: Mapping[str, Any]) -> tuple[str, ...]:
    owners: set[str] = set()

    def add_owner(value: Any) -> None:
        owner = str(value or "").strip().replace("/", ".")
        if owner.startswith(("net.minecraft.", "net.fabricmc.")):
            owners.add(owner.split("$", 1)[0])

    for fact in grounding.get("facts") or ():
        if not isinstance(fact, Mapping):
            continue
        for owner in fact.get("required_imports") or ():
            add_owner(owner)
        symbols = fact.get("api_symbols")
        if isinstance(symbols, Mapping):
            for raw in symbols.values():
                if isinstance(raw, Mapping):
                    add_owner(raw.get("owner"))

    direct = grounding.get("direct_host_context")
    if isinstance(direct, Mapping):
        host = direct.get("host_version_facts")
        if isinstance(host, Mapping):
            symbols = host.get("api_symbols")
            if isinstance(symbols, Mapping):
                for raw in symbols.values():
                    if isinstance(raw, Mapping):
                        add_owner(raw.get("owner"))
    return tuple(sorted(owners))


def _platform_owner_references(source: str) -> tuple[str, ...]:
    refs: set[str] = set()
    for match in _PLATFORM_FQN.finditer(_structure_scan(source)):
        token = match.group(0)
        parts = token.split(".")
        owner_end = -1
        for index, part in enumerate(parts):
            if part and (part[0].isupper() or "$" in part):
                owner_end = index
        if owner_end >= 0:
            refs.add(".".join(parts[: owner_end + 1]).split("$", 1)[0])
    return tuple(sorted(refs))


def _validate_platform_api_admission(
    source: str,
    *,
    section: str,
    grounding: Mapping[str, Any],
) -> None:
    refs = _platform_owner_references(source)
    if not refs:
        return

    normalized_section = str(section or "").strip()
    policy = _section_platform_api_policy(normalized_section)
    if policy == "forbidden":
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_PLATFORM_API_FORBIDDEN: "
            f"{normalized_section or '<unknown>'} is host-classified platform-neutral "
            f"and referenced platform owners {list(refs)!r}."
        )

    approved = _approved_platform_owners(grounding)
    unauthorized = tuple(
        owner
        for owner in refs
        if not any(
            owner == allowed
            or owner.startswith(allowed + ".")
            or allowed.startswith(owner + ".")
            for allowed in approved
        )
    )
    if unauthorized:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_UNGROUNDED_PLATFORM_API: "
            f"{normalized_section} referenced unapproved target owners "
            f"{list(unauthorized)!r}; approved owners={list(approved)!r}."
        )


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


def _compact_prompt_member_contract(
    raw: Mapping[str, Any],
    *,
    owner_concern: str = "",
) -> dict[str, Any]:
    """Project parsed Java API facts to the minimum model-visible contract."""

    result: dict[str, Any] = {}
    if owner_concern:
        result["owner_concern"] = owner_concern
    for key in (
        "kind",
        "symbol",
        "declared_type",
        "return_type",
        "static",
        "mutable",
        "typed_api_source",
    ):
        value = raw.get(key)
        if value not in (None, "", (), []):
            result[key] = value
    parameters = raw.get("parameters")
    if isinstance(parameters, Sequence) and not isinstance(
        parameters, (str, bytes, bytearray)
    ):
        result["parameters"] = [
            {
                key: item.get(key)
                for key in ("name", "type", "varargs")
                if item.get(key) not in (None, "")
            }
            for item in parameters
            if isinstance(item, Mapping)
        ]
    return result


def _dependency_api_context(raw: str, *, max_chars: int = 10000) -> list[dict[str, Any]]:
    """Expose compact exact dependency declarations instead of full source-shaped noise."""

    result: list[dict[str, Any]] = []
    used = 0
    for row in _dependency_context_rows(raw):
        source = str(row.get("source") or "")
        typed_api: list[dict[str, Any]] = []
        if source.strip():
            try:
                typed_api = [
                    _compact_prompt_member_contract(item)
                    for item in public_source_member_contracts(source)
                ]
            except JavaRegionParseError:
                typed_api = []
        compact = {
            "symbol": str(row.get("symbol") or ""),
            "path": str(row.get("path") or ""),
            "responsibility": str(row.get("responsibility") or ""),
            # Parsed source is authoritative when available. Do not duplicate every
            # declaration as both prose public_api and typed_public_api.
            "public_api": [] if typed_api else list(row.get("public_api") or []),
            "typed_public_api": typed_api,
            "typed_api_source": "tree_sitter_java" if typed_api else "unavailable",
        }
        encoded = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        if used + len(encoded) > max_chars:
            fallback = {
                "symbol": compact["symbol"],
                "path": compact["path"],
                "responsibility": compact["responsibility"],
                "public_api": list(row.get("public_api") or []),
                "typed_public_api": [],
                "typed_api_source": "budget_fallback",
            }
            encoded = json.dumps(fallback, ensure_ascii=False, sort_keys=True)
            if used + len(encoded) > max_chars:
                break
            compact = fallback
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


def _typed_dependency_method_contracts(
    raw: str,
) -> dict[str, dict[str, tuple[dict[str, Any], ...]]]:
    """Return Tree-sitter-derived dependency method contracts keyed by owner/name."""

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in _dependency_context_rows(raw):
        owner = str(row.get("symbol") or "").strip()
        source = str(row.get("source") or "").strip()
        if not owner or not source:
            continue
        try:
            contracts = public_source_member_contracts(source)
        except JavaRegionParseError:
            continue
        by_name = result.setdefault(owner, {})
        for contract in contracts:
            if contract.get("kind") != "method":
                continue
            name = str(contract.get("symbol") or "").strip()
            if not name:
                continue
            by_name.setdefault(name, []).append(dict(contract))
    return {
        owner: {
            name: tuple(entries)
            for name, entries in methods.items()
        }
        for owner, methods in result.items()
    }


def _dependency_parameter_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(item.get("name") or "").strip()
        for item in contract.get("parameters") or ()
        if isinstance(item, Mapping)
    )


def _dependency_semantic_role(contract: Mapping[str, Any]) -> str:
    """Infer only high-confidence API roles from authoritative parameter names."""
    names = tuple(name.casefold() for name in _dependency_parameter_names(contract))
    if names == ("name", "value", "context"):
        return "direct_state_write"
    if names == ("name", "context"):
        return "contextual_state_read"
    if names == ("name", "value"):
        return "direct_state_write"
    if names == ("name",):
        return "state_read"
    if names == ("trigger", "context"):
        return "trigger_dispatch"
    if names == ("event", "context"):
        return "event_dispatch"
    if names == ("fromstate", "trigger", "context"):
        return "transition_dispatch"
    return ""


def _dependency_usage_rule(role: str) -> str:
    return {
        "direct_state_write": (
            "Direct key/value state assignment. Use this role when authored logic sets "
            "a named state value; the value is not a trigger."
        ),
        "contextual_state_read": "Read a named state value using the supplied context overlay.",
        "state_read": "Read a named persistent state value.",
        "trigger_dispatch": (
            "Dispatch registered actions by trigger. The context argument is execution "
            "context; this is not a key/value setter and never accepts a state value."
        ),
        "event_dispatch": (
            "Dispatch registered cleanup/event actions by event. This is not a direct "
            "key/value state mutation."
        ),
        "transition_dispatch": (
            "Evaluate registered transition rules from a concrete fromState under a trigger. "
            "This does not assign an arbitrary state key/value."
        ),
    }.get(role, "")


def _dependency_call_contracts(raw: str) -> list[dict[str, Any]]:
    """Flatten exact dependency call signatures plus semantic parameter roles."""

    rows: list[dict[str, Any]] = []
    for owner, methods in sorted(_typed_dependency_method_contracts(raw).items()):
        for name, contracts in sorted(methods.items()):
            for contract in contracts:
                parameters = [
                    str(item.get("type") or "").strip()
                    for item in contract.get("parameters") or ()
                    if isinstance(item, Mapping)
                ]
                parameter_names = list(_dependency_parameter_names(contract))
                semantic_role = _dependency_semantic_role(contract)
                row = {
                    "owner": owner,
                    "method": name,
                    "arity": len(parameters),
                    "parameter_types": parameters,
                    "parameter_names": parameter_names,
                    "return_type": str(contract.get("return_type") or "").strip(),
                    "static": bool(contract.get("static") is True),
                    "call_shape": f"{owner}.{name}(" + ", ".join(parameters) + ")",
                    "invocation_shape": (
                        f"{owner}.{name}(" + ", ".join(parameter_names) + ")"
                    ),
                }
                usage_rule = _dependency_usage_rule(semantic_role)
                if semantic_role:
                    row["semantic_role"] = semantic_role
                if usage_rule:
                    row["usage_rule"] = usage_rule
                rows.append(row)
    return rows


def _canonicalize_dependency_call_semantics(
    value: str,
    *,
    dependency_source: str,
) -> tuple[str, tuple[str, ...]]:
    """Repair only mechanically determined dispatcher-as-setter dependency misuse.

    A small coder can confuse a two-argument trigger dispatcher with a three-argument
    key/value setter. The host rewrites only when authoritative parameter roles prove
    that mismatch, the final argument is the exact context variable, and the same
    owner exposes exactly one static direct-state-write signature with arity three.
    Every ambiguous case remains validator-owned and is never guessed.
    """

    source = str(value or "").strip()
    if not source or not dependency_source:
        return source, ()
    dependencies = _typed_dependency_method_contracts(dependency_source)
    if not dependencies:
        return source, ()

    edits: list[tuple[int, int, str, str]] = []
    for call in class_body_method_invocation_details(source):
        receiver = str(call.get("receiver") or "").strip()
        symbol = str(call.get("symbol") or "").strip()
        arguments = tuple(str(arg or "").strip() for arg in call.get("arguments") or ())
        owner_methods = dependencies.get(receiver)
        if not owner_methods or not symbol:
            continue

        current_contracts = tuple(owner_methods.get(symbol) or ())
        if not current_contracts:
            continue
        arity = len(arguments)
        if any(
            item.get("static") is True
            and len(item.get("parameters") or ()) == arity
            for item in current_contracts
        ):
            continue

        current_roles = {
            _dependency_semantic_role(item)
            for item in current_contracts
            if item.get("static") is True
        }
        if not current_roles.intersection({"trigger_dispatch", "event_dispatch"}):
            continue
        if arity != 3 or not arguments or arguments[-1] != "context":
            continue

        replacements: set[str] = set()
        for candidate_name, candidate_contracts in owner_methods.items():
            for contract in candidate_contracts:
                if (
                    contract.get("static") is True
                    and len(contract.get("parameters") or ()) == arity
                    and _dependency_semantic_role(contract) == "direct_state_write"
                ):
                    replacements.add(candidate_name)
        if len(replacements) != 1:
            continue
        replacement = next(iter(replacements))
        if replacement == symbol:
            continue
        edits.append(
            (
                int(call["name_start_byte"]),
                int(call["name_end_byte"]),
                replacement,
                f"{receiver}.{symbol}/{arity}->{receiver}.{replacement}/{arity}",
            )
        )

    if not edits:
        return source, ()

    encoded = source.encode("utf-8")
    for start, end, replacement, _description in sorted(edits, reverse=True):
        encoded = encoded[:start] + replacement.encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8"), tuple(item[3] for item in edits)


def _dependency_repair_contract(raw: str, diagnostic: str) -> dict[str, Any] | None:
    """Project one dependency diagnostic to the exact owner-local callable surface."""
    match = re.search(
        r"dependency API\s+([A-Za-z_$][A-Za-z0-9_$.]*)\."
        r"([A-Za-z_$][A-Za-z0-9_$]*)\s+called with\s+(\d+)\s+argument",
        str(diagnostic or ""),
    )
    if match is None:
        return None
    owner, method, arity_text = match.groups()
    owner_rows = [
        row for row in _dependency_call_contracts(raw)
        if row.get("owner") == owner
    ]
    if not owner_rows:
        return None
    return {
        "owner": owner,
        "rejected_method": method,
        "rejected_arity": int(arity_text),
        "authoritative_calls": owner_rows,
        "rules": (
            "Use only one authoritative call listed here. parameter_names are semantic "
            "roles, not decoration. A trigger_dispatch/event_dispatch call consumes a "
            "trigger/event plus context and must never be used as a key/value setter. "
            "For a direct state assignment choose a direct_state_write signature. "
            "Do not add, remove, reorder, or invent dependency arguments outside an "
            "authoritative invocation_shape."
        ),
    }


def _simple_object_type(value: Any) -> bool:
    normalized = re.sub(r"\s+", "", str(value or ""))
    return normalized in {"Object", "java.lang.Object"}


_JAVA_PRIMITIVE_TYPES = frozenset({
    "boolean", "byte", "short", "int", "long", "char", "float", "double", "void",
})


def _type_leaf_names(value: Any) -> tuple[str, ...]:
    """Return raw leaf type names from one Java type expression.

    Fully-qualified names are returned intact so the caller can defer their
    existence/binding to JDT/javac instead of guessing from spelling.
    """

    text = str(value or "").strip()
    if not text:
        return ()
    text = re.sub(r"\s*\.\.\.\s*$", "", text)
    while text.endswith("[]"):
        text = text[:-2].strip()
    for prefix in ("? extends ", "? super "):
        if text.startswith(prefix):
            return _type_leaf_names(text[len(prefix):])
    if text == "?":
        return ()

    raw, args = _generic_type_parts(text)
    normalized_raw = re.sub(r"\s+", "", raw)
    leaves: list[str] = []
    if normalized_raw:
        leaves.append(normalized_raw)
    for arg in args:
        leaves.extend(_type_leaf_names(arg))
    return tuple(dict.fromkeys(leaves))


def _dependency_authorized_simple_types(raw: str) -> set[str]:
    names = set(_dependency_declared_identifiers(raw))
    for row in _dependency_context_rows(raw):
        source = str(row.get("source") or "").strip()
        if not source:
            continue
        try:
            contracts = public_source_member_contracts(source)
        except JavaRegionParseError:
            continue
        for contract in contracts:
            if contract.get("kind") == "type":
                symbol = str(contract.get("symbol") or "").strip()
                if symbol:
                    names.add(symbol)
            type_values: list[Any] = []
            if contract.get("kind") == "field":
                type_values.append(contract.get("declared_type"))
            elif contract.get("kind") == "method":
                type_values.append(contract.get("return_type"))
                type_values.extend(
                    parameter.get("type")
                    for parameter in contract.get("parameters") or ()
                    if isinstance(parameter, Mapping)
                )
            for type_value in type_values:
                for leaf in _type_leaf_names(type_value):
                    if "." not in leaf and leaf not in _JAVA_PRIMITIVE_TYPES:
                        names.add(leaf)
    return names


def _validate_declared_type_authority(
    contracts: Sequence[Mapping[str, Any]],
    *,
    dependency_source: str,
    sibling_api: Sequence[Mapping[str, Any]],
) -> None:
    """Reject ungrounded simple declaration types before javac.

    FQCNs are deliberately deferred to JDT/javac because package/classpath
    binding is semantic. Simple names, however, must come from java.lang/JDK,
    the current candidate, an accepted sibling type, or dependency authority.
    """

    allowed = (
        set(_JAVA_LANG_SIMPLE_TYPES)
        | set(_JDK_CANONICAL_SIMPLE_TYPES)
        | _dependency_authorized_simple_types(dependency_source)
    )
    allowed.update(
        str(item.get("symbol") or "").strip()
        for item in contracts
        if item.get("kind") == "type" and str(item.get("symbol") or "").strip()
    )
    allowed.update(
        str(item.get("symbol") or "").strip()
        for item in sibling_api
        if isinstance(item, Mapping)
        and item.get("kind") == "type"
        and str(item.get("symbol") or "").strip()
    )

    unknown: set[str] = set()
    for contract in contracts:
        values: list[Any] = []
        if contract.get("kind") == "field":
            values.append(contract.get("declared_type"))
        elif contract.get("kind") == "method":
            values.append(contract.get("return_type"))
            values.extend(
                parameter.get("type")
                for parameter in contract.get("parameters") or ()
                if isinstance(parameter, Mapping)
            )
        elif contract.get("kind") == "constructor":
            values.extend(
                parameter.get("type")
                for parameter in contract.get("parameters") or ()
                if isinstance(parameter, Mapping)
            )
        for value in values:
            for leaf in _type_leaf_names(value):
                if (
                    not leaf
                    or leaf in _JAVA_PRIMITIVE_TYPES
                    or leaf in allowed
                    or (len(leaf) == 1 and leaf.isupper())
                ):
                    continue
                if "." in leaf:
                    if (
                        leaf.startswith(("java.", "javax."))
                        and leaf not in _KNOWN_CANONICAL_JDK_TYPES
                    ):
                        try:
                            from .jdk_type_index import is_public_jdk_type

                            if not is_public_jdk_type(leaf):
                                unknown.add(leaf)
                        except (OSError, RuntimeError, ValueError):
                            pass
                    continue
                unknown.add(leaf)

    if unknown:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: ungrounded simple Java type name(s): "
            + ", ".join(sorted(unknown))
            + ". Use an authoritative sibling/dependency type, a known JDK type, "
            "or the exact fully-qualified external type."
        )


def _jdk_constructor_accepts_arity(
    shapes: Sequence[Mapping[str, Any]],
    argument_count: int,
) -> bool:
    for shape in shapes:
        try:
            arity = int(shape.get("arity", -1))
        except (TypeError, ValueError):
            continue
        if arity < 0:
            continue
        if shape.get("varargs") is True:
            if argument_count >= max(0, arity - 1):
                return True
        elif argument_count == arity:
            return True
    return False


def _validate_jdk_method_invocations(value: str) -> None:
    """Reject provably nonexistent installed-JDK methods before Gradle compile.

    Exact JDK class receivers and enum-constant receivers are authoritative enough
    to validate without guessing local expression types.
    """

    try:
        from .jdk_type_index import (
            is_public_jdk_type,
            public_jdk_method_shapes,
        )
    except ImportError:
        return

    for call in class_body_method_invocations(value):
        receiver = str(call.get("receiver") or "").strip()
        method = str(call.get("symbol") or "").strip()
        if not receiver or not method:
            continue

        owner = ""
        require_static: bool | None = None
        candidate = _canonical_jdk_class_name(receiver)
        try:
            if candidate.startswith(("java.", "javax.")) and is_public_jdk_type(candidate):
                owner = candidate
                require_static = True
            else:
                parts = receiver.split(".")
                for cut in range(len(parts) - 1, 1, -1):
                    prefix = ".".join(parts[:cut])
                    suffix = parts[cut:]
                    if (
                        len(suffix) == 1
                        and re.fullmatch(r"[A-Z][A-Z0-9_]*", suffix[0])
                        and is_public_jdk_type(prefix)
                    ):
                        owner = prefix
                        require_static = False
                        break
        except (OSError, RuntimeError, ValueError):
            continue

        if not owner:
            continue

        try:
            shapes = public_jdk_method_shapes(owner, method)
        except (OSError, RuntimeError, ValueError):
            continue
        if not shapes:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: installed-JDK type "
                f"{owner} has no public method {method}(...)."
            )

        argument_count = int(call.get("argument_count") or 0)
        matching = [
            shape
            for shape in shapes
            if _jdk_constructor_accepts_arity((shape,), argument_count)
            and (
                require_static is not True
                or shape.get("static") is True
            )
        ]
        if matching:
            continue

        allowed = sorted(
            {
                (
                    f"{int(shape.get('arity', 0)) - 1}+"
                    if shape.get("varargs") is True
                    else str(int(shape.get("arity", 0)))
                )
                for shape in shapes
                if require_static is not True or shape.get("static") is True
            }
        )
        qualifier = " static" if require_static is True else ""
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: installed-JDK"
            f"{qualifier} method {owner}.{method}(...) called with "
            f"{argument_count} argument(s); authoritative public arity is {allowed}."
        )


def _validate_jdk_object_creations(value: str) -> None:
    try:
        from .jdk_type_index import (
            is_public_jdk_type,
            public_jdk_constructor_shapes,
        )
    except ImportError:
        return

    for creation in class_body_object_creations(value):
        raw_type = str(creation.get("type") or "").strip()
        fqcn = _canonical_jdk_class_name(raw_type)
        if not fqcn.startswith(("java.", "javax.")):
            continue
        try:
            if not is_public_jdk_type(fqcn):
                continue
            shapes = public_jdk_constructor_shapes(fqcn)
        except (OSError, RuntimeError, ValueError):
            continue
        argument_count = int(creation.get("argument_count") or 0)
        if shapes and _jdk_constructor_accepts_arity(shapes, argument_count):
            continue
        if not shapes:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: installed-JDK type "
                f"{fqcn} has no public constructor available for direct instantiation."
            )
        allowed = sorted(
            {
                (
                    f"{int(shape.get('arity', 0)) - 1}+"
                    if shape.get("varargs") is True
                    else str(int(shape.get("arity", 0)))
                )
                for shape in shapes
            }
        )
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: installed-JDK constructor "
            f"{fqcn} called with {argument_count} argument(s); authoritative "
            f"public constructor arity is {allowed}."
        )


def _strip_host_orchestrated_dependency_lifecycle_calls(
    value: str,
    *,
    dependency_source: str,
) -> tuple[str, tuple[str, ...]]:
    """Remove standalone dependency lifecycle calls already owned by the host graph.

    Only statement-shaped calls to exact dependency owners with an authoritative
    static initialize()/onInitialize() contract are removed. Other dependency calls
    are never rewritten.
    """

    dependencies = _typed_dependency_method_contracts(dependency_source)
    owners: set[str] = set()
    for owner, methods in dependencies.items():
        for lifecycle in ("initialize", "onInitialize"):
            contracts = methods.get(lifecycle) or ()
            if any(item.get("static") is True for item in contracts):
                owners.add(owner)
                break
    if not owners:
        return value, ()

    source = str(value or "")
    changes: list[str] = []
    for owner in sorted(owners, key=len, reverse=True):
        # Generated atomic regions are import-free, so dependency owners are simple
        # host-provided symbols. Restrict canonicalization to one standalone Java
        # expression statement; embedded calls remain validator-owned failures.
        pattern = re.compile(
            rf"(?m)^(?P<indent>[ \t]*){re.escape(owner)}\s*\.\s*"
            rf"(?P<method>initialize|onInitialize)\s*\([^;\n]*\)\s*;[ \t]*$"
        )

        def replace(match: re.Match[str]) -> str:
            changes.append(f"{owner}.{match.group('method')}")
            return match.group("indent") + "// host-orchestrated dependency lifecycle"

        source = pattern.sub(replace, source)

    return source, tuple(changes)


def _validate_first_pass_java_semantics(
    value: str,
    *,
    dependency_source: str,
    sibling_api: Sequence[Mapping[str, Any]],
) -> None:
    """Reject deterministic Java mistakes before the first Gradle compile.

    Syntax/declaration facts come from tree-sitter-java. Project/classpath-complete
    binding remains owned by javac/JDT/Gradle; this gate intentionally checks only
    facts that are unambiguous from the candidate and host-supplied dependency API.
    """

    try:
        contracts = class_body_member_contracts(value)
    except JavaRegionParseError as exc:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: admitted member source became "
            f"syntactically invalid before semantic validation: {exc}"
        ) from exc
    _validate_declared_type_authority(
        contracts,
        dependency_source=dependency_source,
        sibling_api=sibling_api,
    )
    _validate_jdk_object_creations(value)
    _validate_jdk_method_invocations(value)
    violations: list[str] = []

    def declared_final(item: Mapping[str, Any]) -> bool:
        if item.get("final") is True:
            return True
        declaration = str(item.get("declaration") or "")
        return bool(re.search(r"\\bfinal\\b", declaration))

    final_fields = {
        str(item.get("symbol") or "")
        for item in contracts
        if item.get("kind") == "field"
        and declared_final(item)
        and str(item.get("symbol") or "")
    }
    blank_finals = sorted(
        str(item.get("symbol") or "")
        for item in contracts
        if item.get("kind") == "field"
        and declared_final(item)
        and item.get("initialized") is not True
        and str(item.get("symbol") or "")
    )
    if blank_finals:
        violations.append(
            "blank final field(s) have no legal "
            "concern-owned initialization path: " + ", ".join(blank_finals)
        )

    final_fields.update(
        str(row.get("symbol") or "")
        for row in sibling_api
        if isinstance(row, Mapping)
        and row.get("kind") == "field"
        and row.get("mutable") is False
        and str(row.get("symbol") or "")
    )
    assigned = set(class_body_assignment_targets(value))
    rebound = sorted(final_fields & assigned)
    if rebound:
        violations.append(
            "generated executable code reassigns "
            "final field(s): " + ", ".join(rebound)
        )

    dependencies = _typed_dependency_method_contracts(dependency_source)

    if dependencies:
        for call in class_body_method_invocations(value):
            receiver = str(call.get("receiver") or "").strip()
            name = str(call.get("symbol") or "").strip()
            arity = int(call.get("argument_count") or 0)
            owner_methods = dependencies.get(receiver)
            if owner_methods is None:
                continue
            candidates = owner_methods.get(name)
            if not candidates:
                violations.append(
                    "dependency API "
                    f"{receiver}.{name}(...) does not exist in the Tree-sitter-derived "
                    "authoritative dependency source."
                )
                continue
            matching = [
                item
                for item in candidates
                if len(item.get("parameters") or ()) == arity
            ]
            if not matching:
                expected = sorted(
                    {len(item.get("parameters") or ()) for item in candidates}
                )
                violations.append(
                    "dependency API "
                    f"{receiver}.{name} called with {arity} argument(s); authoritative "
                    f"arity is {expected}."
                )
                continue
            if not any(item.get("static") is True for item in matching):
                violations.append(
                    "dependency API "
                    f"{receiver}.{name}(...) is instance-owned, not a static class call."
                )

    # Local return-type validation is independent of dependency APIs. Never skip it
    # merely because this concern has no external dependency source.
    local_methods: dict[str, list[dict[str, Any]]] = {}
    for item in contracts:
        if item.get("kind") != "method":
            continue
        symbol = str(item.get("symbol") or "").strip()
        if symbol:
            local_methods.setdefault(symbol, []).append(dict(item))

    for returned in class_body_direct_return_calls(value):
        receiver = str(returned.get("receiver") or "").strip()
        name = str(returned.get("symbol") or "").strip()
        arity = int(returned.get("argument_count") or 0)
        target_type = str(returned.get("declared_return_type") or "").strip()

        if receiver in {"", "this"}:
            matching = [
                item
                for item in local_methods.get(name, ())
                if len(item.get("parameters") or ()) == arity
            ]
            origin = f"local call {name}(...)"
        else:
            owner_methods = dependencies.get(receiver)
            if owner_methods is None:
                continue
            matching = [
                item
                for item in owner_methods.get(name, ())
                if len(item.get("parameters") or ()) == arity
            ]
            origin = f"dependency call {receiver}.{name}(...)"

        source_types = {
            str(item.get("return_type") or "").strip()
            for item in matching
            if str(item.get("return_type") or "").strip()
        }
        if (
            len(source_types) == 1
            and _simple_object_type(next(iter(source_types)))
            and target_type
            and not _simple_object_type(target_type)
        ):
            violations.append(
                "method "
                f"{returned.get('method')!r} returns {target_type} but directly "
                f"returns Object-valued {origin}; perform explicit runtime type "
                "narrowing first."
            )

    if violations:
        # A missing method must not hide the next missing method or an incompatible
        # return. One bounded correction receives all facts from this candidate.
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: " + "\n".join(dict.fromkeys(violations))
        )


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


def _field_declared_type(chunk: str, field_name: str) -> str:
    """Return the exact declared type for one single-field declaration."""

    declaration = str(chunk or "").strip().rstrip(";").strip()
    if not declaration or not field_name:
        return ""
    assignment = _top_level_assignment_index(declaration)
    left = declaration[:assignment].strip() if assignment >= 0 else declaration
    match = re.search(rf"\b{re.escape(field_name)}\s*$", left)
    if match is None:
        return ""
    prefix = left[:match.start()].strip()
    tokens = prefix.split()
    while tokens and tokens[0] in _JAVA_MODIFIERS:
        tokens.pop(0)
    declared = " ".join(tokens).strip()
    if not declared or "@" in declared:
        return ""
    return declared


def _canonicalize_jdk_type_expression(value: str) -> str:
    """Canonicalize known JDK raw types while preserving generic structure."""

    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    if text.startswith("?"):
        for prefix in ("? extends ", "? super "):
            if text.startswith(prefix):
                return prefix + _canonicalize_jdk_type_expression(
                    text[len(prefix):]
                )
        return text
    start = text.find("<")
    if start < 0:
        return _canonical_jdk_class_name(text)
    depth = 0
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        return text
    raw = _canonical_jdk_class_name(text[:start].strip())
    args = [
        _canonicalize_jdk_type_expression(part)
        for part in _split_top_level(text[start + 1:end], ",")
        if part.strip()
    ]
    suffix = text[end + 1:].strip()
    rendered = raw + "<" + ", ".join(args) + ">"
    return rendered + ((" " + suffix) if suffix else "")


def _generic_type_parts(value: str) -> tuple[str, tuple[str, ...]]:
    text = _canonicalize_jdk_type_expression(value)
    start = text.find("<")
    if start < 0:
        return text, ()
    depth = 0
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        return text, ()
    raw = text[:start].strip()
    args = tuple(
        part.strip()
        for part in _split_top_level(text[start + 1:end], ",")
        if part.strip()
    )
    return raw, args


def _sibling_field_type_contracts(
    source: str,
    *,
    sibling_concerns: Sequence[str],
) -> dict[str, str]:
    contracts: dict[str, str] = {}
    for concern in sibling_concerns:
        members = _region_content(source, concern=concern, region="MEMBERS")
        for chunk in _top_level_member_chunks(members):
            symbols = _member_declaration_symbols(chunk)
            field_names = [
                display
                for key, display in symbols.items()
                if key.startswith("field:")
            ]
            if len(field_names) != 1:
                continue
            field_name = field_names[0]
            declared = _field_declared_type(chunk, field_name)
            if declared:
                contracts[field_name] = _canonicalize_jdk_type_expression(
                    declared
                )
    return contracts


def _map_type_arguments(value: str) -> tuple[str, str] | None:
    raw, args = _generic_type_parts(value)
    if raw not in {
        "java.util.Map",
        "java.util.HashMap",
        "java.util.LinkedHashMap",
        "java.util.concurrent.ConcurrentHashMap",
    } or len(args) != 2:
        return None
    return args[0], args[1]


def _map_entry_type_arguments(value: str) -> tuple[str, str] | None:
    raw, args = _generic_type_parts(value)
    if raw not in {"java.util.Map.Entry", "Map.Entry"} or len(args) != 2:
        return None
    return args[0], args[1]


def _map_projection_local_type(value: str) -> str:
    """Return a legal local-variable type for a Map key/value projection.

    Wildcards are legal as generic arguments but illegal as standalone local
    declaration types. Reading from an unbounded or lower-bounded wildcard is
    therefore represented as Object; an upper-bounded wildcard can safely use
    its upper bound.
    """

    expected = _canonicalize_jdk_type_expression(value).strip()
    if expected == "?":
        return "java.lang.Object"
    extends_prefix = "? extends "
    if expected.startswith(extends_prefix):
        upper = expected[len(extends_prefix):].strip()
        return upper or "java.lang.Object"
    if expected.startswith("?"):
        return "java.lang.Object"
    return expected


def _rewrite_map_entry_projection_types(
    value: str,
    *,
    authoritative_field_types: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Propagate exact Map<K,V> generics through entrySet/getKey/getValue.

    This is intentionally narrow: it changes only local types whose Java type is
    mechanically determined by Map.entrySet(), Map.Entry.getKey(), or getValue().
    """

    source = str(value or "")
    changes: list[str] = []
    entry_types: dict[str, tuple[str, str]] = {}

    loop_pattern = re.compile(
        r"for\s*\(\s*(?P<entry_type>(?:java\.util\.)?Map\.Entry\s*<.+>)"
        r"\s+(?P<entry_var>[A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*"
        r"(?P<map_var>[A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*entrySet\s*\(\s*\)\s*\)",
        re.DOTALL,
    )

    def replace_loop(match: re.Match[str]) -> str:
        entry_var = match.group("entry_var")
        map_var = match.group("map_var")
        candidate_entry = _map_entry_type_arguments(match.group("entry_type"))
        authoritative_map = _map_type_arguments(
            authoritative_field_types.get(map_var, "")
        )
        exact = authoritative_map or candidate_entry
        if exact is None:
            return match.group(0)
        key_type, value_type = exact
        entry_types[entry_var] = (key_type, value_type)
        expected = (
            "java.util.Map.Entry<"
            + key_type
            + ", "
            + value_type
            + ">"
        )
        actual = _canonicalize_jdk_type_expression(match.group("entry_type"))
        if authoritative_map is None or actual == expected:
            return match.group(0)
        changes.append(
            f"{entry_var}:entry_type:{actual}->{expected}"
        )
        original = match.group(0)
        start, end = match.span("entry_type")
        relative_start = start - match.start()
        relative_end = end - match.start()
        return original[:relative_start] + expected + original[relative_end:]

    source = loop_pattern.sub(replace_loop, source)

    for entry_var, (key_type, value_type) in tuple(entry_types.items()):
        for accessor, expected_type in (
            ("getKey", key_type),
            ("getValue", value_type),
        ):
            pattern = re.compile(
                rf"(?P<indent>^[ \t]*)(?P<declared>[^\n;=]+?)\s+"
                rf"(?P<local>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*"
                rf"{re.escape(entry_var)}\s*\.\s*{accessor}\s*\(\s*\)\s*;",
                re.MULTILINE,
            )

            def replace_projection(
                match: re.Match[str],
                *,
                expected_type: str = expected_type,
                accessor: str = accessor,
            ) -> str:
                declared = match.group("declared").strip()
                modifier_prefix = ""
                declared_type = declared
                if declared_type.startswith("final "):
                    modifier_prefix = "final "
                    declared_type = declared_type[6:].strip()
                actual = _canonicalize_jdk_type_expression(declared_type)
                expected = _map_projection_local_type(expected_type)
                if not expected or actual == expected:
                    return match.group(0)
                actual_raw, _actual_args = _generic_type_parts(actual)
                expected_raw, _expected_args = _generic_type_parts(expected)
                if actual_raw != expected_raw and actual not in {
                    "java.lang.Object",
                    "Object",
                }:
                    return match.group(0)
                changes.append(
                    f"{match.group('local')}:{entry_var}.{accessor}:"
                    f"{actual}->{expected}"
                )
                return (
                    match.group("indent")
                    + modifier_prefix
                    + expected
                    + " "
                    + match.group("local")
                    + " = "
                    + entry_var
                    + "."
                    + accessor
                    + "();"
                )

            source = pattern.sub(replace_projection, source)

    return source, tuple(changes)


def _sibling_symbol_inventory(
    source: str,
    *,
    sibling_concerns: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concern in sibling_concerns:
        members = _region_content(source, concern=concern, region="MEMBERS")
        try:
            contracts = class_body_member_contracts(members)
        except JavaRegionParseError:
            contracts = ()
        if contracts:
            for contract in contracts:
                row = {
                    "owner_concern": concern,
                    **dict(contract),
                    "typed_api_source": "tree_sitter_java",
                }
                if row.get("kind") == "field" and row.get("declared_type"):
                    row["declared_type"] = _canonicalize_jdk_type_expression(
                        str(row["declared_type"])
                    )
                    row["generic_type_is_authoritative"] = True
                rows.append(row)
            continue

        # Compatibility fallback for legacy/incomplete regions that cannot be
        # represented as a complete Tree-sitter class body.
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
                    "typed_api_source": "legacy_fallback",
                }
                if kind == "field":
                    row["mutable"] = not bool(
                        re.search(r"\bfinal\b", declaration)
                    )
                    declared_type = _field_declared_type(chunk, display)
                    if declared_type:
                        row["declared_type"] = (
                            _canonicalize_jdk_type_expression(declared_type)
                        )
                        row["generic_type_is_authoritative"] = True
                rows.append(row)
    return rows


def _prompt_sibling_api(
    source: str,
    *,
    sibling_concerns: Sequence[str],
) -> list[dict[str, Any]]:
    """Keep sibling authority exact but small enough for a local coder."""

    return [
        _compact_prompt_member_contract(
            row,
            owner_concern=str(row.get("owner_concern") or ""),
        )
        for row in _sibling_symbol_inventory(
            source,
            sibling_concerns=sibling_concerns,
        )
    ]


def _concern_semantic_fields(concern: Mapping[str, Any]) -> list[str]:
    schema = concern.get("record_schema")
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        return [
            str(item).strip()
            for item in required
            if str(item).strip()
        ]
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        return [
            str(name).strip()
            for name in properties
            if str(name).strip()
        ]
    return []


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
            "Return only compile-ready Java class-body source for this selected concern region. "
            "Do not return JSON, tool calls, Markdown, prose, package/import declarations, or the "
            "outer class wrapper. Emit complete semantic Java declarations: fields, methods, and "
            "only concern-owned private nested runtime types when genuinely required. "
            "Reuse available_sibling_api/dependency_api exactly; do not redeclare sibling state. "
            "If this concern needs state not present in available_sibling_api, declare the minimal "
            "private static concern-local backing field. Use fully-qualified JDK/external types "
            "when imports would otherwise be required. The existing outer class constructor and "
            "lifecycle are host-owned. Keep methods bounded and concern-local."
        )
        if name in _DECLARATION_ONLY_CONCERNS:
            response_contract += (
                " This is a declaration-only data concern. Emit at least one concern-owned "
                "field and/or private nested data type. Do not emit methods, initialize(), "
                "onInitialize(), registration hooks, load/save lifecycle methods, or calls whose "
                "only purpose is to invoke another class lifecycle. Encode the authored runtime "
                "data requirements in task_authority as data declarations in this region."
            )
        if str(section or "").strip() == "integration":
            response_contract += (
                " Initialization statements are generated in a separate host-owned initialize "
                "region. Never emit an initialize() wrapper in members. If this concern needs no "
                "class-body declarations or helper methods, return exactly "
                "'// no members required'."
            )
    elif response_region == "initialize":
        response_contract = (
            "Return only compile-ready Java statements or balanced control-flow blocks that belong "
            "inside the host-owned initialize() body. Do not return JSON, tool calls, Markdown, "
            "prose, package/import declarations, an initialize() wrapper, or the outer class. "
            "When no initialization is required, return exactly '// no initialization required'."
        )
    else:
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_RESPONSE_REGION_INVALID: {response_region!r}"
        )
    system = (
        "Emit one final Java region only. Do not think aloud, explain, draft, reconsider, "
        "or emit multiple candidate implementations. "
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
        "generic arguments, and mutability. Generic arguments are invariant authority: "
        "never narrow Map<K,Object> to Map<K,String> or otherwise substitute a different "
        "generic argument. For Map<K,V>.entrySet(), Map.Entry is exactly Map.Entry<K,V>, "
        "getKey() is exactly K, and getValue() is exactly V. Never treat an object/record "
        "field as a primitive, never assign "
        "to a field declared final, and never invent a sibling symbol that is not listed. "
        "This candidate must pass host semantic validation and compilation. "
        "If repair_failure or region_correction is supplied, correct the actual rejected "
        "candidate using those diagnostics and preserve unrelated declarations. "
        "Resolve every supplied semantic/API fact before emitting source, and do not implement sibling concerns. "
        + (
            "This section is pure Java domain logic. Do not reference net.minecraft.*, "
            "net.fabricmc.*, registries, resource identifiers, packets, lifecycle hooks, or "
            "game registration APIs. "
            if _section_platform_api_policy(section) == "forbidden"
            else (
                "This is a platform-bound section. You may reference net.minecraft.* or "
                "net.fabricmc.* only when the exact owner is present in implementation_authority "
                "or host_grounding. If no such owner is supplied, keep this concern platform-neutral "
                "and use only JDK/dependency APIs. "
            )
        )
        + "Use only supplied host grounding and dependency APIs; never invent a Minecraft/Fabric API. "
        "dependency_call_contract is exhaustive for dependency method calls: match owner, method, "
        "static=true, arity, parameter types, and parameter_names exactly. parameter_names are semantic "
        "roles: never use a trigger_dispatch/event_dispatch method as a key/value setter; use an exact "
        "direct_state_write signature for direct named-state assignment. If no exact row exists, do not "
        "emit the call. Never add arguments to a zero-arity method. Never call a dependency initialize/"
        "onInitialize lifecycle hook from a concern region."
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
            "task": str(concern.get("task") or "").strip(),
            "rules": [
                str(item).strip()
                for item in concern.get("rules") or ()
                if str(item).strip()
            ],
            "semantic_fields": _concern_semantic_fields(concern),
            # Planning templates describe record extraction, not runtime Java.
            # Authored requirements/records remain in task_authority; never send
            # the planner's "return the next record/done" instructions to a coder.
            "implementation_goal": (
                f"Implement only the {name} semantics stated in "
                "task_authority.source_requirements inside the selected Java class."
            ),
            "java_shape": (
                "declarations_only_fields_or_private_nested_types"
                if name in _DECLARATION_ONLY_CONCERNS and response_region == "members"
                else "concern_owned_class_body_members"
            ),
        },
        "task_authority": _concern_authority(task, concern),
        "state_variable_contract": (
            list(_state_variable_contract(task, concern))
            if section == "state_model" and name == "variables"
            else []
        ),
        "host_grounding": _bounded_grounding(grounding, section=section),
        "implementation_authority": (
            render_generation_implementation_authority_prompt(grounding)
            if _section_platform_api_policy(section) != "forbidden"
            else ""
        ),
        "dependency_api": _dependency_api_context(dependency_source),
        "dependency_call_contract": _dependency_call_contracts(dependency_source),
        "current_selected_region_source": _region_content(
            current_source,
            concern=name,
            region="INIT" if response_region == "initialize" else "MEMBERS",
        ),
        "available_sibling_api": _prompt_sibling_api(
            current_source,
            sibling_concerns=sibling_concerns,
        ),
        "repair_failure": failure or None,
        "generation_recipe": {
            "first_pass_goal": (
                "produce the smallest compile-ready semantic Java source region in one response "
                "and finish well inside the finite output page"
            ),
            "declare_local_domain_types_first": True,
            "require_fully_qualified_external_types": True,
            "output_language": "java_source_region",
            "no_json_ast_protocol": True,
            "sibling_api_is_authoritative": True,
            "dependency_call_contract_is_exhaustive": True,
            "dependency_call_rule": (
                "Before emitting Owner.method(...), match owner, method, arity, static=true, and "
                "parameter_names against dependency_call_contract. Treat parameter_names as semantic "
                "roles. trigger_dispatch/event_dispatch methods are not key/value setters; direct "
                "state assignment requires a direct_state_write signature. If no exact row exists, "
                "do not emit that call. Dependency initialize/onInitialize hooks are host-orchestrated; "
                "concern regions must not call them."
            ),
            "never_mutate_final_sibling_fields": True,
            "declare_missing_concern_local_state": (
                "When this concern reads or writes state absent from available_sibling_api, "
                "declare a private static non-final backing field with a compatible runtime "
                "value type instead of referencing an undeclared symbol or inventing a metadata DTO."
            ),
            **production_java_generation_recipe_policy(),
            "compile_ready_examples": [
                {
                    "bad": "java.util.Map<String,Object> lock = new java.util.ReentrantLock();",
                    "good": "java.util.concurrent.locks.Lock lock = new java.util.concurrent.locks.ReentrantLock();",
                },
                {
                    "bad": "java.util.Map<String,String> value = entry.getValue(); // entry is Map.Entry<String,Map<String,Object>>",
                    "good": "java.util.Map<String,Object> value = entry.getValue();",
                },
                {
                    "bad": "String value = objectMap.get(key);",
                    "good": "Object raw = objectMap.get(key); String value = raw instanceof String s ? s : \"\";",
                },
            ],
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
            "required_output_format": "plain_java_source",
            "model_tools_enabled": False,
            "sibling_concerns_out_of_scope": list(sibling_concerns),
            "scope_rule": (
                "Implement only the selected concern and only the lines in "
                "task_authority.source_requirements. Do not pre-implement sibling concerns. "
                "The host owns declaration ownership and sibling bookkeeping; earlier declarations are immutable."
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

def _compile_report_timed_out(report: Any) -> bool:
    if str(getattr(report, "status", "") or "").strip().upper() == "TIMEOUT":
        return True
    for command in tuple(getattr(report, "commands", ()) or ()):
        if bool(getattr(command, "timed_out", False)):
            return True
        try:
            if int(getattr(command, "exit_code", 0) or 0) == 124:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _compile_timeout_message(report: Any) -> str:
    commands = tuple(getattr(report, "commands", ()) or ())
    command = commands[-1] if commands else None
    duration = getattr(command, "duration_seconds", None) if command is not None else None
    log_path = str(getattr(command, "log_path", "") or "") if command is not None else ""
    suffix = []
    if duration is not None:
        suffix.append(f"duration_seconds={duration}")
    if log_path:
        suffix.append(f"log_path={log_path}")
    detail = "; ".join(suffix)
    return (
        "ATOMIC_CONCERN_COMPILE_TIMEOUT: Gradle compileJava timed out; "
        "this is not a localized Java source diagnostic, so model source repair was not attempted."
        + (f" {detail}" if detail else "")
    )


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
    region_attempt_limit: int | None = None
    retry_structural_rejections: bool = True
    canonicalize_local_final_rebindings: bool = (
        PRODUCTION_CANONICALIZE_LOCAL_FINAL_REBINDINGS
    )
    compile_repair_limit: int | None = None
    completion_decider: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    ordered: tuple[dict[str, Any], ...] = field(init=False)
    source: str = field(init=False)
    state: dict[str, tuple[str, str]] = field(default_factory=dict, init=False)
    summaries: list[str] = field(default_factory=list, init=False)
    seen_failures: set[str] = field(default_factory=set, init=False)
    host_owned_concerns: set[str] = field(default_factory=set, init=False)
    repairs: int = field(default=0, init=False)
    first_pass_rejections: int = field(default=0, init=False)
    repair_region_rejections: int = field(default=0, init=False)
    first_compile_failures: int = field(default=0, init=False)

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


    def _assert_symbol_ownership(self, *, concern: str, members: str) -> None:
        owners = self._sibling_symbol_owners(exclude=concern)
        collisions = sorted(set(_member_declaration_symbols(members)) & set(owners))
        if collisions:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_OWNERSHIP_VIOLATION: accepted sibling declarations are immutable: "
                + "; ".join(f"{key} belongs to {owners[key]}" for key in collisions)
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
        correction = None
        dependency_repair = None
        rejected_region = ""
        attempt_limit = (
            _region_attempt_limit()
            if self.region_attempt_limit is None
            else max(1, int(self.region_attempt_limit))
        )

        if (
            not failure
            and response_region == "members"
            and str(self.section or "").strip() == "behavior_contract"
        ):
            host_members = _deterministic_behavior_contract_members(
                self.task,
                concern,
            )
            output_sha = hashlib.sha256(host_members.encode("utf-8")).hexdigest()
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
            self.host_owned_concerns.add(name)
            return host_members

        if (
            not failure
            and response_region == "members"
            and str(self.section or "").strip() == "state_model"
        ):
            from .structured_state_runtime import render_state_model_concern

            first = _slug(self.ordered[0]["concern"]) if self.ordered else name
            host_members = render_state_model_concern(
                self.task,
                name,
                include_runtime=name == first,
            )
            if host_members is not None:
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
                self.host_owned_concerns.add(name)
                return host_members

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
                self.host_owned_concerns.add(name)
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
            output_sha = ""
            parsed = ""
            candidate_merged = False
            try:
                from .atomic_region_paging import generate_region

                messages = _messages(
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
                )
                if rejected_region:
                    payload = json.loads(messages[-1]["content"])
                    payload["current_selected_region_source"] = rejected_region
                    if correction is not None:
                        payload["region_correction"] = correction.payload()
                        # The immutable declarations already have their own exact
                        # context; avoid sending the same full region twice.
                        payload["current_selected_region_source"] = "\n\n".join(
                            payload["region_correction"]["selected_declarations"]
                        )
                        messages[0]["content"] += (
                            "\nCORRECTION TURN: region_correction narrows this response to "
                            "the listed selected declarations. The host retains all others."
                        )
                    if dependency_repair is not None:
                        payload["dependency_repair_contract"] = dependency_repair
                        messages[0]["content"] += (
                            "\nDEPENDENCY CORRECTION TURN: dependency_repair_contract is "
                            "authoritative. Select only a listed invocation_shape and preserve "
                            "its parameter roles exactly."
                        )
                    messages[-1]["content"] = json.dumps(payload, ensure_ascii=False)
                output = generate_region(
                    self.call_coder, messages, completion_decider=self.completion_decider,
                )
                output_text = str(output or "")
                output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                allow_integration_empty_members = (
                    response_region == "members"
                    and self.require_initialize
                    and str(self.section or "").strip() == "integration"
                )
                parsed = _parse_region_content(
                    output,
                    response_region=response_region,
                    allow_inert_empty=allow_integration_empty_members,
                    allow_host_initialize_only_empty=allow_integration_empty_members,
                )
                if correction is not None:
                    parsed = correction.merge(parsed)
                candidate_merged = True
                parsed, lifecycle_changes = (
                    _strip_host_orchestrated_dependency_lifecycle_calls(
                        parsed,
                        dependency_source=self.dependency_source,
                    )
                )
                if lifecycle_changes:
                    from .root_cause_trace import emit_root_cause

                    emit_root_cause(
                        "atomic_concern_dependency_lifecycle_host_owned",
                        stage="production",
                        operation="atomic_concern_region",
                        gate="host_lifecycle_canonicalization",
                        result="PASS",
                        details={
                            "concern": name,
                            "calls_removed": list(lifecycle_changes),
                        },
                    )
                if response_region == "members":
                    dependency_canonical, dependency_changes = (
                        _canonicalize_dependency_call_semantics(
                            parsed,
                            dependency_source=self.dependency_source,
                        )
                    )
                    if dependency_changes:
                        from .root_cause_trace import emit_root_cause

                        emit_root_cause(
                            "atomic_concern_dependency_call_canonicalized",
                            stage="production",
                            operation="atomic_concern_region",
                            gate="dependency_call_semantic_canonicalization",
                            result="PASS",
                            details={
                                "concern": name,
                                "changes": list(dependency_changes),
                            },
                        )
                    parsed = dependency_canonical
                    sibling_names = tuple(
                        _slug(item["concern"])
                        for item in self.ordered
                        if _slug(item["concern"]) != name
                    )
                    canonical, canonical_changes = _canonicalize_generated_jdk_semantics(
                        parsed,
                        authoritative_field_types=_sibling_field_type_contracts(
                            self.source,
                            sibling_concerns=sibling_names,
                        ),
                        protected_simple_types=(
                            *self._known_simple_types(exclude=name),
                            *_dependency_authorized_simple_types(
                                self.dependency_source
                            ),
                        ),
                    )
                    if canonical_changes:
                        from .root_cause_trace import emit_root_cause

                        emit_root_cause(
                            "atomic_concern_jdk_semantics_canonicalized",
                            stage="production",
                            operation="atomic_concern_region",
                            gate="first_pass_semantic_canonicalization",
                            result="PASS",
                            details={
                                "concern": name,
                                "changes": list(canonical_changes),
                            },
                        )
                    parsed = canonical
                    if self.canonicalize_local_final_rebindings:
                        parsed, final_rebinding_changes = (
                            _canonicalize_local_final_rebindings(parsed)
                        )
                        if final_rebinding_changes:
                            from .root_cause_trace import emit_root_cause

                            emit_root_cause(
                                "atomic_concern_local_final_rebinding_canonicalized",
                                stage="production",
                                operation="atomic_concern_region",
                                gate="first_pass_semantic_canonicalization",
                                result="PASS",
                                details={
                                    "concern": name,
                                    "changes": list(final_rebinding_changes),
                                },
                            )
                    _validate_first_pass_java_semantics(
                        parsed,
                        dependency_source=self.dependency_source,
                        sibling_api=_sibling_symbol_inventory(
                            self.source,
                            sibling_concerns=sibling_names,
                        ),
                    )
                if response_region == "members" and name in _DECLARATION_ONLY_CONCERNS:
                    kinds = class_body_member_kinds(parsed)
                    if kinds and any(
                        kind not in _DECLARATION_ONLY_MEMBER_KINDS
                        for kind in kinds
                    ):
                        projected, dropped = _project_declaration_only_members(parsed)
                        if projected:
                            from .root_cause_trace import emit_root_cause

                            emit_root_cause(
                                "atomic_concern_declaration_only_projection",
                                stage="production",
                                operation="atomic_concern_region",
                                gate="semantic_shape_projection",
                                result="PASS",
                                details={
                                    "concern": name,
                                    "dropped_member_kinds": list(dropped),
                                    "retained_member_kinds": list(
                                        class_body_member_kinds(projected)
                                    ),
                                },
                            )
                            parsed = projected
                            kinds = class_body_member_kinds(parsed)
                    if (
                        not kinds
                        or any(
                            kind not in _DECLARATION_ONLY_MEMBER_KINDS
                            for kind in kinds
                        )
                    ):
                        raise CustomModuleGenerationError(
                            "ATOMIC_CONCERN_SEMANTIC_SHAPE_INVALID: "
                            f"{name} must contain at least one field/private nested data type "
                            f"and no outer methods; found {list(kinds)!r}."
                        )
                _validate_platform_api_admission(
                    parsed,
                    section=self.section,
                    grounding=self.grounding,
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
                if isinstance(exc, AtomicJavaDecisionError) and exc.response_text is not None:
                    output_text = exc.response_text
                    output_sha = exc.response_sha256 or ""
                reason = str(exc)
                rejected_response = (
                    exc.response_text
                    if isinstance(exc, AtomicJavaDecisionError) and exc.response_text is not None
                    else (output_text or None)
                )
                recoverable = atomic_error_recoverable(reason)
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
                        rejected_response=rejected_response,
                    )
                    raise

                structural_terminal = (
                    not self.retry_structural_rejections
                    and atomic_error_terminal_after_normalization(reason)
                )
                if structural_terminal:
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
                        rejected_response=rejected_response,
                    )
                    raise CustomModuleGenerationError(
                        f"ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID: "
                        f"{name}:{response_region} failed after 1 production decode(s): "
                        f"{reason}"
                    ) from exc

                violation = (reason, output_sha)
                if output_sha and violation in seen_violations:
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
                        rejected_response=rejected_response,
                    )
                    raise CustomModuleGenerationError(
                        f"ATOMIC_CONCERN_RESPONSE_NO_PROGRESS: {name}:{response_region} "
                        f"repeated identical invalid output: {reason}"
                    ) from exc

                if output_sha:
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
                    rejected_response=rejected_response,
                )
                if attempt >= attempt_limit:
                    code = (
                        "ATOMIC_CONCERN_FIRST_PASS_RESPONSE_INVALID"
                        if attempt_limit == 1
                        else "ATOMIC_CONCERN_RESPONSE_RETRY_EXHAUSTED"
                    )
                    raise CustomModuleGenerationError(
                        f"{code}: {name}:{response_region} failed after "
                        f"{attempt_limit} production decode(s): {reason}"
                    ) from exc

                if failure:
                    self.repair_region_rejections += 1
                else:
                    self.first_pass_rejections += 1

                dependency_repair = _dependency_repair_contract(
                    self.dependency_source,
                    reason,
                )
                if candidate_merged:
                    # Preserve the actual rejected source; do not ask the model to
                    # recreate it from the original blank scaffold and an error name.
                    rejected_region = parsed
                    if response_region == "members":
                        from .atomic_region_correction import RegionCorrection

                        correction = RegionCorrection.for_diagnostic(
                            parsed,
                            reason,
                            allow_private_restructure=(
                                not failure
                                and name not in self.state
                                and reason.startswith(
                                    (
                                        "ATOMIC_CONCERN_RESPONSE_INVALID:",
                                        "ATOMIC_CONCERN_SEMANTIC_SHAPE_INVALID:",
                                    )
                                )
                            ),
                        )
                elif not rejected_region:
                    rejected_region = output_text

                declaration_only_rule = (
                    " This concern is declaration-only: emit fields and/or private nested "
                    "data types only; outer methods are forbidden."
                    if response_region == "members"
                    and name in _DECLARATION_ONLY_CONCERNS
                    else ""
                )
                validation_failure = (
                    "HOST REGION VALIDATION FAILED BEFORE COMPILATION:\n"
                    + reason
                    + f"\nCorrect only the rejected {response_region} candidate shown in current_selected_region_source."
                    + " When region_correction is present, emit only its selected declarations."
                    + declaration_only_rule
                    + " Do not emit response markers, prose, package/import/top-level/lifecycle declarations. "
                    "Do not introduce, rename, or change the kind of nested types during bounded regeneration. "
                    "Fix only the rejected concern region and preserve valid sibling declarations. "
                    "Implement only this concern; do not add declarations for sibling concerns. "
                    "Earlier sibling declarations are immutable and cannot be redeclared. "
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
        owners = self._sibling_symbol_owners(exclude=name)
        members, dropped_redeclarations = _drop_authoritative_sibling_redeclarations(
            members,
            owners=owners,
        )
        if dropped_redeclarations:
            from .root_cause_trace import emit_root_cause

            emit_root_cause(
                "atomic_concern_sibling_redeclaration_dropped",
                stage="production",
                operation="atomic_concern_region",
                gate="host_symbol_ownership",
                result="PASS",
                details={
                    "concern": name,
                    "symbols": list(dropped_redeclarations),
                    "owners": {
                        key: owners[key]
                        for key in dropped_redeclarations
                    },
                },
            )
        self._assert_symbol_ownership(concern=name, members=members)
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
        if _compile_report_timed_out(report):
            raise CustomModuleGenerationError(_compile_timeout_message(report))
        failure = self.compile_log(report) or str(
            getattr(report, "error", "") or "Gradle compileJava failed."
        )
        name = _failure_concern(self.source, log=failure, relative=self.relative)
        if name in self.host_owned_concerns:
            raise CustomModuleGenerationError(
                "STRUCTURED_STATE_HOST_COMPILER_INVALID: host-generated declarations "
                "cannot be replaced by model repair.\n"
                + _compact_compiler_failure(failure, source=self.source,
                                            relative=self.relative, concern=name)
            )
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

        fingerprint = _compiler_repair_fingerprint(
            failure,
            source=self.source,
            relative=self.relative,
            concern=name,
        )
        if fingerprint in self.seen_failures:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_NO_PROGRESS: identical compiler diagnostics "
                "recurred against the identical concern source.\n"
                + _compact_compiler_failure(
                    failure,
                    source=self.source,
                    relative=self.relative,
                    concern=name,
                )
            )
        self.seen_failures.add(fingerprint)

        repair_context = _compiler_repair_context(
            failure,
            source=self.source,
            relative=self.relative,
            concern=name,
        )
        self._apply(self._concern(name), failure=repair_context)
        self.repairs += 1
        return self._compile()

    def run(self) -> dict[str, Any]:
        from .structured_state_runtime import has_complete_structured_state

        state_model = str(self.section or "").strip() == "state_model"
        structured_state = (
            state_model
            and has_complete_structured_state(self.task, self.ordered)
        )
        variable_only_host_lowering = (
            state_model
            and bool(self.ordered)
            and all(_slug(item["concern"]) == "variables" for item in self.ordered)
            and all(
                bool(_deterministic_state_variable_members(self.task, item))
                for item in self.ordered
            )
        )
        if state_model and not (structured_state or variable_only_host_lowering):
            raise CustomModuleGenerationError(
                "STRUCTURED_STATE_CONTRACT_REQUIRED: state_model execution is "
                "host-compiled only. Canonical structured state records must be "
                "present before production; free-form coder Java fallback is disabled."
            )

        if state_model:
            for concern in self.ordered:
                self._apply(concern)
            report = self._compile()
            if getattr(report, "status", "") != "PASS":
                if _compile_report_timed_out(report):
                    raise CustomModuleGenerationError(_compile_timeout_message(report))
                failure = self.compile_log(report) or str(
                    getattr(report, "error", "")
                    or "Host-compiled structured state model did not compile."
                )
                raise CustomModuleGenerationError(
                    "STRUCTURED_STATE_HOST_COMPILER_INVALID:\n"
                    + _compact_compiler_failure(
                        failure,
                        source=self.source,
                        relative=self.relative,
                        concern="",
                    )
                )
            return {
                "source": self.source,
                "summary": " | ".join(self.summaries),
                "concern_count": len(self.ordered),
                "repair_count": 0,
                "first_pass_rejection_count": self.first_pass_rejections,
                "repair_region_rejection_count": self.repair_region_rejections,
                "first_compile_failure_count": self.first_compile_failures,
            }

        # Build the complete host-owned concern file before invoking Gradle.
        # Tree-sitter and the semantic admission gates validate each concern locally;
        # javac/Gradle is the integration gate for the complete source, not a
        # checkpoint after every partially populated concern.
        compile_repair_limit = (
            _compile_repair_limit()
            if self.compile_repair_limit is None
            else max(0, int(self.compile_repair_limit))
        )
        for concern in self.ordered:
            self._apply(concern)

        report = self._compile()
        if getattr(report, "status", "") != "PASS":
            self.first_compile_failures += 1

        if getattr(report, "status", "") != "PASS" and compile_repair_limit <= 0:
            if _compile_report_timed_out(report):
                raise CustomModuleGenerationError(_compile_timeout_message(report))
            failure = self.compile_log(report) or str(
                getattr(report, "error", "") or "Gradle compileJava failed."
            )
            failing_name = _failure_concern(
                self.source,
                log=failure,
                relative=self.relative,
            )
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_FIRST_PASS_COMPILE_FAILED: production compile failed "
                "after the single generated candidate; model compiler-repair is disabled.\n"
                + _compact_compiler_failure(
                    failure,
                    source=self.source,
                    relative=self.relative,
                    concern=failing_name,
                )
            )

        repair_counts: dict[str, int] = {}
        while getattr(report, "status", "") != "PASS":
            if _compile_report_timed_out(report):
                raise CustomModuleGenerationError(_compile_timeout_message(report))

            failure = self.compile_log(report) or str(
                getattr(report, "error", "") or "Gradle compileJava failed."
            )
            failing_name = _failure_concern(
                self.source,
                log=failure,
                relative=self.relative,
            )
            if failing_name:
                concern_repairs = repair_counts.get(failing_name, 0)
                if concern_repairs >= compile_repair_limit:
                    raise CustomModuleGenerationError(
                        "ATOMIC_CONCERN_COMPILE_REPAIR_EXHAUSTED: "
                        f"{failing_name} still did not compile after "
                        f"{compile_repair_limit} explicitly enabled concern-local repairs.\n"
                        + _compact_compiler_failure(
                            failure,
                            source=self.source,
                            relative=self.relative,
                            concern=failing_name,
                        )
                    )

            report = self._repair_once(report)
            if failing_name:
                repair_counts[failing_name] = repair_counts.get(failing_name, 0) + 1
        return {
            "source": self.source,
            "summary": " | ".join(self.summaries),
            "concern_count": len(self.ordered),
            "repair_count": self.repairs,
            "first_pass_rejection_count": self.first_pass_rejections,
            "repair_region_rejection_count": self.repair_region_rejections,
            "first_compile_failure_count": self.first_compile_failures,
        }

__all__ = [
    "END_MARKER",
    "INITIALIZE_MARKER",
    "MEMBERS_MARKER",
    "AtomicConcernExecutor",
    "build_concern_scaffold",
    "parse_concern_content",
]
