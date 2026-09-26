from __future__ import annotations

"""Host-owned concern regions for bounded small-model Java generation."""

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .custom_module_errors import CustomModuleGenerationError

MEMBERS_MARKER = "<<<MMM_CONCERN_MEMBERS>>>"
INITIALIZE_MARKER = "<<<MMM_CONCERN_INITIALIZE>>>"
END_MARKER = "<<<MMM_CONCERN_END>>>"
_HOST_PREFIX = "MMM_ATOMIC_CONCERN"
_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*;\s*$")
_FORBIDDEN = re.compile(
    r"\b(?:package|import)\s+|\b(?:public\s+)?(?:class|interface|enum|record)\b"
    r"|\b(?:ModInitializer|ClientModInitializer|DedicatedServerModInitializer)\b"
    r"|\bonInitialize(?:Client|Server)?\b"
)
_INITIALIZE_DECL = re.compile(r"\bpublic\s+static\s+void\s+initialize\s*\(")


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().casefold()).strip("_")
    if not slug or re.fullmatch(r"[a-z][a-z0-9_]*", slug) is None:
        raise CustomModuleGenerationError(f"ATOMIC_CONCERN_INVALID_NAME: {value!r}")
    return slug


def _marker(concern: str, region: str, edge: str) -> str:
    return f"// {_HOST_PREFIX}_{_slug(concern).upper()}_{region}_{edge}"


def parse_concern_content(text: str, *, section: str) -> tuple[str, str]:
    """Accept only the active concern body; sibling/class authority is never model-owned."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    for marker in (MEMBERS_MARKER, INITIALIZE_MARKER, END_MARKER):
        if raw.count(marker) != 1:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: each host response marker must occur exactly once."
            )
    members_at = raw.index(MEMBERS_MARKER)
    init_at = raw.index(INITIALIZE_MARKER)
    end_at = raw.index(END_MARKER)
    if not members_at <= init_at <= end_at:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: host response markers are out of order."
        )
    if raw[:members_at].strip() or raw[end_at + len(END_MARKER):].strip():
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: output outside host response markers is forbidden."
        )
    members = raw[members_at + len(MEMBERS_MARKER):init_at].strip()
    initialize = raw[init_at + len(INITIALIZE_MARKER):end_at].strip()
    for value in (members, initialize):
        if _HOST_PREFIX in value:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: host source markers are reserved."
            )
        if "```" in value:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: Markdown code fences are forbidden."
            )
    if _FORBIDDEN.search(members) or _INITIALIZE_DECL.search(members):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_SCOPE_ESCAPE: concern members attempted to change host-owned type/lifecycle structure."
        )
    if str(section or "").strip() != "integration" and initialize:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_SCOPE_ESCAPE: only integration concerns may add initialize() statements."
        )
    if _FORBIDDEN.search(initialize) or _INITIALIZE_DECL.search(initialize):
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_SCOPE_ESCAPE: initialize body attempted to declare host-owned lifecycle/type structure."
        )
    return members, initialize


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


def _failure_measure(log: str) -> int:
    errors = {
        line.strip()
        for line in str(log or "").splitlines()
        if "error" in line.casefold() and line.strip()
    }
    return len(errors) or 1


def _messages(
    *,
    section: str,
    concern: Mapping[str, Any],
    task: Mapping[str, Any],
    grounding: Mapping[str, Any],
    dependency_source: str,
    current_source: str,
    failure: str = "",
) -> list[dict[str, str]]:
    name = _slug(concern.get("concern"))
    system = (
        "Implement exactly one host-selected concern inside one already-selected Java class. "
        "You do not choose files, classes, dependencies, architecture, tools, search routes, APIs, or sibling work. "
        "Return the normal JSON content/summary envelope. content must contain exactly:\n"
        + MEMBERS_MARKER + "\n<class-body members for this concern only>\n"
        + INITIALIZE_MARKER + "\n<initialize statements only when section=integration; otherwise empty>\n"
        + END_MARKER + "\n"
        "Use fully-qualified external API names when needed; do not emit import/package/type declarations. "
        "Do not emit initialize() itself and do not restate sibling concern blocks. "
        "Use only supplied host grounding and dependency source; never invent a Minecraft/Fabric API."
    )
    payload = {
        "phase": "implement_atomic_concern",
        "section": section,
        "concern": {
            "sequence": concern.get("sequence"),
            "identifier": concern.get("identifier"),
            "name": name,
            "task": concern.get("task"),
            "rules": concern.get("rules") or [],
            "record_schema": concern.get("record_schema") or {},
        },
        "approved_task": dict(task),
        "host_grounding": dict(grounding),
        "dependency_source": dependency_source,
        "current_host_owned_source": current_source,
        "repair_failure": failure or None,
        "scope": {
            "members_region": _marker(name, "MEMBERS", "START"),
            "initialize_region": _marker(name, "INIT", "START"),
            "sibling_regions_immutable": True,
            "model_tool_choice": False,
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def generate_atomic_concerns(
    *,
    root: Path,
    target: Path,
    relative: str,
    symbol: str,
    original: str,
    task: Mapping[str, Any],
    section: str,
    concerns: Sequence[Mapping[str, Any]],
    grounding: Mapping[str, Any],
    dependency_source: str,
    require_initialize: bool,
    call_coder: Callable[[Sequence[Mapping[str, str]]], Mapping[str, str]],
    compile_java: Callable[[Path], Any],
    compile_log: Callable[[Any], str],
    write_source: Callable[[Path, str], None],
) -> dict[str, Any]:
    """Execute fixed concerns serially; repair only the compiler-localized concern."""
    ordered = _validate_concerns(concerns)
    source = build_concern_scaffold(
        original, symbol=symbol, concerns=ordered, require_initialize=require_initialize
    )
    state: dict[str, tuple[str, str]] = {}
    summaries: list[str] = []
    for concern in ordered:
        name = _slug(concern["concern"])
        payload = call_coder(_messages(
            section=section, concern=concern, task=task, grounding=grounding,
            dependency_source=dependency_source, current_source=source,
        ))
        members, initialize = parse_concern_content(str(payload.get("content") or ""), section=section)
        source = _replace_region(source, concern=name, region="MEMBERS", content=members)
        if require_initialize:
            source = _replace_region(source, concern=name, region="INIT", content=initialize)
        state[name] = (members, initialize)
        summaries.append(f"{name}: {str(payload.get('summary') or '').strip()}")

    write_source(target, source)
    report = compile_java(root)
    best_measure = _failure_measure(compile_log(report)) if getattr(report, "status", "") != "PASS" else 0
    seen_failures: set[str] = set()
    repairs = 0
    while getattr(report, "status", "") != "PASS":
        failure = compile_log(report) or str(getattr(report, "error", "") or "Gradle compileJava failed.")
        fingerprint = hashlib.sha256(failure.encode("utf-8")).hexdigest()
        if fingerprint in seen_failures:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_NO_PROGRESS: compiler diagnostics repeated.\n" + failure
            )
        seen_failures.add(fingerprint)
        name = _failure_concern(source, log=failure, relative=relative)
        if not name:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_UNLOCALIZED: failure is outside every active concern region.\n" + failure
            )
        concern = next((item for item in ordered if _slug(item["concern"]) == name), None)
        if concern is None:
            raise CustomModuleGenerationError(f"ATOMIC_CONCERN_UNKNOWN_REPAIR_SCOPE: {name}")
        payload = call_coder(_messages(
            section=section, concern=concern, task=task, grounding=grounding,
            dependency_source=dependency_source, current_source=source, failure=failure,
        ))
        members, initialize = parse_concern_content(str(payload.get("content") or ""), section=section)
        if state.get(name) == (members, initialize):
            raise CustomModuleGenerationError(
                f"ATOMIC_CONCERN_REPAIR_NO_PROGRESS: {name} repeated the same bounded source."
            )
        source = _replace_region(source, concern=name, region="MEMBERS", content=members)
        if require_initialize:
            source = _replace_region(source, concern=name, region="INIT", content=initialize)
        state[name] = (members, initialize)
        summaries.append(f"{name} repair: {str(payload.get('summary') or '').strip()}")
        repairs += 1
        write_source(target, source)
        report = compile_java(root)
        if getattr(report, "status", "") != "PASS":
            next_failure = compile_log(report) or str(getattr(report, "error", "") or "")
            next_measure = _failure_measure(next_failure)
            if next_measure >= best_measure:
                raise CustomModuleGenerationError(
                    "ATOMIC_CONCERN_REPAIR_NO_PROGRESS: compiler error measure did not strictly decrease.\n" + next_failure
                )
            best_measure = next_measure

    return {
        "source": source,
        "summary": " | ".join(summaries),
        "concern_count": len(ordered),
        "repair_count": repairs,
    }


__all__ = [
    "END_MARKER",
    "INITIALIZE_MARKER",
    "MEMBERS_MARKER",
    "build_concern_scaffold",
    "generate_atomic_concerns",
    "parse_concern_content",
]
