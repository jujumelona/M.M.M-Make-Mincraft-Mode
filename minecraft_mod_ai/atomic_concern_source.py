from __future__ import annotations

"""Host-owned concern regions for bounded small-model Java generation."""

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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


def _split_response_regions(text: str) -> tuple[str, str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    markers = (MEMBERS_MARKER, INITIALIZE_MARKER, END_MARKER)
    if any(raw.count(marker) != 1 for marker in markers):
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
    # Prefix/suffix text is never materialized into source. Small models may wrap
    # the marker protocol in prose or an outer Markdown fence; ignore that inert
    # envelope while keeping marker cardinality/order and both executable regions
    # fail-closed below.
    return (
        raw[members_at + len(MEMBERS_MARKER):init_at].strip(),
        raw[init_at + len(INITIALIZE_MARKER):end_at].strip(),
    )


def _validate_region_text(value: str, *, initialize_region: bool) -> None:
    if _HOST_PREFIX in value or "```" in value:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: host markers and Markdown fences are forbidden."
        )
    if _FORBIDDEN.search(value) or _INITIALIZE_DECL.search(value):
        region = "initialize body" if initialize_region else "concern members"
        raise CustomModuleGenerationError(
            f"ATOMIC_CONCERN_SCOPE_ESCAPE: {region} attempted to change host-owned type/lifecycle structure."
        )


def parse_concern_content(text: str, *, section: str) -> tuple[str, str]:
    """Accept only the active concern body; sibling/class authority is never model-owned."""
    members, initialize = _split_response_regions(text)
    _validate_region_text(members, initialize_region=False)
    _validate_region_text(initialize, initialize_region=True)
    if str(section or "").strip() != "integration" and initialize:
        raise CustomModuleGenerationError(
            "ATOMIC_CONCERN_SCOPE_ESCAPE: only integration concerns may add initialize() statements."
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
                    "source_requirements": dict(outer.get("source_requirements") or {}),
                    "section_instruction": str(instruction.get("section_instruction") or ""),
                }
                break
    return {
        "task_id": str(task.get("task_id") or ""),
        "semantic_outcome": str(task.get("semantic_outcome") or ""),
        **requirement_payload,
    }


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
        "Return only the following plain-text marker protocol. No JSON, prose, or Markdown fences:\n"
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
        "task_authority": _concern_authority(task, concern),
        "host_grounding": _bounded_grounding(grounding),
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
    best_measure: int = field(default=0, init=False)

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

    def _apply(self, concern: Mapping[str, Any], *, failure: str = "") -> None:
        name = _slug(concern["concern"])
        output = self.call_coder(_messages(
            section=self.section,
            concern=concern,
            task=self.task,
            grounding=self.grounding,
            dependency_source=self.dependency_source,
            current_source=self.source,
            failure=failure,
        ))
        members, initialize = parse_concern_content(output, section=self.section)
        if failure and self.state.get(name) == (members, initialize):
            raise CustomModuleGenerationError(
                f"ATOMIC_CONCERN_REPAIR_NO_PROGRESS: {name} repeated the same bounded source."
            )
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
        fingerprint = hashlib.sha256(failure.encode("utf-8")).hexdigest()
        if fingerprint in self.seen_failures:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_NO_PROGRESS: compiler diagnostics repeated.\n"
                + failure
            )
        self.seen_failures.add(fingerprint)
        name = _failure_concern(self.source, log=failure, relative=self.relative)
        if not name:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_COMPILE_UNLOCALIZED: failure is outside every active concern region.\n"
                + failure
            )
        self._apply(self._concern(name), failure=failure)
        self.repairs += 1
        next_report = self._compile()
        self._assert_improving(next_report)
        return next_report

    def _assert_improving(self, report: Any) -> None:
        if getattr(report, "status", "") == "PASS":
            return
        failure = self.compile_log(report) or str(getattr(report, "error", "") or "")
        measure = _failure_measure(failure)
        if measure >= self.best_measure:
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_REPAIR_NO_PROGRESS: compiler error measure did not strictly decrease.\n"
                + failure
            )
        self.best_measure = measure

    def run(self) -> dict[str, Any]:
        for concern in self.ordered:
            self._apply(concern)
        report = self._compile()
        if getattr(report, "status", "") != "PASS":
            self.best_measure = _failure_measure(self.compile_log(report))
        while getattr(report, "status", "") != "PASS":
            report = self._repair_once(report)
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
