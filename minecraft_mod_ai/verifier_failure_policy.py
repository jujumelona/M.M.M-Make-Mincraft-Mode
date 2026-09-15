from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable


class FailureKind(str, Enum):
    SOURCE_DEFECT = "SOURCE_DEFECT"
    TOOLCHAIN_ENVIRONMENT = "TOOLCHAIN_ENVIRONMENT"
    TRANSIENT_INFRASTRUCTURE = "TRANSIENT_INFRASTRUCTURE"
    UNKNOWN = "UNKNOWN"


_ENV_PATTERNS = (
    re.compile(r"\brelease\s+\d+\s+is\s+not\s+found\s+in\s+the\s+system\b", re.I),
    re.compile(r"\binvalid\s+source\s+release\b", re.I),
    re.compile(r"\brelease\s+version\s+\d+\s+not\s+supported\b", re.I),
    re.compile(r"\b(?:jdk|jre|jdt|javac|java|compiler)\b.*\b(?:missing|unavailable|not found|unsupported)\b", re.I),
    re.compile(r"\btoolchain\b.*\b(?:mismatch|missing|unavailable|unsupported)\b", re.I),
)

_TRANSIENT_PATTERNS = (
    re.compile(r"\b(?:timeout|timed out|connection reset|temporarily unavailable|service unavailable)\b", re.I),
    re.compile(r"\b(?:502|503|504)\b"),
)

_SOURCE_PATTERNS = (
    re.compile(r"\bcannot find symbol\b", re.I),
    re.compile(r"\bincompatible types\b", re.I),
    re.compile(r"\b(?:';' expected|syntax error|not a statement|illegal start of expression)\b", re.I),
)


@dataclass(frozen=True)
class FailureClassification:
    kind: FailureKind
    diagnostic: str


def classify_verifier_failure(text: str | None) -> FailureKind:
    diagnostic = text or ""
    if any(pattern.search(diagnostic) for pattern in _ENV_PATTERNS):
        return FailureKind.TOOLCHAIN_ENVIRONMENT
    if any(pattern.search(diagnostic) for pattern in _TRANSIENT_PATTERNS):
        return FailureKind.TRANSIENT_INFRASTRUCTURE
    if any(pattern.search(diagnostic) for pattern in _SOURCE_PATTERNS):
        return FailureKind.SOURCE_DEFECT
    return FailureKind.UNKNOWN


def classify_failure(text: str | None) -> FailureClassification:
    diagnostic = text or ""
    return FailureClassification(classify_verifier_failure(diagnostic), diagnostic)


class Capability(str, Enum):
    READ_SOURCE = "READ_SOURCE"
    EDIT_EXISTING_SOURCE = "EDIT_EXISTING_SOURCE"
    CREATE_SOURCE = "CREATE_SOURCE"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    capabilities: frozenset[Capability]
    tools: frozenset[str]

    def validate(self) -> "PhaseSpec":
        if Capability.EDIT_EXISTING_SOURCE in self.capabilities:
            missing = {"read_file", "apply_source_edit"} - self.tools
            if missing:
                raise RuntimeError(
                    f"{self.name} phase contract missing required edit tools: {sorted(missing)}"
                )
        if Capability.READ_SOURCE in self.capabilities and "read_file" not in self.tools:
            raise RuntimeError(f"{self.name} phase contract missing read_file")
        return self


RECOVER_PHASE = PhaseSpec(
    name="RECOVER",
    capabilities=frozenset(
        {
            Capability.READ_SOURCE,
            Capability.EDIT_EXISTING_SOURCE,
            Capability.VERIFY,
        }
    ),
    tools=frozenset({"read_file", "apply_source_edit", "verify"}),
).validate()


def source_repair_allowed(kind: FailureKind) -> bool:
    return kind is FailureKind.SOURCE_DEFECT


def stable_hash(value: str | bytes | None) -> str:
    if value is None:
        data = b""
    elif isinstance(value, bytes):
        data = value
    else:
        data = value.encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProgressFingerprint:
    failure_kind: FailureKind
    diagnostic_code: str
    phase: str
    source_hash: str
    artifact_hash: str
    toolchain_hash: str


def make_progress_fingerprint(
    *,
    failure_kind: FailureKind,
    diagnostic_code: str,
    phase: str,
    source_state: str | bytes | None,
    artifact_state: str | bytes | None,
    toolchain_state: str | bytes | None,
) -> ProgressFingerprint:
    return ProgressFingerprint(
        failure_kind=failure_kind,
        diagnostic_code=(diagnostic_code or "").strip().lower(),
        phase=phase,
        source_hash=stable_hash(source_state),
        artifact_hash=stable_hash(artifact_state),
        toolchain_hash=stable_hash(toolchain_state),
    )


def is_no_progress(
    previous: ProgressFingerprint | None,
    current: ProgressFingerprint,
    *,
    material_change: bool,
) -> bool:
    if previous is None or material_change:
        return False
    if current.failure_kind is FailureKind.TRANSIENT_INFRASTRUCTURE:
        return False
    return previous == current


@dataclass(frozen=True)
class JavaPreflightResult:
    ok: bool
    failure_kind: FailureKind | None
    target_release: int
    supported_releases: tuple[int, ...]
    diagnostic: str | None = None


def check_java_target(
    target_release: int,
    supported_releases: Iterable[int],
) -> JavaPreflightResult:
    supported = tuple(sorted({int(release) for release in supported_releases}))
    if int(target_release) in supported:
        return JavaPreflightResult(
            ok=True,
            failure_kind=None,
            target_release=int(target_release),
            supported_releases=supported,
        )
    return JavaPreflightResult(
        ok=False,
        failure_kind=FailureKind.TOOLCHAIN_ENVIRONMENT,
        target_release=int(target_release),
        supported_releases=supported,
        diagnostic=(
            f"requested Java release {int(target_release)} is unavailable; "
            f"supported releases: {supported or 'none'}"
        ),
    )
