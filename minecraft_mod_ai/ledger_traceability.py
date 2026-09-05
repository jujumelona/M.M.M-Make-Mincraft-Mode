from __future__ import annotations

"""Machine-readable traceability for the MMM master requirements ledger.

The Markdown ledger remains product truth. This module mirrors only the routing
metadata that must be executable: family ownership, regression routing, and
acceptance evidence producers. ``audit_ledger_text`` compares a supplied ledger
snapshot with these manifests and fails closed on omissions.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class OwnerRoute:
    primary_owner: str
    collaborators: str


@dataclass(frozen=True)
class RegressionRoute:
    violated_family: str
    route: str


@dataclass(frozen=True)
class AcceptanceRoute:
    evidence_producer: str


@dataclass(frozen=True)
class LedgerTraceAudit:
    requirement_count: int
    family_count: int
    regression_count: int
    acceptance_count: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


_OWNER_GROUPS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("REQ-GOV", "REQ-PROD"), "Orchestrator / ledger governance", "all layers, acceptance"),
    (("REQ-REPO",), "Project/Repository integration", "orchestrator, transaction"),
    (("REQ-MODEL", "REQ-MADP", "REQ-PARSE"), "EffectiveModelProfile + Model Adapter", "benchmark, prompt/context"),
    (("REQ-PROMPT", "REQ-TOOL"), "Context/Tool Surface Builder", "Model Adapter, host policy"),
    (("REQ-HOST", "REQ-ACT"), "Host Action Runtime", "authorization, model adapter"),
    (("REQ-SLM",), "Host Runtime + Benchmark", "context, edit, scheduler"),
    (("REQ-FABRIC", "REQ-FADP", "REQ-TARGET"), "Minecraft/Fabric Domain + FabricTargetAdapter", "evidence, verifier"),
    (("REQ-GRAPH", "REQ-ATYPE"), "Artifact Graph", "target adapter, linker"),
    (("REQ-PLAN", "REQ-SEM", "REQ-SCALE", "REQ-DESIGN"), "Requirement/Design Graph", "evidence, artifact compiler"),
    (("REQ-BIND", "REQ-LINK", "REQ-TASK", "REQ-DATA"), "Plan Compiler / Collect-All Linker", "artifact graph, executor"),
    (("REQ-REUSE", "REQ-REF", "REQ-EVID"), "Evidence/Reuse Authority", "retrieval, license/security"),
    (("REQ-RAG",), "Retrieval/Context", "evidence cache, model adapter"),
    (("REQ-TPL",), "Target Adapter / deterministic generator", "artifact graph, verifier"),
    (("REQ-EDIT",), "Mutation Engine", "host authorization, transaction"),
    (("REQ-VERIFY", "REQ-VMAP"), "Verification Engine", "target adapter, artifact graph"),
    (("REQ-CAUSE", "REQ-REPAIR"), "Failure/Repair Engine", "verifier, retrieval, model adapter"),
    (("REQ-LOG",), "Durable Journal / Blob Store", "every side-effect boundary"),
    (("REQ-TXN", "REQ-FSM", "REQ-EXEC"), "Transaction/Recovery + Orchestrator", "journal, mutation, verifier"),
    (("REQ-TIME",), "Host Runtime", "model/tool/verifier adapters"),
    (("REQ-PERF", "REQ-SCHED"), "Scheduler/Autotune", "conflict graph, benchmark"),
    (("REQ-ASSET",), "Asset Capability Router", "target adapter, verifier"),
    (("REQ-NET", "REQ-PERSIST"), "Fabric Domain / Target Adapter", "artifact graph, verifier"),
    (("REQ-BENCH", "REQ-BCORP"), "Benchmark/Decision Engine", "all measured layers"),
    (("REQ-LORA",), "EffectiveModelProfile", "benchmark; public adapters only"),
    (("REQ-CLEAN", "REQ-QUALITY"), "Architecture/Repository audit", "all production paths"),
    (("REQ-CACHE",), "Cache/Dependency Index", "project snapshot, evidence, verifier"),
    (("REQ-CONFIG",), "Run Configuration Resolver", "target/model profiles"),
    (("REQ-SEC",), "Host Policy / Evidence Authority", "tool runtime, retrieval, mutation"),
    (("REQ-PROJECT",), "Project Discovery", "target resolver, artifact graph"),
    (("REQ-NAME",), "Identity/Artifact Graph", "linker, mutation"),
    (("REQ-PACK",), "Packaging/Global Verification", "Gradle/Loader, evidence manifest"),
    (("REQ-UI",), "Progress UI adapter", "durable journal only"),
    (("REQ-FAULT",), "Fault-Injection Harness", "real production adapters"),
    (("REQ-ARCH",), "Architecture boundary governance", "all canonical owners"),
    (("REQ-OBS",), "Evidence/Observation registry", "journal, target/model profiles"),
    (("REQ-TRACE",), "Regression/Acceptance/Decision trace registry", "benchmark, verifier, fault harness"),
    (("REQ-RESEARCH",), "Research/Epistemic governance", "evidence authority, target/model profiles"),
    (("REQ-OWNER",), "Ledger self-audit / architecture boundary governance", "all canonical owners, trace registry"),
)

FAMILY_OWNERS: dict[str, OwnerRoute] = {
    family: OwnerRoute(owner, collaborators)
    for families, owner, collaborators in _OWNER_GROUPS
    for family in families
}

_REGRESSION_ROWS: tuple[tuple[str, str, str], ...] = (
    ("REG-001", "LOG", "inject logger failure while a primary exception exists; assert first-cause identity unchanged"),
    ("REG-002", "BIND/LINK", "malformed task with no production binding; collect-all preflight must reject"),
    ("REG-003", "BIND/VMAP", "runtime capability bound only to test artifact; linker rejects"),
    ("REG-004", "BIND/ACT/EDIT", "model requests unowned target; authorization rejects before write"),
    ("REG-005", "REUSE/TOOL", "fresh/no-donor task; reuse tool absent from exposed surface"),
    ("REG-006", "ATYPE/BIND", "abstract locator supplied as path; type validator rejects"),
    ("REG-007", "BIND", "task requires source creation but writable set lacks slot; preflight rejects"),
    ("REG-008", "LINK", "multi-defect plan; all deterministic defects returned before coder call"),
    ("REG-009", "LOG/UI", "truncate/kill console; durable journal/blob still contains full tail"),
    ("REG-010", "TIME/LOG", "production-size model timeout; classify and persist complete request state"),
    ("REG-011", "REPAIR", "identical normalized failure repeated; finite repeat detector terminates/escalates"),
    ("REG-012", "FABRIC/NET/VMAP", "client class reachable from common/server; source-set/launch verifier fails"),
    ("REG-013", "GRAPH/VMAP", "registered object missing resource/data counterpart; linker/verifier fails"),
    ("REG-014", "VERIFY", "compile succeeds but runtime scenario fails; task cannot be completed"),
    ("REG-015", "PARSE", "recoverable multiple/fenced/prose JSON candidates; normalizer recovers boundedly"),
    ("REG-016", "PARSE/MADP", "empty content/no tool call; finite protocol failure path"),
    ("REG-017", "TOOL/PROMPT/BENCH", "increase schema/context size; detect measured tool-call degradation"),
    ("REG-018", "RAG", "prompt-shaped full-sentence query fixture; query planner must decompose/repair"),
    ("REG-019", "RAG/CACHE", "unchanged snapshot repeated; index/cache reuse asserted"),
    ("REG-020", "ASSET", "advertise unavailable media capability; capability registry refuses verified status"),
    ("REG-021", "SEM", "Space Mode prerequisite graph; zero-edge graph rejected"),
    ("REG-022", "DESIGN", "invented numeric defaults; provenance gate prevents user-requirement promotion"),
    ("REG-023", "MADP/BENCH", "tiny tool probe passes but long-context tool case fails; capability not generalized"),
    ("REG-024", "MADP/ARCH", "raw Qwen/llama tags outside adapter module; architecture/dependency test rejects"),
    ("REG-025", "SCALE/PLAN", "chunked long proposal; source-span coverage must remain 100% explicit"),
    ("REG-026", "TARGET", "mutate target assumption after freeze; hash/invariant rejects drift"),
    ("REG-027", "VERIFY", "metadata says PASS without receipt; completion gate rejects"),
    ("REG-028", "TARGET/FADP", "26.2 target with legacy mapping assumption; semantic target validation rejects/normalizes"),
    ("REG-029", "FADP/QUALITY", "duplicated version conditionals outside target adapter; architecture lint/audit fails"),
    ("REG-030", "REF/FADP", "incompatible naming/version example; evidence compatibility gate rejects or translates"),
    ("REG-031", "CACHE/VERIFY", "change verifier/source/target input; stale receipt cannot be reused"),
    ("REG-032", "PROJECT", "valid existing Fabric project; skeleton generator must not overwrite"),
    ("REG-033", "ACT/BIND", "arbitrary path string not backed by target ID; mutation rejected"),
    ("REG-034", "PACK/VERIFY", "source/build tree looks valid but packaged JAR misses artifact; package gate fails"),
    ("REG-035", "QUALITY/ARCH", "new monolithic orchestration path; dependency/complexity audit fails"),
    ("REG-036", "SEC", "adversarial retrieved instruction; host authority unchanged"),
    ("REG-037", "DATA/CLEAN", "contradictory legacy task/capsule states; canonical-boundary test rejects divergence"),
    ("REG-SEM-001", "SEM", "original multi-stage zero-dependency fixture; semantic graph validator rejects"),
    ("REG-DESIGN-001", "DESIGN", "original invented-default fixture; provenance audit rejects"),
)

REGRESSION_MANIFEST: dict[str, RegressionRoute] = {
    regression_id: RegressionRoute(family, route)
    for regression_id, family, route in _REGRESSION_ROWS
}

_ACCEPTANCE_RANGES: tuple[tuple[int, int, str], ...] = (
    (1, 7, "requirement coverage report + artifact graph/linker preflight receipts"),
    (8, 13, "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    (14, 19, "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    (20, 27, "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    (28, 33, "WAL/transaction/restart fault-injection receipts"),
    (34, 36, "repository dependency/reference/dead-path cleanup audit"),
    (37, 40, "repeated production E2E + both-model + fault suite report"),
    (41, 50, "semantic/model/target/scheduler architecture regression suite"),
    (51, 60, "target adapter, cache, project, security, packaging, quality and decision receipts"),
    (61, 64, "ledger owner/traceability/epistemic self-audit report"),
)

ACCEPTANCE_MANIFEST: dict[str, AcceptanceRoute] = {
    f"ACC-{value:03d}": AcceptanceRoute(producer)
    for start, end, producer in _ACCEPTANCE_RANGES
    for value in range(start, end + 1)
}

_REQUIREMENT_DEF_RE = re.compile(
    r"^\*\*(REQ-[A-Z0-9]+-[A-Z0-9]+)\s+—\s+"
    r"(FIXED|FORBIDDEN|MUST-TEST(?:\s*/[^*]+)?|OPEN|ACCEPTANCE)\*\*(?:[^\n]*)$",
    re.MULTILINE,
)
_REGRESSION_TOKEN_RE = re.compile(r"REG-(?:\d{3}|[A-Z]+-\d{3})")
_ACCEPTANCE_TOKEN_RE = re.compile(r"ACC-\d{3}")


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: set[str] = set()
    for value in values:
        if value in seen:
            result.add(value)
        seen.add(value)
    return tuple(sorted(result))


def _ledger_owner_families(text: str) -> set[str]:
    heading = "## 36.25 Requirement-family ownership matrix"
    end_heading = "## 36.26 Regression, acceptance, and decision traceability"
    start = text.find(heading)
    end = text.find(end_heading, start + len(heading)) if start >= 0 else -1
    block = text[start : end if end >= 0 else None] if start >= 0 else ""
    return set(re.findall(r"`(REQ-[A-Z0-9]+)`", block))


def audit_ledger_text(text: str) -> LedgerTraceAudit:
    requirement_matches = tuple(_REQUIREMENT_DEF_RE.finditer(text))
    requirement_ids = tuple(match.group(1) for match in requirement_matches)
    families = {requirement_id.rsplit("-", 1)[0] for requirement_id in requirement_ids}
    active_regressions = set(_REGRESSION_TOKEN_RE.findall(text))
    active_acceptances = set(_ACCEPTANCE_TOKEN_RE.findall(text))
    issues: list[str] = []

    duplicates = _duplicates(requirement_ids)
    if duplicates:
        issues.append("duplicate requirement IDs: " + ", ".join(duplicates))

    missing_code_owners = sorted(families - set(FAMILY_OWNERS))
    if missing_code_owners:
        issues.append("requirement families missing architectural owner: " + ", ".join(missing_code_owners))

    ledger_owner_families = _ledger_owner_families(text)
    missing_ledger_owners = sorted(families - ledger_owner_families)
    if missing_ledger_owners:
        issues.append("requirement families missing ownership-matrix row: " + ", ".join(missing_ledger_owners))

    missing_regressions = sorted(active_regressions - set(REGRESSION_MANIFEST))
    if missing_regressions:
        issues.append("regressions missing executable route: " + ", ".join(missing_regressions))

    missing_acceptances = sorted(
        active_acceptances - set(ACCEPTANCE_MANIFEST),
        key=lambda value: int(value.split("-", 1)[1]),
    )
    if missing_acceptances:
        issues.append("acceptance gates missing evidence producer: " + ", ".join(missing_acceptances))

    return LedgerTraceAudit(
        requirement_count=len(requirement_ids),
        family_count=len(families),
        regression_count=len(active_regressions),
        acceptance_count=len(active_acceptances),
        issues=tuple(issues),
    )


def audit_ledger_file(path: str | Path) -> LedgerTraceAudit:
    return audit_ledger_text(Path(path).read_text(encoding="utf-8"))


def validate_manifest_snapshot() -> None:
    if len(FAMILY_OWNERS) != 62:
        raise RuntimeError(f"LEDGER_OWNER_MANIFEST_COUNT: expected 62, got {len(FAMILY_OWNERS)}")
    if len(REGRESSION_MANIFEST) != 39:
        raise RuntimeError(f"LEDGER_REGRESSION_MANIFEST_COUNT: expected 39, got {len(REGRESSION_MANIFEST)}")
    if set(ACCEPTANCE_MANIFEST) != {f"ACC-{value:03d}" for value in range(1, 65)}:
        raise RuntimeError("LEDGER_ACCEPTANCE_MANIFEST_COVERAGE: ACC-001..ACC-064 must all be routed")


validate_manifest_snapshot()


__all__ = [
    "ACCEPTANCE_MANIFEST",
    "FAMILY_OWNERS",
    "REGRESSION_MANIFEST",
    "AcceptanceRoute",
    "LedgerTraceAudit",
    "OwnerRoute",
    "RegressionRoute",
    "audit_ledger_file",
    "audit_ledger_text",
    "validate_manifest_snapshot",
]
