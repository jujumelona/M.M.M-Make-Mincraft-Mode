from __future__ import annotations

"""Machine-readable traceability for the MMM master requirements ledger.

The Markdown ledger remains product truth. This module mirrors executable routing
metadata only. A planned route is never treated as PASS evidence.
"""

from dataclasses import dataclass
import hashlib
import json
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
    triggering_fixture_or_log: str
    violated_requirements: tuple[str, ...]
    test_case: str
    expected_first_cause_class: str
    required_acceptance: tuple[str, ...]
    execution_status: str = "planned"

    @property
    def route(self) -> str:
        return self.test_case


@dataclass(frozen=True)
class AcceptanceRoute:
    evidence_producer: str
    owning_requirements: tuple[str, ...]
    verifier_case: str
    receipt_type: str
    pass_predicate: str
    applicability: str


@dataclass(frozen=True)
class DecisionReceipt:
    requirement_id: str
    alternatives: tuple[str, ...]
    model_identity: str
    target_identity: str
    runtime_identity: str
    corpus_version: str
    repeated_trial_results: tuple[str, ...]
    failure_taxonomy: tuple[str, ...]
    selected_option: str
    rejection_reasons: tuple[str, ...]
    expiration_or_retest_triggers: tuple[str, ...]

    def validate(self) -> None:
        if not self.requirement_id.startswith("REQ-"):
            raise ValueError("DECISION_RECEIPT_REQUIREMENT_ID")
        if len(self.alternatives) < 2 or len(set(self.alternatives)) != len(self.alternatives):
            raise ValueError("DECISION_RECEIPT_ALTERNATIVES")
        if self.selected_option not in self.alternatives:
            raise ValueError("DECISION_RECEIPT_SELECTION")
        if not self.model_identity or not self.target_identity or not self.runtime_identity:
            raise ValueError("DECISION_RECEIPT_EXACT_IDENTITY")
        if not self.corpus_version:
            raise ValueError("DECISION_RECEIPT_CORPUS")
        if len(self.repeated_trial_results) < 2:
            raise ValueError("DECISION_RECEIPT_REPEATED_TRIALS")
        if not self.failure_taxonomy:
            raise ValueError("DECISION_RECEIPT_FAILURE_TAXONOMY")
        if len(self.rejection_reasons) < len(self.alternatives) - 1:
            raise ValueError("DECISION_RECEIPT_REJECTION_REASONS")
        if not self.expiration_or_retest_triggers:
            raise ValueError("DECISION_RECEIPT_RETEST_TRIGGERS")

    def content_sha256(self) -> str:
        self.validate()
        payload = {
            "requirement_id": self.requirement_id,
            "alternatives": self.alternatives,
            "model_identity": self.model_identity,
            "target_identity": self.target_identity,
            "runtime_identity": self.runtime_identity,
            "corpus_version": self.corpus_version,
            "repeated_trial_results": self.repeated_trial_results,
            "failure_taxonomy": self.failure_taxonomy,
            "selected_option": self.selected_option,
            "rejection_reasons": self.rejection_reasons,
            "expiration_or_retest_triggers": self.expiration_or_retest_triggers,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()


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

_REGRESSION_ROWS = (
    ("REG-001", "LOG", "inject logger failure while a primary exception exists; assert first-cause identity unchanged", ("REQ-LOG-004",), "original_primary_failure", ("ACC-029",)),
    ("REG-002", "BIND/LINK", "malformed task with no production binding; collect-all preflight must reject", ("REQ-BIND-001", "REQ-LINK-006"), "missing_production_binding", ("ACC-006",)),
    ("REG-003", "BIND/VMAP", "runtime capability bound only to test artifact; linker rejects", ("REQ-BIND-002", "REQ-LINK-007", "REQ-VMAP-005"), "test_only_runtime_binding", ("ACC-005",)),
    ("REG-004", "BIND/ACT/EDIT", "model requests unowned target; authorization rejects before write", ("REQ-BIND-005", "REQ-ACT-001", "REQ-EDIT-001"), "unauthorized_mutation_target", ("ACC-006", "ACC-056")),
    ("REG-005", "REUSE/TOOL", "fresh/no-donor task; reuse tool absent from exposed surface", ("REQ-REUSE-001", "REQ-TOOL-003"), "illegal_tool_exposure", ("ACC-009", "ACC-015")),
    ("REG-006", "ATYPE/BIND", "abstract locator supplied as path; type validator rejects", ("REQ-ATYPE-004", "REQ-BIND-007", "REQ-BIND-008"), "abstract_locator_type_error", ("ACC-007",)),
    ("REG-007", "BIND", "task requires source creation but writable set lacks slot; preflight rejects", ("REQ-BIND-011",), "missing_creation_slot", ("ACC-006",)),
    ("REG-008", "LINK", "multi-defect plan; all deterministic defects returned before coder call", ("REQ-LINK-002", "REQ-LINK-034"), "collect_all_preflight_defects", ("ACC-004",)),
    ("REG-009", "LOG/UI", "truncate/kill console; durable journal/blob still contains full tail", ("REQ-LOG-001", "REQ-UI-005"), "durable_log_unavailable", ("ACC-028", "ACC-029")),
    ("REG-010", "TIME/LOG", "production-size model timeout; classify and persist complete request state", ("REQ-TIME-007", "REQ-LOG-001"), "model_timeout", ("ACC-013", "ACC-019")),
    ("REG-011", "REPAIR", "identical normalized failure repeated; finite repeat detector terminates/escalates", ("REQ-REPAIR-008", "REQ-PARSE-004"), "repeated_normalized_failure", ("ACC-013", "ACC-019")),
    ("REG-012", "FABRIC/NET/VMAP", "client class reachable from common/server; source-set/launch verifier fails", ("REQ-FABRIC-026", "REQ-FABRIC-029", "REQ-GRAPH-020"), "illegal_client_server_dependency", ("ACC-023",)),
    ("REG-013", "GRAPH/VMAP", "registered object missing resource/data counterpart; linker/verifier fails", ("REQ-GRAPH-002", "REQ-GRAPH-003", "REQ-LINK-019"), "missing_resource_or_data_prerequisite", ("ACC-003", "ACC-021")),
    ("REG-014", "VERIFY", "compile succeeds but runtime scenario fails; task cannot be completed", ("REQ-VERIFY-013", "REQ-PROD-004"), "runtime_semantic_failure", ("ACC-024",)),
    ("REG-015", "PARSE", "recoverable multiple/fenced/prose JSON candidates; normalizer recovers boundedly", ("REQ-PARSE-007", "REQ-PARSE-009"), "recoverable_model_format_defect", ("ACC-009", "ACC-015")),
    ("REG-016", "PARSE/MADP", "empty content/no tool call; finite protocol failure path", ("REQ-PARSE-004", "REQ-MADP-003"), "empty_model_action", ("ACC-009", "ACC-015")),
    ("REG-017", "TOOL/PROMPT/BENCH", "increase schema/context size; detect measured tool-call degradation", ("REQ-TOOL-005", "REQ-PROMPT-007"), "tool_call_degradation", ("ACC-010", "ACC-016", "ACC-045")),
    ("REG-018", "RAG", "prompt-shaped full-sentence query fixture; query planner must decompose/repair", ("REQ-RAG-005", "REQ-RAG-006"), "retrieval_query_planning_defect", ("ACC-037",)),
    ("REG-019", "RAG/CACHE", "unchanged snapshot repeated; index/cache reuse asserted", ("REQ-RAG-007", "REQ-CACHE-001"), "cache_reuse_defect", ("ACC-054",)),
    ("REG-020", "ASSET", "advertise unavailable media capability; capability registry refuses verified status", ("REQ-ASSET-005", "REQ-ASSET-011"), "unavailable_media_capability", ("ACC-021",)),
    ("REG-021", "SEM", "Space Mode prerequisite graph; zero-edge graph rejected", ("REQ-SEM-010",), "missing_semantic_prerequisite_edge", ("ACC-041",)),
    ("REG-022", "DESIGN", "invented numeric defaults; provenance gate prevents user-requirement promotion", ("REQ-DESIGN-003", "REQ-DESIGN-009"), "invented_design_constant", ("ACC-048",)),
    ("REG-023", "MADP/BENCH", "tiny tool probe passes but long-context tool case fails; capability not generalized", ("REQ-MADP-004", "REQ-MODEL-046"), "long_context_capability_mismatch", ("ACC-043", "ACC-044", "ACC-045")),
    ("REG-024", "MADP/ARCH", "raw Qwen/llama tags outside adapter module; architecture/dependency test rejects", ("REQ-MADP-011", "REQ-MADP-012"), "model_syntax_boundary_leak", ("ACC-046",)),
    ("REG-025", "SCALE/PLAN", "chunked long proposal; source-span coverage must remain 100% explicit", ("REQ-SCALE-003", "REQ-SCALE-012"), "proposal_coverage_loss", ("ACC-001", "ACC-042")),
    ("REG-026", "TARGET", "mutate target assumption after freeze; hash/invariant rejects drift", ("REQ-TARGET-001", "REQ-TARGET-005", "REQ-TARGET-006"), "target_profile_drift", ("ACC-047",)),
    ("REG-027", "VERIFY", "metadata says PASS without receipt; completion gate rejects", ("REQ-HOST-005", "REQ-VERIFY-001"), "unverified_pass_claim", ("ACC-053",)),
    ("REG-028", "TARGET/FADP", "26.2 target with legacy mapping assumption; semantic target validation rejects/normalizes", ("REQ-TARGET-014", "REQ-FADP-009"), "invalid_target_naming_regime", ("ACC-051",)),
    ("REG-029", "FADP/QUALITY", "duplicated version conditionals outside target adapter; architecture lint/audit fails", ("REQ-FADP-003", "REQ-QUALITY-006"), "target_logic_boundary_leak", ("ACC-051", "ACC-059")),
    ("REG-030", "REF/FADP", "incompatible naming/version example; evidence compatibility gate rejects or translates", ("REQ-REF-007", "REQ-TARGET-012"), "incompatible_reference_evidence", ("ACC-052",)),
    ("REG-031", "CACHE/VERIFY", "change verifier/source/target input; stale receipt cannot be reused", ("REQ-CACHE-005", "REQ-CACHE-010"), "stale_verifier_receipt", ("ACC-054",)),
    ("REG-032", "PROJECT", "valid existing Fabric project; skeleton generator must not overwrite", ("REQ-PROJECT-004", "REQ-PROJECT-008"), "existing_project_overwrite", ("ACC-055",)),
    ("REG-033", "ACT/BIND", "arbitrary path string not backed by target ID; mutation rejected", ("REQ-ACT-001", "REQ-ACT-007"), "path_text_authority_bypass", ("ACC-056",)),
    ("REG-034", "PACK/VERIFY", "source/build tree looks valid but packaged JAR misses artifact; package gate fails", ("REQ-PACK-002", "REQ-PACK-005"), "packaged_artifact_missing_required_content", ("ACC-058",)),
    ("REG-035", "QUALITY/ARCH", "new monolithic orchestration path; dependency/complexity audit fails", ("REQ-QUALITY-001", "REQ-ARCH-013"), "architecture_boundary_cycle_or_god_path", ("ACC-059",)),
    ("REG-036", "SEC", "adversarial retrieved instruction; host authority unchanged", ("REQ-SEC-003", "REQ-SEC-008"), "retrieved_instruction_authority_escalation", ("ACC-057",)),
    ("REG-037", "DATA/CLEAN", "contradictory legacy task/capsule states; canonical-boundary test rejects divergence", ("REQ-DATA-001", "REQ-DATA-002", "REQ-CLEAN-001"), "competing_mutable_core_state", ("ACC-036", "ACC-049")),
    ("REG-SEM-001", "SEM", "original multi-stage zero-dependency fixture; semantic graph validator rejects", ("REQ-SEM-010",), "missing_semantic_prerequisite_edge", ("ACC-041",)),
    ("REG-DESIGN-001", "DESIGN", "original invented-default fixture; provenance audit rejects", ("REQ-DESIGN-009",), "invented_design_constant", ("ACC-048",)),
)

REGRESSION_MANIFEST: dict[str, RegressionRoute] = {
    regression_id: RegressionRoute(
        violated_family=family,
        triggering_fixture_or_log=f"ledger:{regression_id}",
        violated_requirements=requirements,
        test_case=test_case,
        expected_first_cause_class=first_cause,
        required_acceptance=acceptance,
    )
    for regression_id, family, test_case, requirements, first_cause, acceptance
    in _REGRESSION_ROWS
}

_ACCEPTANCE_ROWS = (
    ("ACC-001", ("REQ-SCALE-012", "REQ-PLAN-006"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "long proposal requirements retained", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-002", ("REQ-PLAN-004", "REQ-TRACE-002"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "traceability complete", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-003", ("REQ-PLAN-009", "REQ-GRAPH-020"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "artifact graph coherent", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-004", ("REQ-LINK-034",), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "40+ task plan collect-all validation works", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-005", ("REQ-BIND-002", "REQ-VMAP-005"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "no test-only runtime provider", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-006", ("REQ-BIND-005", "REQ-BIND-006"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "no unresolved ownership", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-007", ("REQ-BIND-007", "REQ-BIND-008", "REQ-ATYPE-004"), "planner_domain_linker_acceptance", "coverage_or_linker_receipt", "no abstract locator/path confusion", "always", "requirement coverage report + artifact graph/linker preflight receipts"),
    ("ACC-008", ("REQ-MADP-001", "REQ-MADP-002"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "actual model profile identified", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-009", ("REQ-TOOL-006", "REQ-MADP-003"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "tool protocol regression passes", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-010", ("REQ-MODEL-046", "REQ-MADP-004"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "long-context production tasks pass repeatedly", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-011", ("REQ-SLM-005", "REQ-EDIT-008"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "edit protocol passes representative tasks", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-012", ("REQ-MODEL-041", "REQ-REPAIR-002"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "compile-repair loop converges", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-013", ("REQ-PARSE-004", "REQ-REPAIR-008"), "qwen35_profile_and_benchmark", "model_probe_benchmark_receipt", "no runaway repeat loop", "model=qwen3.5", "Qwen3.5 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-014", ("REQ-MADP-001", "REQ-MADP-002"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "actual model profile identified", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-015", ("REQ-TOOL-007", "REQ-MADP-003"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "tool protocol regression passes", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-016", ("REQ-MODEL-046", "REQ-MADP-004"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "long-context production tasks pass repeatedly", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-017", ("REQ-SLM-005", "REQ-EDIT-008"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "edit protocol passes representative tasks", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-018", ("REQ-MODEL-041", "REQ-REPAIR-002"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "compile-repair loop converges", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-019", ("REQ-PARSE-004", "REQ-REPAIR-008"), "qwen38_profile_and_benchmark", "model_probe_benchmark_receipt", "no runaway repeat loop", "model=qwen3.8", "Qwen3.8 EffectiveModelProfile probe + held-out MMM benchmark receipts"),
    ("ACC-020", ("REQ-VERIFY-004", "REQ-PACK-001"), "fabric_acceptance_020", "verifier_receipt", "real Gradle build", "target Fabric production run", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-021", ("REQ-VERIFY-008", "REQ-VERIFY-009"), "fabric_acceptance_021", "verifier_receipt", "resource/data load", "target Fabric production run", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-022", ("REQ-VERIFY-011",), "fabric_acceptance_022", "verifier_receipt", "client smoke where required", "when client behavior is required", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-023", ("REQ-VERIFY-012",), "fabric_acceptance_023", "verifier_receipt", "dedicated server smoke where required", "when dedicated-server safety is applicable", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-024", ("REQ-VERIFY-013", "REQ-VERIFY-014"), "fabric_acceptance_024", "verifier_receipt", "GameTests/runtime scenarios for major mechanics", "when proposal contains major mechanics", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-025", ("REQ-VERIFY-015", "REQ-PERSIST-001"), "fabric_acceptance_025", "verifier_receipt", "persistence reload tests", "when persistent state exists", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-026", ("REQ-VERIFY-016", "REQ-NET-001"), "fabric_acceptance_026", "verifier_receipt", "networking tests where present", "when networking exists", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-027", ("REQ-FABRIC-071", "REQ-FABRIC-072"), "fabric_acceptance_027", "verifier_receipt", "worldgen/dimension load where present", "when worldgen/dimensions exist", "target-specific Gradle/resource/client/server/GameTest/runtime receipts"),
    ("ACC-028", ("REQ-TXN-001", "REQ-LOG-001"), "durability_fault_028", "fault_injection_receipt", "crash-safe journal", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-029", ("REQ-LOG-004",), "durability_fault_029", "fault_injection_receipt", "first cause never masked", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-030", ("REQ-TXN-006",), "durability_fault_030", "fault_injection_receipt", "process-kill recovery", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-031", ("REQ-EDIT-007", "REQ-TXN-004"), "durability_fault_031", "fault_injection_receipt", "partial mutation rollback/reconcile", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-032", ("REQ-TXN-003",), "durability_fault_032", "fault_injection_receipt", "prior successful tasks preserved", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-033", ("REQ-CACHE-008", "REQ-TXN-005"), "durability_fault_033", "fault_injection_receipt", "no duplicate replay side effect", "always", "WAL/transaction/restart fault-injection receipts"),
    ("ACC-034", ("REQ-CLEAN-001", "REQ-CLEAN-002"), "cleanup_audit_034", "architecture_audit_receipt", "obsolete competing execution paths removed", "always", "repository dependency/reference/dead-path cleanup audit"),
    ("ACC-035", ("REQ-CLEAN-003",), "cleanup_audit_035", "architecture_audit_receipt", "dead code/file deletion justified by reference audit", "always", "repository dependency/reference/dead-path cleanup audit"),
    ("ACC-036", ("REQ-DATA-001", "REQ-DATA-002"), "cleanup_audit_036", "architecture_audit_receipt", "one authoritative state representation per core concept", "always", "repository dependency/reference/dead-path cleanup audit"),
    ("ACC-037", ("REQ-PROD-003", "REQ-PROD-005"), "e2e_acceptance_037", "e2e_or_fault_receipt", "at least one large real proposal succeeds end-to-end", "always", "repeated production E2E + both-model + fault suite report"),
    ("ACC-038", ("REQ-PROD-004", "REQ-BENCH-001"), "e2e_acceptance_038", "e2e_or_fault_receipt", "repeated success, not one lucky run", "always", "repeated production E2E + both-model + fault suite report"),
    ("ACC-039", ("REQ-MODEL-001", "REQ-MODEL-002"), "e2e_acceptance_039", "e2e_or_fault_receipt", "success under both supported model selections on representative workloads", "always", "repeated production E2E + both-model + fault suite report"),
    ("ACC-040", ("REQ-FAULT-020", "REQ-FAULT-021"), "e2e_acceptance_040", "e2e_or_fault_receipt", "fault-injection suite does not violate invariants", "always", "repeated production E2E + both-model + fault suite report"),
    ("ACC-041", ("REQ-SEM-010",), "architecture_regression_041", "regression_suite_receipt", "semantic prerequisite graph detects the Space Mode zero-edge regression", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-042", ("REQ-SCALE-003", "REQ-SCALE-011", "REQ-SCALE-012"), "architecture_regression_042", "regression_suite_receipt", "every original proposal span has explicit coverage status", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-043", ("REQ-MADP-001", "REQ-MADP-002"), "architecture_regression_043", "regression_suite_receipt", "Qwen3.5 effective profile probe is keyed by exact runtime/template/config identity", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-044", ("REQ-MADP-001", "REQ-MADP-002"), "architecture_regression_044", "regression_suite_receipt", "Qwen3.8 effective profile probe is keyed by exact runtime/template/config identity", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-045", ("REQ-MADP-004",), "architecture_regression_045", "regression_suite_receipt", "large-context tool-call probes exist, not only tiny preflight probes", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-046", ("REQ-MADP-011", "REQ-MADP-012"), "architecture_regression_046", "regression_suite_receipt", "model-specific raw syntax cannot escape adapter boundary", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-047", ("REQ-TARGET-001", "REQ-TARGET-005", "REQ-TARGET-006"), "architecture_regression_047", "regression_suite_receipt", "target profile remains immutable across implementation tasks", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-048", ("REQ-DESIGN-009",), "architecture_regression_048", "regression_suite_receipt", "invented gameplay constants cannot silently become explicit user requirements", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-049", ("REQ-DATA-001", "REQ-DATA-002", "REQ-DATA-003"), "architecture_regression_049", "regression_suite_receipt", "canonical task/artifact/verification contracts have no competing writable representations", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-050", ("REQ-SCHED-003", "REQ-SCHED-005"), "architecture_regression_050", "regression_suite_receipt", "conflict scheduler demonstrates deterministic results under safe parallel execution", "always", "semantic/model/target/scheduler architecture regression suite"),
    ("ACC-051", ("REQ-FADP-009", "REQ-TARGET-010", "REQ-TARGET-014"), "contract_acceptance_051", "contract_or_decision_receipt", "Fabric target adapter distinguishes mapped/obfuscated and current unobfuscated naming regimes", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-052", ("REQ-REF-007", "REQ-TARGET-012", "REQ-TARGET-013"), "contract_acceptance_052", "contract_or_decision_receipt", "wrong-version/mapping reference code is rejected or translated with evidence", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-053", ("REQ-VMAP-006",), "contract_acceptance_053", "contract_or_decision_receipt", "artifact-to-verifier matrix produces receipts for every final production artifact", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-054", ("REQ-CACHE-010",), "contract_acceptance_054", "contract_or_decision_receipt", "cache invalidation tests reject stale receipts", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-055", ("REQ-PROJECT-008",), "contract_acceptance_055", "contract_or_decision_receipt", "new/existing/broken project modes preserve the correct project contract", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-056", ("REQ-ACT-007",), "contract_acceptance_056", "contract_or_decision_receipt", "host-issued target IDs, not model path text, control mutation authority", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-057", ("REQ-SEC-008",), "contract_acceptance_057", "contract_or_decision_receipt", "prompt-injection regression cannot expand host authority", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-058", ("REQ-PACK-005",), "contract_acceptance_058", "contract_or_decision_receipt", "final packaged JAR passes inspection/loader smoke", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-059", ("REQ-QUALITY-010",), "contract_acceptance_059", "contract_or_decision_receipt", "code-quality audit finds no new orchestration god module or duplicate mutable core state", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-060", ("REQ-BCORP-008", "REQ-TRACE-003"), "contract_acceptance_060", "contract_or_decision_receipt", "every frozen MUST-TEST decision has a benchmark decision receipt", "always", "target adapter, cache, project, security, packaging, quality and decision receipts"),
    ("ACC-061", ("REQ-OWNER-004",), "ledger_self_audit_061", "ledger_audit_receipt", "every requirement family has an architectural owner", "always", "ledger owner/traceability/epistemic self-audit report"),
    ("ACC-062", ("REQ-TRACE-001", "REQ-TRACE-007"), "ledger_self_audit_062", "ledger_audit_receipt", "every regression has a test/fault route and violated-requirement link", "always", "ledger owner/traceability/epistemic self-audit report"),
    ("ACC-063", ("REQ-TRACE-002", "REQ-TRACE-007"), "ledger_self_audit_063", "ledger_audit_receipt", "every acceptance gate has an evidence-producing route", "always", "ledger owner/traceability/epistemic self-audit report"),
    ("ACC-064", ("REQ-RESEARCH-007",), "ledger_self_audit_064", "ledger_audit_receipt", "architecture review contains no unsupported certainty for OPEN/MUST-TEST items", "always", "ledger owner/traceability/epistemic self-audit report"),
)

ACCEPTANCE_MANIFEST: dict[str, AcceptanceRoute] = {
    acceptance_id: AcceptanceRoute(
        evidence_producer=producer,
        owning_requirements=requirements,
        verifier_case=verifier_case,
        receipt_type=receipt_type,
        pass_predicate=predicate,
        applicability=applicability,
    )
    for (
        acceptance_id,
        requirements,
        verifier_case,
        receipt_type,
        predicate,
        applicability,
        producer,
    ) in _ACCEPTANCE_ROWS
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


def _manifest_structure_issues() -> list[str]:
    issues: list[str] = []
    for regression_id, route in REGRESSION_MANIFEST.items():
        if not route.triggering_fixture_or_log:
            issues.append(f"{regression_id} missing triggering fixture/log link")
        if not route.violated_requirements or any(
            not value.startswith("REQ-") for value in route.violated_requirements
        ):
            issues.append(f"{regression_id} missing violated REQ link")
        if not route.test_case:
            issues.append(f"{regression_id} missing regression/fault case")
        if not route.expected_first_cause_class:
            issues.append(f"{regression_id} missing expected first-cause class")
        if not route.required_acceptance or any(
            not value.startswith("ACC-") for value in route.required_acceptance
        ):
            issues.append(f"{regression_id} missing required ACC evidence")
        if route.execution_status not in {"planned", "executable"}:
            issues.append(f"{regression_id} invalid execution status")

    for acceptance_id, route in ACCEPTANCE_MANIFEST.items():
        if not route.owning_requirements or any(
            not value.startswith("REQ-") for value in route.owning_requirements
        ):
            issues.append(f"{acceptance_id} missing owning requirement")
        if not route.verifier_case:
            issues.append(f"{acceptance_id} missing verifier/fault/benchmark case")
        if not route.receipt_type:
            issues.append(f"{acceptance_id} missing receipt type")
        if not route.pass_predicate:
            issues.append(f"{acceptance_id} missing individual pass predicate")
        if not route.applicability:
            issues.append(f"{acceptance_id} missing target/model applicability")
    return issues


def audit_ledger_text(text: str) -> LedgerTraceAudit:
    requirement_matches = tuple(_REQUIREMENT_DEF_RE.finditer(text))
    requirement_ids = tuple(match.group(1) for match in requirement_matches)
    requirement_id_set = set(requirement_ids)
    families = {requirement_id.rsplit("-", 1)[0] for requirement_id in requirement_ids}
    active_regressions = set(_REGRESSION_TOKEN_RE.findall(text))
    active_acceptances = set(_ACCEPTANCE_TOKEN_RE.findall(text))
    issues = _manifest_structure_issues()

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
        issues.append("regressions missing executable/planned route: " + ", ".join(missing_regressions))

    missing_acceptances = sorted(
        active_acceptances - set(ACCEPTANCE_MANIFEST),
        key=lambda value: int(value.split("-", 1)[1]),
    )
    if missing_acceptances:
        issues.append("acceptance gates missing evidence producer: " + ", ".join(missing_acceptances))

    # Tiny unit-test fixtures contain only a subset. A full ledger has all owner families.
    if set(FAMILY_OWNERS).issubset(families):
        referenced_requirements = {
            requirement
            for route in REGRESSION_MANIFEST.values()
            for requirement in route.violated_requirements
        } | {
            requirement
            for route in ACCEPTANCE_MANIFEST.values()
            for requirement in route.owning_requirements
        }
        stale_requirement_links = sorted(referenced_requirements - requirement_id_set)
        if stale_requirement_links:
            issues.append("manifest links nonexistent requirements: " + ", ".join(stale_requirement_links))
        stale_acceptance_links = sorted(
            {
                acceptance
                for route in REGRESSION_MANIFEST.values()
                for acceptance in route.required_acceptance
            } - active_acceptances
        )
        if stale_acceptance_links:
            issues.append("regression links nonexistent acceptance IDs: " + ", ".join(stale_acceptance_links))

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
    expected_acceptances = {f"ACC-{value:03d}" for value in range(1, 65)}
    if len(FAMILY_OWNERS) != 62:
        raise RuntimeError(f"LEDGER_OWNER_MANIFEST_COUNT: expected 62, got {len(FAMILY_OWNERS)}")
    if len(REGRESSION_MANIFEST) != 39:
        raise RuntimeError(
            f"LEDGER_REGRESSION_MANIFEST_COUNT: expected 39, got {len(REGRESSION_MANIFEST)}"
        )
    if set(ACCEPTANCE_MANIFEST) != expected_acceptances:
        raise RuntimeError("LEDGER_ACCEPTANCE_MANIFEST_COVERAGE: ACC-001..ACC-064 must all be routed")
    issues = _manifest_structure_issues()
    if issues:
        raise RuntimeError("LEDGER_TRACE_MANIFEST_INCOMPLETE: " + "; ".join(issues))


def validate_decision_receipt(receipt: DecisionReceipt) -> str:
    receipt.validate()
    return receipt.content_sha256()


validate_manifest_snapshot()


__all__ = [
    "ACCEPTANCE_MANIFEST",
    "FAMILY_OWNERS",
    "REGRESSION_MANIFEST",
    "AcceptanceRoute",
    "DecisionReceipt",
    "LedgerTraceAudit",
    "OwnerRoute",
    "RegressionRoute",
    "audit_ledger_file",
    "audit_ledger_text",
    "validate_decision_receipt",
    "validate_manifest_snapshot",
]
