"""Provenance and outcome records for behavior-level source reuse.

A search hit is never executable evidence. Verified donors, porting references,
provider failures, and empty searches remain distinguishable in the work plan.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from collections.abc import Mapping


@dataclass(frozen=True)
class SourceReuseEvidence:
    evidence_id: str
    capability: str
    requirement_ids: tuple[str, ...]
    mode: str
    status: str
    repository: str = ""
    commit_sha: str = ""
    license_id: str = ""
    source_id: str = ""
    files: tuple[dict, ...] = ()
    symbols: tuple[str, ...] = ()
    source_units: tuple[dict, ...] = ()
    target_paths: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    initialization: tuple[str, ...] = ()
    state_bindings: tuple[str, ...] = ()
    remaining_behaviors: tuple[str, ...] = ()
    verification: dict = field(default_factory=dict)
    failure_code: str = ""
    failure_detail: str = ""

    def __post_init__(self):
        if self.status not in {
            "verified", "installed", "port_required", "search_empty",
            "provider_error", "download_error", "target_compile_failed",
            "rejected", "unresolved",
            "reference_available",
        }:
            raise ValueError("SOURCE_REUSE_EVIDENCE_STATUS")
        if self.mode not in {"source_transplant", "port", "fresh", "reference"}:
            raise ValueError("SOURCE_REUSE_EVIDENCE_MODE")
        if self.status in {"verified", "installed", "port_required"} and not (
            self.repository and self.commit_sha and self.license_id and self.files
        ):
            raise ValueError("SOURCE_REUSE_EVIDENCE_PROVENANCE_REQUIRED")
        if self.status == "installed" and not self.verification.get("installed_hashes"):
            raise ValueError("SOURCE_REUSE_INSTALL_HASH_REQUIRED")
        if self.status in {"provider_error", "download_error", "target_compile_failed"} and not self.failure_code:
            raise ValueError("SOURCE_REUSE_FAILURE_CODE_REQUIRED")

    def to_dict(self):
        return asdict(self)


def _evidence_id(capability: str, source_id: str, status: str) -> str:
    digest = hashlib.sha256((capability + "\0" + source_id + "\0" + status).encode()).hexdigest()[:20]
    return "reuse_" + digest


def _capability_requirements(requirements, graph):
    by_capability = {}
    known = {str(r.get("requirement_id")) for r in requirements}
    for row in graph.get("requirement_bindings", ()):
        rid, capability = str(row.get("requirement_id", "")), str(row.get("capability", ""))
        if rid in known and capability:
            by_capability.setdefault(capability, set()).add(rid)
    return by_capability


def _donor_files(donor):
    return tuple(dict(x) for x in donor.get("files", ()) if isinstance(x, Mapping))


def build_source_reuse_evidence(requirements, reuse_plan, target_symbols):
    """Convert pinned donor decisions into inspectable obligations, never inferred source.

    Code and installed hashes are attached by later materialization receipts;
    target_symbols is the real inspected API inventory, not a language-model guess.
    """
    if not isinstance(reuse_plan, Mapping):
        raise ValueError("SOURCE_REUSE_PLAN_REQUIRED")
    by_cap = _capability_requirements(requirements, reuse_plan.get("capability_graph", {}))
    rows = []
    for decision in reuse_plan.get("capabilities", ()):
        capability = str(decision.get("capability", ""))
        source_id = str(decision.get("source_id", ""))
        donor = decision.get("donor") or {}
        mode = str(decision.get("mode", ""))
        if mode == "source_transplant":
            status = "verified"
        elif mode == "fresh":
            status = str(decision.get("search_status") or "unresolved")
            # Older saved plans emitted the discovery-only "catalog_only"
            # status. Such metadata is not a reusable source receipt.
            if status == "catalog_only":
                status = "unresolved"
            if status not in {"search_empty", "provider_error", "download_error",
                              "target_compile_failed", "rejected", "unresolved", "reference_available"}:
                raise ValueError("SOURCE_REUSE_UNRECOGNIZED_FRESH_REASON")
        else:
            raise ValueError("SOURCE_REUSE_UNRECOGNIZED_DECISION_MODE")
        failure_code = str(decision.get("failure_code") or "")
        if status == "provider_error" and not failure_code:
            # Legacy plans did not attach the required provider failure code.
            failure_code = "CATALOG_PROVIDER_ERROR_UNSPECIFIED"
        files = _donor_files(donor)
        if mode == "source_transplant" and not decision.get("proof_receipt"):
            raise ValueError("SOURCE_REUSE_PROOF_REQUIRED")
        symbols = tuple(s for s in donor.get("source_symbols", ())
                        if s in target_symbols and target_symbols[s].get("kind"))
        rows.append(SourceReuseEvidence(
            evidence_id=_evidence_id(capability, source_id, status),
            capability=capability,
            requirement_ids=tuple(sorted(by_cap.get(capability, ()))),
            mode=mode,
            status=status,
            repository=str(donor.get("repository", "")),
            commit_sha=str(donor.get("commit_sha", "")),
            license_id=str(donor.get("license_id", "")),
            source_id=source_id,
            files=files,
            symbols=symbols,
            dependencies=tuple(donor.get("required_dependencies", ())),
            remaining_behaviors=tuple(decision.get("remaining_behaviors", ())),
            verification=dict(decision.get("proof_receipt") or {}),
            failure_code=failure_code,
            failure_detail=str(decision.get("rationale", "")),
        ))
    for reference in reuse_plan.get("reference_sources", ()):
        donor = reference.get("donor") or {}
        capability = str(reference.get("capability", ""))
        if donor:
            repository = str(donor.get("repository", ""))
            commit_sha = str(donor.get("commit_sha", ""))
            license_id = str(donor.get("license_id", ""))
            files = _donor_files(donor)
        else:
            from .curseforge_semantic_source_review import direct_source_identity
            _, owner, name = direct_source_identity(str(reference.get("source_url") or ""))
            repository = owner + "/" + name
            commit_sha = str(reference.get("pinned_sha") or "")
            license_id = str(reference.get("license_spdx") or "")
            files = ({"path": str(reference.get("selected_file") or ""),
                      "sha256": str(reference.get("file_sha256") or "")},)
        source_id = "host-donor:" + repository + "@" + commit_sha
        rows.append(SourceReuseEvidence(
            evidence_id=_evidence_id(capability, source_id, "reference_available"),
            capability=capability,
            requirement_ids=tuple(sorted(by_cap.get(capability, ()))),
            # Partial slices and rejected target proofs can guide an optional
            # port without becoming mandatory source dependencies or APIs.
            mode="reference", status="reference_available", source_id=source_id,
            repository=repository,
            commit_sha=commit_sha,
            license_id=license_id,
            files=files,
            dependencies=tuple(donor.get("required_dependencies", ())),
            failure_detail=str(reference.get("migration_diagnostics", "")),
        ))
    return tuple(rows)
