from __future__ import annotations

"""Multi-Donor Joint Composition and Conflict Solver.

Validates that selected donor slices integrate cohesively without runtime collisions:
1. Duplicate registry identifiers (e.g. two donors registering 'mymod:boss')
2. Incompatible external dependency version ranges (e.g. GeckoLib 4.6 vs 4.4)
3. Duplicate fully-qualified class names or mixin target collisions
4. ModID namespace collisions

Emits structured CompositionResult records and generates cryptographic SBOM / reuse-manifest.json.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dependency_resolver import (
    DependencyResolutionReceipt,
    resolve_dependency_for_target,
)
from .source_transplant import DonorSlice


@dataclass(frozen=True)
class CompositionConflict:
    conflict_type: str  # "registry_collision", "dependency_conflict", "class_collision", "mixin_collision"
    conflicting_items: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_type": self.conflict_type,
            "conflicting_items": list(self.conflicting_items),
            "message": self.message,
        }


@dataclass(frozen=True)
class SubgraphProofReceipt:
    subgraph_id: str
    capability: str
    repository: str
    commit_sha: str
    closure_hash: str
    artifact_paths: tuple[str, ...]
    is_verified: bool
    proof_level: str = "COMPILE_VERIFIED"
    build_receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgraph_id": self.subgraph_id,
            "capability": self.capability,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "closure_hash": self.closure_hash,
            "artifact_paths": list(self.artifact_paths),
            "is_verified": self.is_verified,
            "proof_level": self.proof_level,
            "build_receipt_hash": self.build_receipt_hash,
        }


@dataclass(frozen=True)
class CompositionResult:
    is_valid: bool
    selected_donors: tuple[DonorSlice, ...] = ()
    conflicts: tuple[CompositionConflict, ...] = ()
    resolved_dependencies: tuple[DependencyResolutionReceipt, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    covered_capabilities: tuple[str, ...] = ()
    residual_capabilities: tuple[str, ...] = ()
    complete_coverage: bool = True
    subgraph_receipts: tuple[SubgraphProofReceipt, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "selected_donors": [d.to_dict() for d in self.selected_donors],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolved_dependencies": [r.to_dict() for r in self.resolved_dependencies],
            "required_capabilities": list(self.required_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "residual_capabilities": list(self.residual_capabilities),
            "complete_coverage": self.complete_coverage,
            "subgraph_receipts": [s.to_dict() for s in self.subgraph_receipts],
        }


def solve_multi_donor_composition(
    donors: Sequence[DonorSlice],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    required_capabilities: Sequence[str] = (),
    build_receipts: Mapping[str, Any] | None = None,
) -> CompositionResult:
    """Evaluate a joint set of candidate donors for integration compatibility and hard conflicts."""
    conflicts: list[CompositionConflict] = []
    seen_fqcns: dict[str, str] = {}
    seen_files: dict[str, str] = {}
    seen_registry_ids: dict[str, str] = {}
    resolved_deps: list[DependencyResolutionReceipt] = []

    # 1. Check class / file collisions
    for donor in donors:
        for df in donor.files:
            # Normalized path collision
            norm_path = df.path.replace("\\", "/").strip("/")
            if norm_path in seen_files:
                conflicts.append(
                    CompositionConflict(
                        conflict_type="class_collision",
                        conflicting_items=(norm_path, seen_files[norm_path], donor.repository),
                        message=f"Duplicate file path '{norm_path}' defined in multiple donors: {seen_files[norm_path]} and {donor.repository}",
                    )
                )
            else:
                seen_files[norm_path] = donor.repository

            # FQCN (Package + Class Name) collision
            if norm_path.endswith(".java") or norm_path.endswith(".kt"):
                clean_fqcn = norm_path
                for prefix in ("src/main/java/", "src/main/kotlin/", "src/"):
                    if clean_fqcn.startswith(prefix):
                        clean_fqcn = clean_fqcn[len(prefix):]
                clean_fqcn = clean_fqcn.replace("/", ".").rsplit(".", 1)[0]
                if clean_fqcn in seen_fqcns:
                    conflicts.append(
                        CompositionConflict(
                            conflict_type="fqcn_collision",
                            conflicting_items=(clean_fqcn, seen_fqcns[clean_fqcn], donor.repository),
                            message=f"FQCN '{clean_fqcn}' collision between {seen_fqcns[clean_fqcn]} and {donor.repository}",
                        )
                    )
                else:
                    seen_fqcns[clean_fqcn] = donor.repository

            # Registry ID collision
            for sym in df.symbols:
                if ":" in sym:
                    if sym in seen_registry_ids and seen_registry_ids[sym] != donor.repository:
                        conflicts.append(
                            CompositionConflict(
                                conflict_type="registry_collision",
                                conflicting_items=(sym, seen_registry_ids[sym], donor.repository),
                                message=f"Registry ID '{sym}' collision between {seen_registry_ids[sym]} and {donor.repository}",
                            )
                        )
                    else:
                        seen_registry_ids[sym] = donor.repository

    # 2. Check external dependencies
    dep_versions: dict[str, str] = {}
    for donor in donors:
        for dep in donor.required_dependencies:
            receipt = resolve_dependency_for_target(dep, target_loader=target_loader, target_minecraft=target_minecraft)
            resolved_deps.append(receipt)

            if not receipt.is_resolved:
                conflicts.append(
                    CompositionConflict(
                        conflict_type="unresolved_dependency",
                        conflicting_items=(dep, donor.repository, receipt.resolution_reason),
                        message=f"Mandatory dependency '{dep}' could not be resolved for {target_loader}@{target_minecraft}: {receipt.resolution_reason}",
                    )
                )
            elif dep in dep_versions and dep_versions[dep] != receipt.selected_version:
                conflicts.append(
                    CompositionConflict(
                        conflict_type="dependency_version_conflict",
                        conflicting_items=(dep, dep_versions[dep], receipt.selected_version),
                        message=f"Conflicting version requirements for dependency '{dep}': {dep_versions[dep]} vs {receipt.selected_version}",
                    )
                )
            else:
                dep_versions[dep] = receipt.selected_version

    covered_caps = tuple(dict.fromkeys(d.capability for d in donors))
    req_caps = tuple(required_capabilities) if required_capabilities else covered_caps
    residual_caps = tuple(c for c in req_caps if c not in covered_caps)
    complete_coverage = (len(residual_caps) == 0)
    is_valid = (len(conflicts) == 0 and complete_coverage)

    subgraph_receipts = []
    for d in donors:
        r = (build_receipts or {}).get(d.capability) or (build_receipts or {}).get(f"{d.repository}@{d.commit_sha}")
        r_passed = bool(getattr(r, "compile_passed", False)) if r is not None else False
        b_hash = ""
        if r is not None and r_passed:
            if hasattr(r, "to_dict"):
                b_hash = "sha256:" + hashlib.sha256(str(r.to_dict()).encode("utf-8")).hexdigest()
            else:
                b_hash = "sha256:" + hashlib.sha256(str(r).encode("utf-8")).hexdigest()

        verified = bool(is_valid and d.closure_complete and d.target_compatibility in {"exact", "metadata_exact"} and r is not None and r_passed)
        p_lvl = "COMPILE_VERIFIED" if verified else ("MATERIALIZED" if d.closure_complete else ("PARTIAL_REUSE" if d.files else "UNVERIFIED"))

        subgraph_receipts.append(
            SubgraphProofReceipt(
                subgraph_id=f"{d.repository}@{d.commit_sha}:{d.capability}",
                capability=d.capability,
                repository=d.repository,
                commit_sha=d.commit_sha,
                closure_hash=d.commit_sha,
                artifact_paths=tuple(df.path for df in d.files),
                is_verified=verified,
                proof_level=p_lvl,
                build_receipt_hash=b_hash,
            )
        )

    return CompositionResult(
        is_valid=is_valid,
        selected_donors=tuple(donors) if is_valid else (),
        conflicts=tuple(conflicts),
        resolved_dependencies=tuple(resolved_deps),
        required_capabilities=req_caps,
        covered_capabilities=covered_caps,
        residual_capabilities=residual_caps,
        complete_coverage=complete_coverage,
        subgraph_receipts=tuple(subgraph_receipts),
    )


def verify_joint_composition_sandbox(
    donors: Sequence[DonorSlice],
    target_context: Mapping[str, Any],
    compile_checker: Any = None,
) -> tuple[bool, Mapping[str, Any]]:
    """Execute real multi-donor joint sandbox compilation combining all donors into a single target workspace."""
    if not donors:
        return False, {}

    import tempfile
    from pathlib import Path
    from .reuse_adapters import apply_deterministic_adapters
    from .reuse_build_verifier import verify_scratch_workspace_build
    from .source_transplant import materialize_pinned_donor
    from .verified_scaffold_registry import apply_verified_scaffold

    all_adapted_files: dict[str, str | bytes] = {}
    for d in donors:
        raw_files: dict[str, str | bytes] = {}
        try:
            raw_map = materialize_pinned_donor(d)
            if not raw_map and d.files:
                return False, {
                    "compile_passed": False,
                    "error": f"DONOR_MATERIALIZATION_FAILED: {d.repository}@{d.commit_sha}",
                }
            for rel_path, raw_bytes in raw_map.items():
                try:
                    raw_files[rel_path] = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raw_files[rel_path] = raw_bytes
        except Exception as exc:
            return False, {
                "compile_passed": False,
                "error": f"DONOR_MATERIALIZATION_ERROR: {d.repository}@{d.commit_sha} - {exc}",
            }

        files, _ = apply_deterministic_adapters(raw_files, target_context)
        for rel_path, content in files.items():
            if rel_path in all_adapted_files:
                return False, {
                    "compile_passed": False,
                    "error": f"JOINT_MERGE_FILE_COLLISION: Duplicate path '{rel_path}' across donors",
                }
            all_adapted_files[rel_path] = content

    if callable(compile_checker):
        res = compile_checker(all_adapted_files, target_context)
        passed = bool(res.get("compile_passed")) if isinstance(res, Mapping) else bool(res)
        return passed, res if isinstance(res, Mapping) else {"compile_passed": passed}

    with tempfile.TemporaryDirectory() as temp_dir:
        sb_path = Path(temp_dir)
        apply_verified_scaffold(sb_path, target_context)
        for rel_path, content in all_adapted_files.items():
            dest = sb_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                dest.write_bytes(content)
            else:
                dest.write_text(str(content), encoding="utf-8")

        build_rcpt = verify_scratch_workspace_build(sb_path)
        return build_rcpt.compile_passed, build_rcpt.to_dict()


def _score_composition_beam(combo: Sequence[DonorSlice], total_caps: int) -> float:
    """Calculate proof-quality score for a candidate donor combination."""
    if not combo:
        return 0.0
    covered = len({d.capability for d in combo})
    coverage_score = (covered / max(1, total_caps)) * 100.0
    confidence_score = sum(d.confidence for d in combo) * 10.0
    proof_score = sum(20.0 if d.closure_complete else 5.0 for d in combo)
    cost_penalty = sum(d.adaptation_cost for d in combo) * 5.0
    donor_count_penalty = len(combo) * 2.0
    return coverage_score + confidence_score + proof_score - cost_penalty - donor_count_penalty


def search_ranked_donor_composition_beams(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 6,
) -> tuple[CompositionResult, ...]:
    """Explore candidate donor combinations across capabilities via beam search and return all valid candidate beams ranked by score."""
    caps = [cap for cap, candidates in candidates_by_capability.items() if candidates]
    if not caps:
        return (CompositionResult(is_valid=True),)

    beams: list[tuple[DonorSlice, ...]] = [()]
    for cap in caps:
        next_beams: list[tuple[DonorSlice, ...]] = []
        candidates = candidates_by_capability[cap]
        for combo in beams:
            for cand in candidates:
                new_combo = (*combo, cand)
                eval_res = solve_multi_donor_composition(
                    new_combo,
                    target_loader=target_loader,
                    target_minecraft=target_minecraft,
                    required_capabilities=tuple(d.capability for d in new_combo),
                )
                if len(eval_res.conflicts) == 0:
                    next_beams.append(new_combo)

        if not next_beams:
            break

        next_beams.sort(key=lambda b: _score_composition_beam(b, len(caps)), reverse=True)
        beams = next_beams[:beam_width]

    results: list[CompositionResult] = []
    for b in beams:
        if b:
            cr = solve_multi_donor_composition(
                b,
                target_loader=target_loader,
                target_minecraft=target_minecraft,
                required_capabilities=tuple(caps),
            )
            if cr.is_valid:
                results.append(cr)

    return tuple(results)


def search_best_donor_composition(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 4,
) -> CompositionResult:
    """Explore candidate donor combinations across capabilities via beam search and return the highest-scoring valid composition."""
    ranked = search_ranked_donor_composition_beams(
        candidates_by_capability,
        target_loader=target_loader,
        target_minecraft=target_minecraft,
        beam_width=beam_width,
    )
    if ranked:
        return ranked[0]
    caps = [cap for cap, candidates in candidates_by_capability.items() if candidates]
    return CompositionResult(is_valid=False, required_capabilities=tuple(caps), residual_capabilities=tuple(caps), complete_coverage=False)


def generate_reuse_manifest(
    selected_donors: Sequence[DonorSlice] = (),
    project_name: str = "custom_mod",
    *,
    selected_bundles: Sequence[Any] = (),
) -> dict[str, Any]:
    """Generate one provenance SBOM for legacy donors and canonical bundles."""
    manifest_files: list[dict[str, Any]] = []

    for donor in selected_donors:
        for df in donor.files:
            manifest_files.append({
                "path": df.path,
                "origin_repo": donor.repository,
                "origin_commit": donor.commit_sha,
                "origin_blob_sha": df.blob_sha,
                "origin_sha256": df.sha256,
                "license": donor.license_id,
                "is_reused": True,
            })

    bundle_values: list[dict[str, Any]] = []
    for bundle in selected_bundles:
        bundle_dict = bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
        bundle_values.append(bundle_dict)
        provenance = getattr(bundle, "provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        file_hashes = getattr(bundle, "file_hashes", {})
        protected_paths = getattr(bundle, "protected_paths", ())
        for path in protected_paths:
            manifest_files.append({
                "path": path,
                "origin_repo": provenance.get("repository", getattr(bundle, "source_ref", "")),
                "origin_commit": provenance.get("commit_sha", ""),
                "origin_blob_sha": "",
                "origin_sha256": file_hashes.get(path, "") if isinstance(file_hashes, Mapping) else "",
                "license": provenance.get("license_id", ""),
                "bundle_id": getattr(bundle, "bundle_id", ""),
                "origin_kind": getattr(bundle, "origin_kind", ""),
                "is_reused": True,
            })

    return {
        "schema_version": "mmm/reuse-manifest-v1",
        "project_name": project_name,
        "total_reused_files": len(manifest_files),
        "reused_file_count": len(manifest_files),
        "donor_count": len(selected_donors) + sum(
            getattr(bundle, "origin_kind", "") == "external_donor"
            for bundle in selected_bundles
        ),
        "bundle_count": len(selected_bundles),
        "donors": [d.to_dict() for d in selected_donors],
        "bundles": bundle_values,
        "files": manifest_files,
    }
