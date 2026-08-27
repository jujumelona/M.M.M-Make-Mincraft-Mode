from __future__ import annotations

"""Multi-donor composition with static shortlist and executable joint proof.

Static compatibility only selects candidates. A reusable donor subgraph must carry an
executable receipt bound to its exact commit and artifact closure. Production joint
verification first proves every donor independently, then materializes those donors
together, injects only dependency resolver receipts, and compiles the combined artifact
on an attested host scaffold.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dependency_resolver import (
    DependencyResolutionReceipt,
    inject_resolved_dependencies_into_build_gradle,
    parse_donor_build_metadata,
    resolve_dependency_for_target,
)
from .source_transplant import DonorSlice


@dataclass(frozen=True)
class CompositionConflict:
    conflict_type: str
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
    proof_level: str = "UNVERIFIED"
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
            "selected_donors": [donor.to_dict() for donor in self.selected_donors],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "resolved_dependencies": [
                receipt.to_dict() for receipt in self.resolved_dependencies
            ],
            "required_capabilities": list(self.required_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "residual_capabilities": list(self.residual_capabilities),
            "complete_coverage": self.complete_coverage,
            "subgraph_receipts": [
                receipt.to_dict() for receipt in self.subgraph_receipts
            ],
        }


def _receipt_field(receipt: Any, name: str, default: Any = None) -> Any:
    if isinstance(receipt, Mapping):
        return receipt.get(name, default)
    return getattr(receipt, name, default)


def _donor_key(donor: DonorSlice) -> str:
    return f"{donor.repository}@{donor.commit_sha}"


def _donor_closure_hash(donor: DonorSlice) -> str:
    payload = "".join(
        f"{donor_file.path}:{donor_file.sha256}"
        for donor_file in sorted(donor.files, key=lambda item: item.path)
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_receipt_hash(receipt: Any) -> str:
    if hasattr(receipt, "to_dict"):
        value = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        value = dict(receipt)
    else:
        return ""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bound_compile_receipt(donor: DonorSlice, receipt: Any) -> bool:
    if receipt is None or not bool(_receipt_field(receipt, "compile_passed", False)):
        return False
    if str(_receipt_field(receipt, "commit_sha", "")) != donor.commit_sha:
        return False
    if str(_receipt_field(receipt, "closure_hash", "")) != _donor_closure_hash(donor):
        return False
    return str(_receipt_field(receipt, "proof_level", "")) in {
        "COMPILE_VERIFIED",
        "BEHAVIOR_VERIFIED",
        "INTEGRATION_VERIFIED",
        "RUNTIME_BOOT_VERIFIED",
        "HOST_VERIFIED",
    }


def _receipt_for_donor(
    donor: DonorSlice,
    receipts: Mapping[str, Any] | None,
) -> Any:
    if not receipts:
        return None
    return receipts.get(_donor_key(donor)) or receipts.get(donor.capability)


def _host_build_infrastructure(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    return normalized in {
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradle.properties",
        "gradlew",
        "gradlew.bat",
    } or normalized.startswith("gradle/wrapper/")


def _static_conflicts(donors: Sequence[DonorSlice]) -> list[CompositionConflict]:
    conflicts: list[CompositionConflict] = []
    seen_fqcns: dict[str, str] = {}
    seen_files: dict[str, str] = {}
    seen_registry_ids: dict[str, str] = {}

    for donor in donors:
        for donor_file in donor.files:
            norm_path = donor_file.path.replace("\\", "/").strip("/")
            if _host_build_infrastructure(norm_path):
                continue
            prior_file_owner = seen_files.get(norm_path)
            if prior_file_owner is not None:
                conflicts.append(
                    CompositionConflict(
                        "class_collision",
                        (norm_path, prior_file_owner, donor.repository),
                        (
                            f"Duplicate file path '{norm_path}' defined in multiple "
                            f"donors: {prior_file_owner} and {donor.repository}"
                        ),
                    )
                )
            else:
                seen_files[norm_path] = donor.repository

            if norm_path.endswith((".java", ".kt")):
                fqcn = norm_path
                for prefix in ("src/main/java/", "src/main/kotlin/", "src/"):
                    if fqcn.startswith(prefix):
                        fqcn = fqcn[len(prefix) :]
                        break
                fqcn = fqcn.replace("/", ".").rsplit(".", 1)[0]
                prior_fqcn_owner = seen_fqcns.get(fqcn)
                if prior_fqcn_owner is not None:
                    conflicts.append(
                        CompositionConflict(
                            "fqcn_collision",
                            (fqcn, prior_fqcn_owner, donor.repository),
                            (
                                f"FQCN '{fqcn}' collision between "
                                f"{prior_fqcn_owner} and {donor.repository}"
                            ),
                        )
                    )
                else:
                    seen_fqcns[fqcn] = donor.repository

            for symbol in donor_file.symbols:
                if ":" not in symbol:
                    continue
                prior_registry_owner = seen_registry_ids.get(symbol)
                if (
                    prior_registry_owner is not None
                    and prior_registry_owner != donor.repository
                ):
                    conflicts.append(
                        CompositionConflict(
                            "registry_collision",
                            (symbol, prior_registry_owner, donor.repository),
                            (
                                f"Registry ID '{symbol}' collision between "
                                f"{prior_registry_owner} and {donor.repository}"
                            ),
                        )
                    )
                else:
                    seen_registry_ids[symbol] = donor.repository
    return conflicts


def _resolve_declared_dependencies(
    donors: Sequence[DonorSlice],
    *,
    target_loader: str,
    target_minecraft: str,
) -> tuple[list[DependencyResolutionReceipt], list[CompositionConflict]]:
    receipts: list[DependencyResolutionReceipt] = []
    conflicts: list[CompositionConflict] = []
    selected_by_name: dict[str, str] = {}
    for donor in donors:
        for dependency in donor.required_dependencies:
            receipt = resolve_dependency_for_target(
                dependency,
                target_loader=target_loader,
                target_minecraft=target_minecraft,
            )
            receipts.append(receipt)
            if not receipt.is_resolved:
                conflicts.append(
                    CompositionConflict(
                        "unresolved_dependency",
                        (dependency, donor.repository, receipt.resolution_reason),
                        (
                            f"Mandatory dependency '{dependency}' could not be resolved "
                            f"for {target_loader}@{target_minecraft}: "
                            f"{receipt.resolution_reason}"
                        ),
                    )
                )
                continue
            previous = selected_by_name.get(receipt.dependency_name)
            if previous is not None and previous != receipt.resolved_coordinate:
                conflicts.append(
                    CompositionConflict(
                        "dependency_version_conflict",
                        (
                            receipt.dependency_name,
                            previous,
                            receipt.resolved_coordinate,
                        ),
                        (
                            "Conflicting resolved coordinates for dependency "
                            f"'{receipt.dependency_name}': {previous} vs "
                            f"{receipt.resolved_coordinate}"
                        ),
                    )
                )
            else:
                selected_by_name[receipt.dependency_name] = receipt.resolved_coordinate
    return receipts, conflicts


def solve_multi_donor_composition(
    donors: Sequence[DonorSlice],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    required_capabilities: Sequence[str] = (),
    build_receipts: Mapping[str, Any] | None = None,
) -> CompositionResult:
    conflicts = _static_conflicts(donors)
    resolved_deps, dep_conflicts = _resolve_declared_dependencies(
        donors,
        target_loader=target_loader,
        target_minecraft=target_minecraft,
    )
    conflicts.extend(dep_conflicts)

    covered_caps = tuple(dict.fromkeys(donor.capability for donor in donors))
    req_caps = tuple(dict.fromkeys(required_capabilities or covered_caps))
    residual_caps = tuple(cap for cap in req_caps if cap not in covered_caps)
    complete_coverage = not residual_caps
    is_valid = not conflicts and complete_coverage

    subgraph_receipts: list[SubgraphProofReceipt] = []
    for donor in donors:
        receipt = _receipt_for_donor(donor, build_receipts)
        verified = bool(
            is_valid
            and donor.closure_complete
            and donor.target_compatibility in {"exact", "metadata_exact"}
            and _bound_compile_receipt(donor, receipt)
        )
        closure_hash = _donor_closure_hash(donor)
        subgraph_receipts.append(
            SubgraphProofReceipt(
                subgraph_id=f"{_donor_key(donor)}:{donor.capability}:{closure_hash}",
                capability=donor.capability,
                repository=donor.repository,
                commit_sha=donor.commit_sha,
                closure_hash=closure_hash,
                artifact_paths=tuple(
                    donor_file.path
                    for donor_file in donor.files
                    if not _host_build_infrastructure(donor_file.path)
                ),
                is_verified=verified,
                proof_level=(
                    str(_receipt_field(receipt, "proof_level", "COMPILE_VERIFIED"))
                    if verified
                    else "CLOSURE_COMPLETE"
                    if donor.closure_complete
                    else "PINNED"
                    if donor.commit_sha
                    else "DISCOVERED"
                ),
                build_receipt_hash=(
                    _canonical_receipt_hash(receipt) if verified else ""
                ),
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


def _joint_artifact_hash(files: Mapping[str, str | bytes]) -> str:
    payload = "".join(
        f"{path}:{hashlib.sha256((value.encode('utf-8') if isinstance(value, str) else value)).hexdigest()}"
        for path, value in sorted(files.items())
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_individual_proofs(
    donors: Sequence[DonorSlice],
    target_context: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return donor-bound executable receipts, proving any missing donor now."""

    receipts: dict[str, Any] = dict(supplied or {})
    failures: list[str] = []
    from .reuse_proof_executor import execute_reuse_proof

    for donor in donors:
        key = _donor_key(donor)
        receipt = _receipt_for_donor(donor, receipts)
        if not _bound_compile_receipt(donor, receipt):
            receipt = execute_reuse_proof(
                donor,
                target_workspace="",
                target_context=target_context,
                run_tests=False,
            )
            receipts[key] = receipt
        if not _bound_compile_receipt(donor, receipt):
            failures.append(key)
    return receipts, tuple(failures)


def verify_joint_composition_sandbox(
    donors: Sequence[DonorSlice],
    target_context: Mapping[str, Any],
    compile_checker: Any = None,
    *,
    individual_build_receipts: Mapping[str, Any] | None = None,
    required_capabilities: Sequence[str] = (),
    require_individual_proof: bool | None = None,
) -> tuple[bool, Mapping[str, Any]]:
    """Prove donors independently, then compile their exact combined artifact."""

    if not donors:
        return False, {"compile_passed": False, "error": "NO_DONORS"}

    required = tuple(dict.fromkeys(required_capabilities))
    covered = {donor.capability for donor in donors}
    missing = tuple(cap for cap in required if cap not in covered)
    if missing:
        return False, {
            "compile_passed": False,
            "error": "INCOMPLETE_JOINT_CAPABILITY_COVERAGE",
            "missing_capabilities": list(missing),
        }

    # Production (real build verifier) always requires independent donor proof.
    # Synthetic compile_checker seams remain opt-in so legacy unit tests do not gain
    # authority over production proof semantics.
    must_prove = (
        compile_checker is None
        if require_individual_proof is None
        else bool(require_individual_proof)
    )
    effective_receipts = dict(individual_build_receipts or {})
    if must_prove:
        effective_receipts, proof_failures = _ensure_individual_proofs(
            donors,
            target_context,
            effective_receipts,
        )
        if proof_failures:
            return False, {
                "compile_passed": False,
                "error": "UNVERIFIED_JOINT_DONOR",
                "donors": list(proof_failures),
            }

    donor_receipt_hashes: dict[str, str] = {}
    if must_prove:
        for donor in donors:
            receipt = _receipt_for_donor(donor, effective_receipts)
            if not _bound_compile_receipt(donor, receipt):
                return False, {
                    "compile_passed": False,
                    "error": "UNVERIFIED_JOINT_DONOR",
                    "donor": _donor_key(donor),
                }
            donor_receipt_hashes[_donor_key(donor)] = _canonical_receipt_hash(receipt)

    import tempfile
    from pathlib import Path

    from .reuse_adapters import apply_deterministic_adapters
    from .reuse_build_verifier import verify_scratch_workspace_build
    from .source_transplant import materialize_pinned_donor
    from .verified_scaffold_registry import apply_verified_scaffold

    all_adapted_files: dict[str, str | bytes] = {}
    declared_dependencies: list[str] = []
    for donor in donors:
        raw_files: dict[str, str | bytes] = {}
        try:
            raw_map = materialize_pinned_donor(donor)
        except Exception as exc:
            return False, {
                "compile_passed": False,
                "error": f"DONOR_MATERIALIZATION_ERROR: {_donor_key(donor)} - {exc}",
            }
        if not raw_map and donor.files:
            return False, {
                "compile_passed": False,
                "error": f"DONOR_MATERIALIZATION_FAILED: {_donor_key(donor)}",
            }
        for rel_path, raw_bytes in raw_map.items():
            try:
                raw_files[rel_path] = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_files[rel_path] = raw_bytes

        declared_dependencies.extend(donor.required_dependencies)
        declared_dependencies.extend(parse_donor_build_metadata(raw_files))
        adapted, _ = apply_deterministic_adapters(raw_files, target_context)
        for rel_path, content in adapted.items():
            if _host_build_infrastructure(rel_path):
                continue
            if rel_path in all_adapted_files:
                return False, {
                    "compile_passed": False,
                    "error": (
                        "JOINT_MERGE_FILE_COLLISION: Duplicate path "
                        f"'{rel_path}' across donors"
                    ),
                }
            all_adapted_files[rel_path] = content

    loader = str(target_context.get("loader") or "fabric")
    minecraft_version = str(target_context.get("minecraft_version") or "1.21.1")
    dependency_receipts = tuple(
        resolve_dependency_for_target(
            dependency,
            target_loader=loader,
            target_minecraft=minecraft_version,
        )
        for dependency in dict.fromkeys(declared_dependencies)
    )
    unresolved = tuple(
        f"{receipt.dependency_name}:{receipt.resolution_reason}"
        for receipt in dependency_receipts
        if not receipt.is_resolved
    )
    if unresolved:
        return False, {
            "compile_passed": False,
            "error": "UNRESOLVED_JOINT_DEPENDENCIES",
            "unresolved_dependencies": list(unresolved),
            "resolved_dependencies": [
                receipt.to_dict() for receipt in dependency_receipts
            ],
        }

    joint_hash = _joint_artifact_hash(all_adapted_files)
    if callable(compile_checker):
        result = compile_checker(all_adapted_files, target_context)
        payload = dict(result) if isinstance(result, Mapping) else {
            "compile_passed": bool(result)
        }
        payload["joint_artifact_hash"] = joint_hash
        payload["donor_receipt_hashes"] = donor_receipt_hashes
        payload["resolved_dependencies"] = [
            receipt.to_dict() for receipt in dependency_receipts
        ]
        return bool(payload.get("compile_passed")), payload

    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_path = Path(temp_dir)
        apply_verified_scaffold(sandbox_path, target_context)
        if dependency_receipts:
            kts_file = sandbox_path / "build.gradle.kts"
            groovy_file = sandbox_path / "build.gradle"
            build_target = kts_file if kts_file.exists() else groovy_file
            if not build_target.exists():
                return False, {
                    "compile_passed": False,
                    "error": "JOINT_DEPENDENCY_INJECTION_BUILD_FILE_MISSING",
                }
            try:
                build_text = build_target.read_text(encoding="utf-8")
                injected, _ = inject_resolved_dependencies_into_build_gradle(
                    build_text,
                    dependency_receipts,
                    is_kotlin_dsl=kts_file.exists(),
                )
                build_target.write_text(injected, encoding="utf-8")
            except (OSError, RuntimeError, ValueError) as exc:
                return False, {
                    "compile_passed": False,
                    "error": f"JOINT_DEPENDENCY_INJECTION_FAILED: {exc}",
                }

        for rel_path, content in all_adapted_files.items():
            destination = sandbox_path / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            else:
                destination.write_text(str(content), encoding="utf-8")

        build_receipt = verify_scratch_workspace_build(sandbox_path)
        payload = build_receipt.to_dict()
        payload["resolved_dependencies"] = [
            receipt.to_dict() for receipt in dependency_receipts
        ]
        payload["joint_artifact_hash"] = joint_hash
        payload["donor_receipt_hashes"] = donor_receipt_hashes
        payload["required_capabilities"] = list(required)
        payload["covered_capabilities"] = sorted(covered)
        return build_receipt.compile_passed, payload


def _score_composition_beam(combo: Sequence[DonorSlice], total_caps: int) -> float:
    if not combo:
        return 0.0
    covered = len({donor.capability for donor in combo})
    return (
        (covered / max(1, total_caps)) * 100.0
        + sum(donor.confidence for donor in combo) * 10.0
        + sum(20.0 if donor.closure_complete else 5.0 for donor in combo)
        - sum(donor.adaptation_cost for donor in combo) * 5.0
        - len(combo) * 2.0
    )


def search_ranked_donor_composition_beams(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 6,
    build_receipts: Mapping[str, Any] | None = None,
) -> tuple[CompositionResult, ...]:
    if not candidates_by_capability:
        return (CompositionResult(is_valid=True),)
    caps = list(candidates_by_capability)
    if any(not candidates_by_capability[capability] for capability in caps):
        return ()

    beams: list[tuple[DonorSlice, ...]] = [()]
    for capability in caps:
        next_beams: list[tuple[DonorSlice, ...]] = []
        for combo in beams:
            for candidate in candidates_by_capability[capability]:
                new_combo = (*combo, candidate)
                evaluation = solve_multi_donor_composition(
                    new_combo,
                    target_loader=target_loader,
                    target_minecraft=target_minecraft,
                    required_capabilities=tuple(
                        donor.capability for donor in new_combo
                    ),
                    build_receipts=build_receipts,
                )
                if not evaluation.conflicts:
                    next_beams.append(new_combo)
        if not next_beams:
            return ()
        next_beams.sort(
            key=lambda beam: _score_composition_beam(beam, len(caps)),
            reverse=True,
        )
        beams = next_beams[: max(1, beam_width)]

    results: list[CompositionResult] = []
    for beam in beams:
        result = solve_multi_donor_composition(
            beam,
            target_loader=target_loader,
            target_minecraft=target_minecraft,
            required_capabilities=tuple(caps),
            build_receipts=build_receipts,
        )
        if not result.is_valid:
            continue
        if build_receipts is not None and not all(
            receipt.is_verified for receipt in result.subgraph_receipts
        ):
            continue
        results.append(result)
    return tuple(results)


def search_best_donor_composition(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 4,
    build_receipts: Mapping[str, Any] | None = None,
) -> CompositionResult:
    ranked = search_ranked_donor_composition_beams(
        candidates_by_capability,
        target_loader=target_loader,
        target_minecraft=target_minecraft,
        beam_width=beam_width,
        build_receipts=build_receipts,
    )
    if ranked:
        return ranked[0]
    caps = tuple(candidates_by_capability)
    return CompositionResult(
        is_valid=False,
        required_capabilities=caps,
        residual_capabilities=caps,
        complete_coverage=False,
    )


def generate_reuse_manifest(
    selected_donors: Sequence[DonorSlice] = (),
    project_name: str = "custom_mod",
    *,
    selected_bundles: Sequence[Any] = (),
) -> dict[str, Any]:
    manifest_files: list[dict[str, Any]] = []
    for donor in selected_donors:
        for donor_file in donor.files:
            if _host_build_infrastructure(donor_file.path):
                continue
            manifest_files.append(
                {
                    "path": donor_file.path,
                    "origin_repo": donor.repository,
                    "origin_commit": donor.commit_sha,
                    "origin_blob_sha": donor_file.blob_sha,
                    "origin_sha256": donor_file.sha256,
                    "license": donor.license_id,
                    "is_reused": True,
                }
            )

    bundle_values: list[dict[str, Any]] = []
    for bundle in selected_bundles:
        bundle_values.append(
            bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
        )
        provenance = getattr(bundle, "provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        file_hashes = getattr(bundle, "file_hashes", {})
        for path in getattr(bundle, "protected_paths", ()):
            manifest_files.append(
                {
                    "path": path,
                    "origin_repo": provenance.get(
                        "repository", getattr(bundle, "source_ref", "")
                    ),
                    "origin_commit": provenance.get("commit_sha", ""),
                    "origin_blob_sha": "",
                    "origin_sha256": (
                        file_hashes.get(path, "")
                        if isinstance(file_hashes, Mapping)
                        else ""
                    ),
                    "license": provenance.get("license_id", ""),
                    "bundle_id": getattr(bundle, "bundle_id", ""),
                    "origin_kind": getattr(bundle, "origin_kind", ""),
                    "is_reused": True,
                }
            )

    return {
        "schema_version": "mmm/reuse-manifest-v1",
        "project_name": project_name,
        "total_reused_files": len(manifest_files),
        "reused_file_count": len(manifest_files),
        "donor_count": len(selected_donors)
        + sum(
            getattr(bundle, "origin_kind", "") == "external_donor"
            for bundle in selected_bundles
        ),
        "bundle_count": len(selected_bundles),
        "donors": [donor.to_dict() for donor in selected_donors],
        "bundles": bundle_values,
        "files": manifest_files,
    }
