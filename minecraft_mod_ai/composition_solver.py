from __future__ import annotations

"""Multi-donor joint composition and conflict solver.

Static compatibility can shortlist a composition, but only a donor-bound executable
build receipt can certify a subgraph.  Joint verification materializes all donors into
one verified target scaffold, resolves every declared dependency, injects only resolved
coordinates, and compiles the combined result.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dependency_resolver import (
    DependencyResolutionReceipt,
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
    """Accept only an executable receipt cryptographically bound to this donor closure."""

    if receipt is None or not bool(getattr(receipt, "compile_passed", False)):
        return False
    if str(getattr(receipt, "commit_sha", "")) != donor.commit_sha:
        return False
    if str(getattr(receipt, "closure_hash", "")) != _donor_closure_hash(donor):
        return False
    return str(getattr(receipt, "proof_level", "")) in {
        "COMPILE_VERIFIED",
        "BEHAVIOR_VERIFIED",
        "INTEGRATION_VERIFIED",
        "RUNTIME_BOOT_VERIFIED",
        "HOST_VERIFIED",
    }


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


def solve_multi_donor_composition(
    donors: Sequence[DonorSlice],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    required_capabilities: Sequence[str] = (),
    build_receipts: Mapping[str, Any] | None = None,
) -> CompositionResult:
    """Evaluate one joint donor set for static compatibility and receipt truthfulness."""

    conflicts: list[CompositionConflict] = []
    seen_fqcns: dict[str, str] = {}
    seen_files: dict[str, str] = {}
    seen_registry_ids: dict[str, str] = {}
    resolved_deps: list[DependencyResolutionReceipt] = []

    for donor in donors:
        for donor_file in donor.files:
            norm_path = donor_file.path.replace("\\", "/").strip("/")
            if _host_build_infrastructure(norm_path):
                continue
            if norm_path in seen_files:
                conflicts.append(
                    CompositionConflict(
                        conflict_type="class_collision",
                        conflicting_items=(
                            norm_path,
                            seen_files[norm_path],
                            donor.repository,
                        ),
                        message=(
                            f"Duplicate file path '{norm_path}' defined in multiple "
                            f"donors: {seen_files[norm_path]} and {donor.repository}"
                        ),
                    )
                )
            else:
                seen_files[norm_path] = donor.repository

            if norm_path.endswith((".java", ".kt")):
                clean_fqcn = norm_path
                for prefix in ("src/main/java/", "src/main/kotlin/", "src/"):
                    if clean_fqcn.startswith(prefix):
                        clean_fqcn = clean_fqcn[len(prefix) :]
                clean_fqcn = clean_fqcn.replace("/", ".").rsplit(".", 1)[0]
                if clean_fqcn in seen_fqcns:
                    conflicts.append(
                        CompositionConflict(
                            conflict_type="fqcn_collision",
                            conflicting_items=(
                                clean_fqcn,
                                seen_fqcns[clean_fqcn],
                                donor.repository,
                            ),
                            message=(
                                f"FQCN '{clean_fqcn}' collision between "
                                f"{seen_fqcns[clean_fqcn]} and {donor.repository}"
                            ),
                        )
                    )
                else:
                    seen_fqcns[clean_fqcn] = donor.repository

            for symbol in donor_file.symbols:
                if ":" not in symbol:
                    continue
                if (
                    symbol in seen_registry_ids
                    and seen_registry_ids[symbol] != donor.repository
                ):
                    conflicts.append(
                        CompositionConflict(
                            conflict_type="registry_collision",
                            conflicting_items=(
                                symbol,
                                seen_registry_ids[symbol],
                                donor.repository,
                            ),
                            message=(
                                f"Registry ID '{symbol}' collision between "
                                f"{seen_registry_ids[symbol]} and {donor.repository}"
                            ),
                        )
                    )
                else:
                    seen_registry_ids[symbol] = donor.repository

    dep_versions: dict[str, str] = {}
    for donor in donors:
        for dependency in donor.required_dependencies:
            receipt = resolve_dependency_for_target(
                dependency,
                target_loader=target_loader,
                target_minecraft=target_minecraft,
            )
            resolved_deps.append(receipt)
            if not receipt.is_resolved:
                conflicts.append(
                    CompositionConflict(
                        conflict_type="unresolved_dependency",
                        conflicting_items=(
                            dependency,
                            donor.repository,
                            receipt.resolution_reason,
                        ),
                        message=(
                            f"Mandatory dependency '{dependency}' could not be resolved "
                            f"for {target_loader}@{target_minecraft}: "
                            f"{receipt.resolution_reason}"
                        ),
                    )
                )
            elif (
                dependency in dep_versions
                and dep_versions[dependency] != receipt.selected_version
            ):
                conflicts.append(
                    CompositionConflict(
                        conflict_type="dependency_version_conflict",
                        conflicting_items=(
                            dependency,
                            dep_versions[dependency],
                            receipt.selected_version,
                        ),
                        message=(
                            f"Conflicting version requirements for dependency "
                            f"'{dependency}': {dep_versions[dependency]} vs "
                            f"{receipt.selected_version}"
                        ),
                    )
                )
            else:
                dep_versions[dependency] = receipt.selected_version

    covered_caps = tuple(dict.fromkeys(donor.capability for donor in donors))
    req_caps = tuple(required_capabilities) if required_capabilities else covered_caps
    residual_caps = tuple(cap for cap in req_caps if cap not in covered_caps)
    complete_coverage = not residual_caps
    is_valid = not conflicts and complete_coverage

    subgraph_receipts: list[SubgraphProofReceipt] = []
    for donor in donors:
        donor_key = f"{donor.repository}@{donor.commit_sha}"
        receipt = (build_receipts or {}).get(donor.capability) or (
            build_receipts or {}
        ).get(donor_key)
        verified = bool(
            is_valid
            and donor.closure_complete
            and donor.target_compatibility in {"exact", "metadata_exact"}
            and _bound_compile_receipt(donor, receipt)
        )
        if verified:
            proof_level = str(getattr(receipt, "proof_level", "COMPILE_VERIFIED"))
            build_receipt_hash = _canonical_receipt_hash(receipt)
        elif donor.closure_complete:
            proof_level = "CLOSURE_COMPLETE"
            build_receipt_hash = ""
        elif donor.commit_sha:
            proof_level = "PINNED"
            build_receipt_hash = ""
        else:
            proof_level = "DISCOVERED"
            build_receipt_hash = ""

        closure_hash = _donor_closure_hash(donor)
        subgraph_receipts.append(
            SubgraphProofReceipt(
                subgraph_id=(
                    f"{donor.repository}@{donor.commit_sha}:"
                    f"{donor.capability}:{closure_hash}"
                ),
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
                proof_level=proof_level,
                build_receipt_hash=build_receipt_hash,
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
    """Compile all selected donors together with exact resolved dependencies."""

    if not donors:
        return False, {"compile_passed": False, "error": "NO_DONORS"}

    import tempfile
    from pathlib import Path

    from .reuse_adapters import (
        DependencyAdaptationPlan,
        apply_deterministic_adapters,
    )
    from .reuse_build_verifier import verify_scratch_workspace_build
    from .source_transplant import materialize_pinned_donor
    from .verified_scaffold_registry import apply_verified_scaffold

    all_adapted_files: dict[str, str | bytes] = {}
    all_raw_files: dict[str, str | bytes] = {}
    declared_dependencies: list[str] = []

    for donor in donors:
        raw_files: dict[str, str | bytes] = {}
        try:
            raw_map = materialize_pinned_donor(donor)
            if not raw_map and donor.files:
                return False, {
                    "compile_passed": False,
                    "error": (
                        "DONOR_MATERIALIZATION_FAILED: "
                        f"{donor.repository}@{donor.commit_sha}"
                    ),
                }
            for rel_path, raw_bytes in raw_map.items():
                try:
                    raw_files[rel_path] = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raw_files[rel_path] = raw_bytes
        except Exception as exc:
            return False, {
                "compile_passed": False,
                "error": (
                    f"DONOR_MATERIALIZATION_ERROR: {donor.repository}@"
                    f"{donor.commit_sha} - {exc}"
                ),
            }

        declared_dependencies.extend(donor.required_dependencies)
        declared_dependencies.extend(parse_donor_build_metadata(raw_files))
        for rel_path, content in raw_files.items():
            if rel_path not in all_raw_files:
                all_raw_files[rel_path] = content

        files, _ = apply_deterministic_adapters(raw_files, target_context)
        for rel_path, content in files.items():
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
    minecraft_version = str(
        target_context.get("minecraft_version") or "1.21.1"
    )
    dependency_receipts: list[DependencyResolutionReceipt] = []
    unresolved_dependencies: list[str] = []
    resolved_coordinates: list[str] = []
    for dependency in dict.fromkeys(declared_dependencies):
        resolution = resolve_dependency_for_target(
            dependency,
            target_loader=loader,
            target_minecraft=minecraft_version,
        )
        dependency_receipts.append(resolution)
        if not resolution.is_resolved or not resolution.resolved_coordinate:
            unresolved_dependencies.append(
                f"{dependency}:{resolution.resolution_reason}"
            )
        else:
            resolved_coordinates.append(resolution.resolved_coordinate)

    if unresolved_dependencies:
        return False, {
            "compile_passed": False,
            "error": "UNRESOLVED_JOINT_DEPENDENCIES",
            "unresolved_dependencies": unresolved_dependencies,
            "resolved_dependencies": [
                receipt.to_dict() for receipt in dependency_receipts
            ],
        }

    if callable(compile_checker):
        result = compile_checker(all_adapted_files, target_context)
        passed = (
            bool(result.get("compile_passed"))
            if isinstance(result, Mapping)
            else bool(result)
        )
        return (
            passed,
            result if isinstance(result, Mapping) else {"compile_passed": passed},
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_path = Path(temp_dir)
        apply_verified_scaffold(sandbox_path, target_context)

        if resolved_coordinates:
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
                injected, _ = DependencyAdaptationPlan.inject_dependencies_into_build_gradle(
                    build_text,
                    tuple(dict.fromkeys(resolved_coordinates)),
                    loader=loader,
                    minecraft_version=minecraft_version,
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
        payload["joint_artifact_hash"] = "sha256:" + hashlib.sha256(
            "".join(
                f"{path}:{hashlib.sha256((value.encode('utf-8') if isinstance(value, str) else value)).hexdigest()}"
                for path, value in sorted(all_adapted_files.items())
            ).encode("utf-8")
        ).hexdigest()
        return build_receipt.compile_passed, payload


def _score_composition_beam(
    combo: Sequence[DonorSlice],
    total_caps: int,
) -> float:
    if not combo:
        return 0.0
    covered = len({donor.capability for donor in combo})
    coverage_score = (covered / max(1, total_caps)) * 100.0
    confidence_score = sum(donor.confidence for donor in combo) * 10.0
    proof_score = sum(
        20.0 if donor.closure_complete else 5.0 for donor in combo
    )
    cost_penalty = sum(donor.adaptation_cost for donor in combo) * 5.0
    donor_count_penalty = len(combo) * 2.0
    return (
        coverage_score
        + confidence_score
        + proof_score
        - cost_penalty
        - donor_count_penalty
    )


def search_ranked_donor_composition_beams(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 6,
) -> tuple[CompositionResult, ...]:
    """Return statically valid, complete donor beams ranked by proof-quality score."""

    if not candidates_by_capability:
        return (CompositionResult(is_valid=True),)
    caps = list(candidates_by_capability)
    if any(not candidates_by_capability[cap] for cap in caps):
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
        )
        if result.is_valid:
            results.append(result)
    return tuple(results)


def search_best_donor_composition(
    candidates_by_capability: Mapping[str, Sequence[DonorSlice]],
    *,
    target_loader: str = "fabric",
    target_minecraft: str = "1.21.1",
    beam_width: int = 4,
) -> CompositionResult:
    ranked = search_ranked_donor_composition_beams(
        candidates_by_capability,
        target_loader=target_loader,
        target_minecraft=target_minecraft,
        beam_width=beam_width,
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
    """Generate one provenance SBOM for legacy donors and canonical bundles."""

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
        bundle_dict = (
            bundle.to_dict() if hasattr(bundle, "to_dict") else dict(bundle)
        )
        bundle_values.append(bundle_dict)
        provenance = getattr(bundle, "provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        file_hashes = getattr(bundle, "file_hashes", {})
        protected_paths = getattr(bundle, "protected_paths", ())
        for path in protected_paths:
            manifest_files.append(
                {
                    "path": path,
                    "origin_repo": provenance.get(
                        "repository",
                        getattr(bundle, "source_ref", ""),
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
