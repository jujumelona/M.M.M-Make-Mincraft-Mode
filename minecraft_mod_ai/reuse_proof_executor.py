from __future__ import annotations

"""Executable reuse proof engine and multi-candidate fallback loop.

Verified reuse is never awarded from metadata, fuzzy test names, or aggregate test
success. Compilation must execute in an isolated target build environment and
behavior proof additionally requires an MMM-owned, implementation-bound acceptance
test with an exact JUnit XML identity and individual PASS result.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_dependency_graph import ArtifactDependencyGraph
from .build_model import BuildModel
from .proof_level import ProofLevel, validate_proof_transition
from .reuse_adapters import AdapterReceipt, apply_deterministic_adapters
from .reuse_license import is_reusable_source_license
from .source_transplant import (
    DonorSlice,
    SourceTransplantError,
    validate_donor_slice_manifest,
)


class ReuseTargetWorkspaceError(RuntimeError):
    """Target workspace could not be copied into the isolated proof sandbox."""


@dataclass(frozen=True)
class ResidualWorkOrder:
    capability: str
    reused_classes: tuple[str, ...] = ()
    reused_symbols: tuple[str, ...] = ()
    missing_interfaces: tuple[str, ...] = ()
    missing_resources: tuple[str, ...] = ()
    unbound_registries: tuple[str, ...] = ()
    glue_code_requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "reused_classes": list(self.reused_classes),
            "reused_symbols": list(self.reused_symbols),
            "missing_interfaces": list(self.missing_interfaces),
            "missing_resources": list(self.missing_resources),
            "unbound_registries": list(self.unbound_registries),
            "glue_code_requirements": list(self.glue_code_requirements),
        }


@dataclass(frozen=True)
class ReuseProofReceipt:
    candidate_id: str
    capability: str
    commit_sha: str
    closure_hash: str
    proof_level: str
    compile_passed: bool
    tests_passed: bool
    unresolved_symbols: tuple[str, ...]
    missing_resources: tuple[str, ...]
    adaptations_applied: tuple[AdapterReceipt, ...]
    verified_capabilities: tuple[str, ...]
    residual_capabilities: tuple[str, ...]
    authoritative_compile: bool = False
    failure_scope: str = ""
    failure_code: str = ""
    failure_message: str = ""
    dependency_receipts: tuple[Mapping[str, Any], ...] = ()
    verified_artifacts: tuple[str, ...] = ()
    residual_artifacts: tuple[str, ...] = ()
    verified_symbols: tuple[str, ...] = ()
    residual_symbols: tuple[str, ...] = ()
    tests_executed: int = 0
    tests_passed_count: int = 0
    capability_acceptance_tests: tuple[str, ...] = ()
    matched_capability_tests: tuple[str, ...] = ()
    requirement_acceptance_map: tuple[tuple[str, str, str, bool], ...] = ()
    host_test_source_hash: str = ""
    host_test_sources: tuple[str, ...] = ()
    exact_host_test_ids: tuple[str, ...] = ()
    work_order: ResidualWorkOrder | None = None
    contract: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/reuse-proof-receipt-v1",
            "candidate_id": self.candidate_id,
            "capability": self.capability,
            "commit_sha": self.commit_sha,
            "closure_hash": self.closure_hash,
            "proof_level": self.proof_level,
            "compile_passed": self.compile_passed,
            "tests_passed": self.tests_passed,
            "unresolved_symbols": list(self.unresolved_symbols),
            "missing_resources": list(self.missing_resources),
            "adaptations_applied": [a.to_dict() for a in self.adaptations_applied],
            "verified_capabilities": list(self.verified_capabilities),
            "residual_capabilities": list(self.residual_capabilities),
            "authoritative_compile": self.authoritative_compile,
            "failure_scope": self.failure_scope,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "dependency_receipts": [dict(item) for item in self.dependency_receipts],
            "verified_artifacts": list(self.verified_artifacts),
            "residual_artifacts": list(self.residual_artifacts),
            "verified_symbols": list(self.verified_symbols),
            "residual_symbols": list(self.residual_symbols),
            "tests_executed": self.tests_executed,
            "tests_passed_count": self.tests_passed_count,
            "capability_acceptance_tests": list(self.capability_acceptance_tests),
            "matched_capability_tests": list(self.matched_capability_tests),
            "requirement_acceptance_map": [
                list(item) for item in self.requirement_acceptance_map
            ],
            "host_test_source_hash": self.host_test_source_hash,
            "host_test_sources": list(self.host_test_sources),
            "exact_host_test_ids": list(self.exact_host_test_ids),
            "work_order": self.work_order.to_dict() if self.work_order else None,
            "contract": (
                self.contract.to_dict()
                if hasattr(self.contract, "to_dict")
                else self.contract
            ),
        }


def _closure_sha256(donor_slice: DonorSlice) -> str:
    payload = [
        [item.path, item.blob_sha, item.sha256, item.size_bytes]
        for item in sorted(donor_slice.files, key=lambda entry: entry.path)
    ]
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _residual_java_artifact_path(
    symbol: str,
    target_context: Mapping[str, Any],
) -> str:
    """Map one declared residual Java symbol to its exact target-owned source path."""

    raw = str(symbol or "").strip().replace("$", ".")
    simple_name = raw.rsplit(".", 1)[-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", simple_name):
        return ""
    target_package = str(
        target_context.get("target_package") or "ai.minecraft.generated.mod"
    ).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", target_package):
        return ""
    return f"src/main/java/{target_package.replace('.', '/')}/{simple_name}.java"


def _residual_resource_artifact_path(
    path: str,
    target_context: Mapping[str, Any],
) -> str:
    normalized = _safe_workspace_relative_path(path)
    if not normalized:
        return ""
    original = normalized
    if normalized.startswith("src/main/resources/"):
        normalized = normalized.removeprefix("src/main/resources/")
    if not normalized.startswith(("assets/", "data/")):
        return original if original.startswith("src/") else ""
    kind, _source_namespace, *rest = normalized.split("/")
    if not rest:
        return ""
    target_modid = str(target_context.get("target_modid") or "generated_mod").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_.-]*", target_modid):
        return ""
    return f"src/main/resources/{kind}/{target_modid}/" + "/".join(rest)


def _safe_workspace_relative_path(path: Any) -> str:
    """Return a canonical in-workspace path or reject an unsafe spelling."""

    raw = str(path or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return ""
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return ""
    return "/".join(parts)


def _existing_workspace_hashes(
    target_workspace: str | Path,
    paths: Sequence[str],
) -> dict[str, str]:
    """Bind residual replacements to target bytes observed during proof."""

    raw_root = str(target_workspace or "").strip()
    if not raw_root:
        return {}
    try:
        root = Path(raw_root).expanduser().resolve()
    except (OSError, RuntimeError):
        return {}
    if not root.is_dir() or root.is_symlink():
        return {}

    hashes: dict[str, str] = {}
    for raw_path in paths:
        normalized = _safe_workspace_relative_path(raw_path)
        if not normalized:
            continue
        target = root.joinpath(*normalized.split("/"))
        try:
            resolved = target.resolve()
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not resolved.is_file() or resolved.is_symlink():
            continue
        try:
            hashes[normalized] = "sha256:" + hashlib.sha256(
                resolved.read_bytes()
            ).hexdigest()
        except OSError:
            continue
    return hashes


def _sandbox_destination(root: Path, relative_path: Any) -> Path:
    normalized = _safe_workspace_relative_path(relative_path)
    if not normalized:
        raise ReuseTargetWorkspaceError("Reuse proof artifact path is unsafe.")
    root_resolved = root.resolve()
    destination = root.joinpath(*normalized.split("/"))
    try:
        destination.resolve(strict=False).relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ReuseTargetWorkspaceError("Reuse proof artifact escaped its sandbox.") from exc
    return destination


def scaffold_minimal_ephemeral_workspace(
    sandbox_path: Path,
    target_context: Mapping[str, Any],
) -> None:
    """Synthesize verified build files and a real Gradle wrapper template."""

    from .verified_scaffold_registry import apply_verified_scaffold

    apply_verified_scaffold(sandbox_path, target_context)


def _dependency_receipt_value(receipt: Any, key: str, default: Any = "") -> Any:
    if isinstance(receipt, Mapping):
        return receipt.get(key, default)
    return getattr(receipt, key, default)


def _render_proof_build_model(
    sandbox_path: Path,
    target_context: Mapping[str, Any],
    dependency_receipts: Sequence[Any],
) -> None:
    """Render the proof build through the same authoritative model as assembly."""

    model = BuildModel.for_target_context(target_context)
    for receipt in dependency_receipts:
        if not bool(_dependency_receipt_value(receipt, "is_resolved", False)):
            raise ValueError("Unresolved dependency cannot enter the proof build model.")
        repository = str(_dependency_receipt_value(receipt, "repository", "")).strip()
        coordinate = str(
            _dependency_receipt_value(receipt, "resolved_coordinate", "")
        ).strip()
        configuration = str(
            _dependency_receipt_value(receipt, "gradle_configuration", "")
        ).strip()
        fingerprint = str(
            _dependency_receipt_value(receipt, "resolution_fingerprint", "")
        ).strip()
        if not repository or not coordinate or not configuration:
            raise ValueError("Resolved dependency receipt lacks authoritative Gradle fields.")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):
            raise ValueError("Resolved dependency receipt lacks an authoritative fingerprint.")
        model.add_repository(repository)
        model.add_dependency(
            coordinate,
            configuration,
            sha256=str(_dependency_receipt_value(receipt, "artifact_hash", "")),
        )

    kotlin_build = sandbox_path / "build.gradle.kts"
    if kotlin_build.exists() or kotlin_build.is_symlink():
        kotlin_build.unlink()
    (sandbox_path / "build.gradle").write_text(
        model.render_gradle(
            modid=str(target_context.get("target_modid") or "generated_mod")
        ),
        encoding="utf-8",
    )


def _compute_dependency_closed_subgraphs(
    adapted_files: Mapping[str, Any],
    donor_slice: DonorSlice,
) -> list[list[str]]:
    """Return directional closures using repository-time graph evidence when present.

    Real donor discovery computes artifact nodes/edges before the slice is materialized.
    That pre-slice graph is authoritative here. Re-parsing adapted files is retained
    only as a compatibility fallback for synthetic unit-test donors without graph
    receipts.
    """

    if not adapted_files:
        return []

    available_paths = set(adapted_files)
    if donor_slice.artifact_nodes:
        graph = ArtifactDependencyGraph()
        for donor_node in donor_slice.artifact_nodes:
            if donor_node.id not in available_paths:
                continue
            graph.add_node(donor_node)
        for edge in donor_slice.artifact_edges:
            if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
                graph.add_edge(edge.source_id, edge.target_id)
        for edge in donor_slice.unresolved_edges:
            if edge.source_id in graph.nodes:
                from .artifact_dependency_graph import UnresolvedArtifactEdge

                graph.unresolved_edges.append(
                    UnresolvedArtifactEdge(
                        source_id=edge.source_id,
                        requested_target=edge.requested_target,
                        relation=edge.relation,
                        reason=edge.reason,
                    )
                )
        return graph.compute_directional_closures()

    sym_map = {
        donor_file.path: donor_file.symbols
        for donor_file in donor_slice.files
        if donor_file.path in adapted_files
    }
    graph = ArtifactDependencyGraph.build_from_files(
        adapted_files,
        known_symbols=sym_map,
    )
    return graph.compute_directional_closures()


def _symbols_for_artifacts(
    donor_slice: DonorSlice,
    artifact_paths: Sequence[str],
) -> tuple[str, ...]:
    allowed = set(artifact_paths)
    return tuple(
        dict.fromkeys(
            symbol
            for donor_file in donor_slice.files
            if donor_file.path in allowed
            for symbol in donor_file.symbols
            if symbol
        )
    )


def _rejected_receipt(
    donor_slice: DonorSlice,
    *,
    candidate_id: str,
    closure_hash: str,
    failure_code: str,
    failure_message: str = "",
    failure_scope: str = "donor",
    proof_level: ProofLevel = ProofLevel.DISCOVERED,
    missing_resources: Sequence[str] = (),
) -> ReuseProofReceipt:
    return ReuseProofReceipt(
        candidate_id=candidate_id,
        capability=donor_slice.capability,
        commit_sha=donor_slice.commit_sha,
        closure_hash=closure_hash,
        proof_level=proof_level.value,
        compile_passed=False,
        tests_passed=False,
        unresolved_symbols=(),
        missing_resources=tuple(missing_resources),
        adaptations_applied=(),
        verified_capabilities=(),
        residual_capabilities=(donor_slice.capability,),
        failure_scope=failure_scope,
        failure_code=failure_code,
        failure_message=failure_message,
    )


def execute_reuse_proof(
    donor_slice: DonorSlice,
    *,
    target_workspace: str | Path,
    target_context: Mapping[str, Any],
    discovery_client: Any = None,
    compile_checker: Any = None,
    run_tests: bool = True,
) -> ReuseProofReceipt:
    """Materialize one donor slice and execute fail-closed reuse verification."""

    candidate_id = f"{donor_slice.repository}@{donor_slice.commit_sha}"
    closure_hash = _closure_sha256(donor_slice)
    current_level = ProofLevel.DISCOVERED

    try:
        validate_donor_slice_manifest(donor_slice)
    except SourceTransplantError as exc:
        return _rejected_receipt(
            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash,
            failure_code="MANIFEST_INVALID", failure_message=str(exc),
        )

    if not is_reusable_source_license(donor_slice.license_id):
        return _rejected_receipt(
            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash,
            failure_code="LICENSE_NOT_REUSABLE",
            failure_message=f"License {donor_slice.license_id!r} is not approved for source reuse.",
        )

    valid, _ = validate_proof_transition(
        current_level,
        ProofLevel.LICENSE_VERIFIED,
        receipt={"license": donor_slice.license_id},
    )
    if not valid:
        return _rejected_receipt(
            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash,
            failure_code="LICENSE_PROOF_INVALID",
        )
    current_level = ProofLevel.LICENSE_VERIFIED

    valid, _ = validate_proof_transition(
        current_level,
        ProofLevel.PINNED,
        receipt={"commit_sha": donor_slice.commit_sha},
    )
    if not valid:
        return _rejected_receipt(
            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash,
            proof_level=current_level, failure_code="PIN_INVALID",
        )
    current_level = ProofLevel.PINNED

    if donor_slice.closure_complete:
        valid, _ = validate_proof_transition(
            current_level,
            ProofLevel.CLOSURE_COMPLETE,
            receipt={"closure_complete": True},
        )
        if valid:
            current_level = ProofLevel.CLOSURE_COMPLETE

    from .source_transplant import materialize_pinned_donor

    in_memory_files: dict[str, str | bytes] = {}
    materialization_failed = False
    materialization_error = ""
    try:
        raw_map = materialize_pinned_donor(
            donor_slice,
            discovery_client=discovery_client,
        )
        for rel_path, raw_bytes in raw_map.items():
            try:
                in_memory_files[rel_path] = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                in_memory_files[rel_path] = raw_bytes
    except SourceTransplantError as exc:
        if not in_memory_files:
            materialization_failed = True
            materialization_error = str(exc)

    if materialization_failed:
        return _rejected_receipt(
            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash,
            proof_level=current_level, failure_code="MATERIALIZATION_FAILED",
            failure_message=materialization_error,
            missing_resources=tuple(
                edge.requested_target for edge in donor_slice.unresolved_edges
            ),
        )


    adapted_files, adapter_receipts = apply_deterministic_adapters(
        in_memory_files,
        target_context,
    )

    if adapted_files:
        valid, _ = validate_proof_transition(
            current_level,
            ProofLevel.MATERIALIZED,
            receipt={"files": len(adapted_files)},
        )
        if valid:
            current_level = ProofLevel.MATERIALIZED

    from .dependency_resolver import (
        parse_donor_build_metadata,
        resolve_dependency_for_target,
    )

    loader = str(target_context.get("loader") or "fabric")
    mc_ver = str(target_context.get("minecraft_version") or "1.21.1")
    donor_declared_deps = parse_donor_build_metadata(in_memory_files)
    all_needed_deps = tuple(
        dict.fromkeys(
            list(donor_slice.required_dependencies) + list(donor_declared_deps)
        )
    )
    resolved_dependencies: list[Any] = []
    unresolved_mandatory_deps: list[str] = []
    for dependency in all_needed_deps:
        dep_receipt = resolve_dependency_for_target(
            dependency,
            target_loader=loader,
            target_minecraft=mc_ver,
        )
        resolved_dependencies.append(dep_receipt)
        if not dep_receipt.is_resolved:
            unresolved_mandatory_deps.append(
                f"{dependency}:{dep_receipt.resolution_reason}"
            )

    import shutil
    import tempfile

    compile_passed = False
    tests_passed = False
    unresolved_symbols: list[str] = list(unresolved_mandatory_deps)
    missing_resources: list[str] = [
        edge.requested_target for edge in donor_slice.unresolved_edges
    ]
    all_receipts = list(adapter_receipts)
    generated_tests: dict[str, str] = {}
    test_source_hash = ""
    dependency_injection_failed = False
    authoritative_compile_execution = not callable(compile_checker)

    with tempfile.TemporaryDirectory() as sandbox_dir:
        sandbox_path = Path(sandbox_dir)
        ws_path = None
        if not callable(compile_checker) and target_workspace:
            possible_workspace = Path(target_workspace)
            if possible_workspace.exists():
                ws_path = possible_workspace

        if ws_path and ws_path.is_dir():

            def ignore_patterns(_path: str, names: Sequence[str]) -> set[str]:
                ignored = {
                    ".git",
                    ".gradle",
                    "build",
                    ".idea",
                    ".vscode",
                    ".gemini",
                    "__pycache__",
                    "cache",
                }
                return {name for name in names if name in ignored}

            try:
                shutil.copytree(
                    ws_path,
                    sandbox_path,
                    ignore=ignore_patterns,
                    dirs_exist_ok=True,
                )
            except (OSError, shutil.Error) as exc:
                raise ReuseTargetWorkspaceError(
                    "Failed to copy target workspace into reuse proof sandbox."
                ) from exc

        exact_dependency_receipts = tuple(
            receipt for receipt in resolved_dependencies if receipt.is_resolved
        )
        if authoritative_compile_execution:
            scaffold_minimal_ephemeral_workspace(sandbox_path, target_context)
            try:
                _render_proof_build_model(
                    sandbox_path,
                    target_context,
                    exact_dependency_receipts,
                )
            except (OSError, RuntimeError, ValueError, ImportError) as inj_err:
                dependency_injection_failed = True
                reason = f"BUILD_MODEL_RENDER_FAILED: {inj_err}"
                unresolved_symbols.append(reason)
                unresolved_mandatory_deps.append(reason)

        for rel_path, content in adapted_files.items():
            dest = _sandbox_destination(sandbox_path, rel_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                dest.write_bytes(content)
            else:
                dest.write_text(str(content), encoding="utf-8")

        from .acceptance_contracts import materialize_host_acceptance_tests

        generated_tests, test_source_hash = materialize_host_acceptance_tests(
            sandbox_path,
            donor_slice.capability,
        )

        tests_executed = 0
        tests_passed_count = 0
        executed_test_ids: tuple[str, ...] = ()
        individual_results: Mapping[str, bool] = {}

        if callable(compile_checker):
            check_result = compile_checker(adapted_files, target_context)
            if isinstance(check_result, Mapping):
                compile_passed = bool(check_result.get("compile_passed"))
                tests_passed = bool(check_result.get("tests_passed"))
                tests_executed = int(check_result.get("tests_executed", 0))
                tests_passed_count = int(check_result.get("tests_passed_count", 0))
                executed_test_ids = tuple(check_result.get("executed_test_ids") or ())
                individual_results = dict(
                    check_result.get("individual_test_results") or {}
                )
                unresolved_symbols.extend(check_result.get("unresolved_symbols") or [])
                missing_resources.extend(check_result.get("missing_resources") or [])
            else:
                compile_passed = bool(check_result)
                tests_passed = False
        else:
            from .reuse_build_verifier import verify_scratch_workspace_build

            build_receipt = verify_scratch_workspace_build(
                sandbox_path,
                run_tests=run_tests,
            )
            compile_passed = build_receipt.compile_passed
            tests_passed = build_receipt.tests_passed
            tests_executed = build_receipt.tests_executed
            tests_passed_count = build_receipt.tests_passed_count
            executed_test_ids = build_receipt.executed_test_ids
            individual_results = build_receipt.individual_test_results
            unresolved_symbols.extend(build_receipt.unresolved_symbols)
            missing_resources.extend(build_receipt.missing_resources)

    from .acceptance_contracts import get_host_acceptance_contracts

    req_contracts = get_host_acceptance_contracts(donor_slice.capability)
    matched_tests: list[str] = []
    acceptance_map: list[tuple[str, str, str, bool]] = []
    exact_host_test_ids: list[str] = []
    authoritative_host_execution = authoritative_compile_execution

    for acceptance_contract in req_contracts:
        expected_id = (
            "ai.minecraft.acceptance."
            f"{acceptance_contract.host_test_class}."
            f"{acceptance_contract.host_test_method}"
        )
        source_path = (
            "src/test/java/ai/minecraft/acceptance/"
            f"{acceptance_contract.host_test_class}.java"
        )
        exact_host_test_ids.append(expected_id)
        source_materialized = source_path in generated_tests
        implementation_bound = (
            getattr(acceptance_contract, "implementation_bound", False) is True
        )
        exact_executed = expected_id in executed_test_ids
        exact_individual_pass = individual_results.get(expected_id) is True
        is_passed = bool(
            authoritative_host_execution
            and source_materialized
            and implementation_bound
            and exact_executed
            and exact_individual_pass
        )
        matched_tid = expected_id if exact_executed else "none"
        acceptance_map.append(
            (
                acceptance_contract.requirement_id,
                acceptance_contract.description,
                matched_tid,
                is_passed,
            )
        )
        if is_passed:
            matched_tests.append(expected_id)

    matched_capability_tests = tuple(dict.fromkeys(matched_tests))
    requirement_acceptance_map = tuple(acceptance_map)

    unresolved_set = set(unresolved_symbols)
    donor_symbols_by_path = {
        donor_file.path: set(donor_file.symbols)
        for donor_file in donor_slice.files
    }
    verified_art_list: list[str] = []
    residual_art_list: list[str] = []
    verified_subgraph_count = 0

    if (
        authoritative_compile_execution
        and compile_passed
        and donor_slice.closure_complete
        and not unresolved_mandatory_deps
    ):
        verified_art_list.extend(adapted_files.keys())
    else:
        subgraphs = _compute_dependency_closed_subgraphs(adapted_files, donor_slice)
        for component in subgraphs:
            comp_has_error = False
            for path in component:
                content = adapted_files.get(path, "")
                text_content = (
                    content
                    if isinstance(content, str)
                    else content.decode("utf-8", errors="ignore")
                )
                donor_symbols = donor_symbols_by_path.get(path, set())
                if (
                    any(symbol in text_content for symbol in unresolved_set if symbol)
                    or any(symbol in donor_symbols for symbol in unresolved_set if symbol)
                    or any(
                        symbol.casefold() in path.casefold()
                        for symbol in unresolved_set
                        if symbol
                    )
                ):
                    comp_has_error = True
                    break

            if comp_has_error:
                residual_art_list.extend(component)
                continue

            comp_files = {path: adapted_files[path] for path in component}
            comp_passed = False
            if callable(compile_checker):
                # Caller-supplied checkers are diagnostic-only and cannot mint
                # reusable subgraph proof. Avoid repeating the diagnostic per subgraph.
                residual_art_list.extend(component)
                continue
            else:
                with tempfile.TemporaryDirectory(prefix="mmm_subgraph_") as sub_tmp:
                    sub_path = Path(sub_tmp)
                    scaffold_minimal_ephemeral_workspace(
                        sub_path,
                        target_context=target_context,
                    )
                    for relative_path, content in comp_files.items():
                        destination = _sandbox_destination(sub_path, relative_path)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(content, bytes):
                            destination.write_bytes(content)
                        else:
                            destination.write_text(str(content), encoding="utf-8")
                    _render_proof_build_model(
                        sub_path,
                        target_context,
                        exact_dependency_receipts,
                    )
                    from .reuse_build_verifier import verify_scratch_workspace_build

                    sub_receipt = verify_scratch_workspace_build(
                        sub_path,
                        run_tests=False,
                    )
                    comp_passed = sub_receipt.compile_passed

            if comp_passed:
                verified_subgraph_count += 1
                verified_art_list.extend(component)
            else:
                residual_art_list.extend(component)

    verified_artifacts = tuple(dict.fromkeys(verified_art_list))
    residual_artifacts = tuple(dict.fromkeys(residual_art_list))
    verified_symbols = _symbols_for_artifacts(donor_slice, verified_artifacts)
    residual_symbols_from_artifacts = _symbols_for_artifacts(
        donor_slice,
        residual_artifacts,
    )
    residual_symbols = tuple(
        dict.fromkeys(
            list(unresolved_symbols) + list(residual_symbols_from_artifacts)
        )
    )

    all_contracts_implementation_bound = bool(req_contracts) and all(
        getattr(contract, "implementation_bound", False) is True
        for contract in req_contracts
    )
    has_full_acceptance = bool(
        authoritative_host_execution
        and req_contracts
        and test_source_hash
        and all_contracts_implementation_bound
        and len(generated_tests) == len(req_contracts)
        and all(item[3] for item in acceptance_map)
        and tests_executed >= len(req_contracts)
        and tests_passed
    )

    if (
        authoritative_compile_execution
        and compile_passed
        and donor_slice.closure_complete
        and not unresolved_mandatory_deps
        and not dependency_injection_failed
    ):
        valid, _ = validate_proof_transition(
            current_level,
            ProofLevel.COMPILE_VERIFIED,
            receipt={"compile_passed": True, "authoritative_compile": True},
        )
        if valid:
            current_level = ProofLevel.COMPILE_VERIFIED
            if has_full_acceptance:
                behavior_receipt = {
                    "acceptance_passed": True,
                    "count": len(matched_tests),
                    "implementation_bound": True,
                    "exact_results": True,
                    "test_source_hash": test_source_hash,
                }
                behavior_valid, _ = validate_proof_transition(
                    current_level,
                    ProofLevel.BEHAVIOR_VERIFIED,
                    receipt=behavior_receipt,
                )
                if behavior_valid:
                    current_level = ProofLevel.BEHAVIOR_VERIFIED
        verified_caps = (
            (donor_slice.capability,) if current_level.is_verified() else ()
        )
        residual_caps = (
            () if current_level.is_verified() else (donor_slice.capability,)
        )
    elif verified_subgraph_count > 0 and len(verified_artifacts) > 0 and (
        residual_artifacts
        or unresolved_symbols
        or not donor_slice.closure_complete
    ):
        subgraph_valid, _ = validate_proof_transition(
            current_level, ProofLevel.SUBGRAPH_COMPILE_VERIFIED,
            receipt={"verified_subgraphs": verified_subgraph_count, "authoritative_compile": True},
        )
        if subgraph_valid:
            current_level = ProofLevel.SUBGRAPH_COMPILE_VERIFIED
            partial_valid, _ = validate_proof_transition(
                current_level,
                ProofLevel.PARTIAL_REUSE,
                receipt={"partial": True},
            )
            if partial_valid:
                current_level = ProofLevel.PARTIAL_REUSE
        verified_caps = ()
        residual_caps = (donor_slice.capability,)
    elif adapted_files:
        current_level = ProofLevel.MATERIALIZED
        verified_caps = ()
        residual_caps = (donor_slice.capability,)
    else:
        current_level = ProofLevel.FRESH_REQUIRED
        verified_caps = ()
        residual_caps = (donor_slice.capability,)

    reused_classes = tuple(
        path
        for path in verified_artifacts
        if path.endswith((".java", ".kt"))
    )
    missing_res = tuple(dict.fromkeys(missing_resources))
    unbound_registries = tuple(
        symbol for symbol in unresolved_symbols if ":" in symbol
    )
    missing_interfaces = tuple(
        symbol for symbol in residual_symbols if ":" not in symbol
    )
    glue_requirements = (
        (
            f"Integrate {len(verified_artifacts)} reused artifacts with host "
            f"{donor_slice.capability} lifecycle"
        ),
    ) if verified_artifacts else ()

    work_order = ResidualWorkOrder(
        capability=donor_slice.capability,
        reused_classes=reused_classes,
        reused_symbols=verified_symbols,
        missing_interfaces=missing_interfaces,
        missing_resources=missing_res,
        unbound_registries=unbound_registries,
        glue_code_requirements=glue_requirements,
    )

    from .residual_generation_contract import (
        GlueContract,
        RegistryRequirement,
        ResidualGenerationContract,
        ResourceRequirement,
    )

    protected_hashes: dict[str, str] = {}
    for path in verified_artifacts:
        content = adapted_files.get(path, "")
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        protected_hashes[path] = "sha256:" + hashlib.sha256(
            content_bytes
        ).hexdigest()

    resource_requirements = tuple(
        ResourceRequirement(
            logical_id=missing_resource,
            resource_type="data" if "data/" in missing_resource else "texture",
            target_path=missing_resource,
        )
        for missing_resource in missing_res
    )
    registry_requirements = tuple(
        RegistryRequirement(
            registry_key="minecraft:custom",
            entry_id=registry,
            backing_class="",
        )
        for registry in unbound_registries
    )
    glue_contracts = tuple(
        GlueContract(
            target_symbol=symbol,
            caller_symbol=donor_slice.capability,
            purpose="Lifecycle integration",
        )
        for symbol in verified_symbols
    )

    exact_residual_paths = {
        normalized
        for path in residual_artifacts
        if (normalized := _safe_workspace_relative_path(path))
    }
    exact_residual_paths.update(
        path
        for path in (
            _residual_resource_artifact_path(item, target_context)
            for item in missing_res
        )
        if path
    )
    exact_residual_paths.update(
        path
        for path in (
            _residual_java_artifact_path(symbol, target_context)
            for symbol in missing_interfaces
        )
        if path
    )
    exact_residual_paths.difference_update(protected_hashes)
    expected_old_sha256 = _existing_workspace_hashes(
        target_workspace,
        tuple(sorted(exact_residual_paths)),
    )
    allowed_write_paths = tuple(sorted(expected_old_sha256))
    required_new_artifacts = tuple(
        path
        for path in sorted(exact_residual_paths)
        if path not in expected_old_sha256
    )

    residual_contract = ResidualGenerationContract(
        capability=donor_slice.capability,
        requirement_ids=(donor_slice.capability,),
        protected_artifacts=protected_hashes,
        protected_symbols=verified_symbols,
        allowed_write_paths=allowed_write_paths,
        expected_old_sha256=expected_old_sha256,
        allowed_create_prefixes=(),
        required_new_artifacts=required_new_artifacts,
        required_symbols=residual_symbols,
        required_interfaces=missing_interfaces,
        required_resource_edges=resource_requirements,
        required_registry_bindings=registry_requirements,
        glue_contracts=glue_contracts,
    )

    failure_scope = ""
    failure_code = ""
    failure_message = ""
    if not current_level.allows_reuse():
        if not authoritative_compile_execution:
            failure_scope = "verification"
            failure_code = "NON_AUTHORITATIVE_COMPILE_CHECKER"
            failure_message = "Caller-supplied compile_checker is diagnostic-only."
        elif dependency_injection_failed:
            failure_scope = "dependency"
            failure_code = "BUILD_MODEL_RENDER_FAILED"
        elif unresolved_mandatory_deps:
            failure_scope = "dependency"
            failure_code = "DEPENDENCY_UNRESOLVED"
        elif not adapted_files:
            failure_scope = "donor"
            failure_code = "NO_MATERIALIZED_ARTIFACTS"
        elif not compile_passed:
            failure_scope = "verification"
            failure_code = "COMPILE_FAILED"

    return ReuseProofReceipt(
        candidate_id=candidate_id,
        capability=donor_slice.capability,
        commit_sha=donor_slice.commit_sha,
        closure_hash=closure_hash,
        proof_level=current_level.value,
        compile_passed=(
            authoritative_compile_execution
            and compile_passed
            and donor_slice.closure_complete
            and not unresolved_mandatory_deps
            and not dependency_injection_failed
        ),
        tests_passed=has_full_acceptance,
        unresolved_symbols=tuple(dict.fromkeys(unresolved_symbols)),
        missing_resources=missing_res,
        adaptations_applied=tuple(all_receipts),
        verified_capabilities=verified_caps,
        residual_capabilities=residual_caps,
        authoritative_compile=authoritative_compile_execution,
        failure_scope=failure_scope,
        failure_code=failure_code,
        failure_message=failure_message,
        dependency_receipts=tuple(
            receipt.to_dict() for receipt in resolved_dependencies
        ),
        verified_artifacts=verified_artifacts,
        residual_artifacts=residual_artifacts,
        verified_symbols=verified_symbols,
        residual_symbols=residual_symbols,
        tests_executed=tests_executed,
        tests_passed_count=tests_passed_count,
        capability_acceptance_tests=tuple(exact_host_test_ids),
        matched_capability_tests=matched_capability_tests,
        requirement_acceptance_map=requirement_acceptance_map,
        host_test_source_hash=test_source_hash,
        host_test_sources=tuple(sorted(generated_tests)),
        exact_host_test_ids=tuple(exact_host_test_ids),
        work_order=work_order,
        contract=residual_contract,
    )


def execute_candidate_fallback_loop(
    candidates: Sequence[DonorSlice],
    capability: str,
    *,
    target_workspace: str | Path,
    target_context: Mapping[str, Any],
    discovery_client: Any = None,
    compile_checker: Any = None,
) -> tuple[DonorSlice | None, tuple[ReuseProofReceipt, ...]]:
    """Try donor candidates until full proof or the best partial proof is found."""

    requested_capability = str(capability or "").strip()
    if not requested_capability:
        raise ValueError("capability must be non-empty")

    receipts: list[ReuseProofReceipt] = []
    best_partial: tuple[tuple[int, int, int, int, int], DonorSlice] | None = None

    for candidate in candidates:
        if str(candidate.capability or "").strip() != requested_capability:
            receipts.append(_rejected_receipt(
                candidate, candidate_id=f"{candidate.repository}@{candidate.commit_sha}",
                closure_hash=_closure_sha256(candidate), failure_code="CAPABILITY_MISMATCH",
                failure_message=(f"Candidate capability {candidate.capability!r} does not match "
                                 f"requested capability {requested_capability!r}."),
            ))
            continue

        receipt = execute_reuse_proof(
            candidate,
            target_workspace=target_workspace,
            target_context=target_context,
            discovery_client=discovery_client,
            compile_checker=compile_checker,
        )
        receipts.append(receipt)
        receipt_level = ProofLevel.from_value(receipt.proof_level)
        if receipt.compile_passed and receipt_level.is_verified():
            return candidate, tuple(receipts)
        if receipt_level == ProofLevel.PARTIAL_REUSE:
            partial_score = (
                len(receipt.verified_artifacts),
                len(receipt.verified_symbols),
                -len(receipt.residual_artifacts),
                -len(receipt.unresolved_symbols),
                -len(receipt.missing_resources),
            )
            if best_partial is None or partial_score > best_partial[0]:
                best_partial = (partial_score, candidate)

    if best_partial is not None:
        return best_partial[1], tuple(receipts)

    return None, tuple(receipts)
