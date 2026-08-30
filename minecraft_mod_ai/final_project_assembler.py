from __future__ import annotations

"""Proof-gated assembly of reusable bundles and generated residual work."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .build_model import BuildModel
from .composition_solver import generate_reuse_manifest
from .residual_generation_contract import (
    ResidualGenerationContract,
    validate_residual_write_against_contracts,
)
from .resource_merge_registry import ResourceMergeRegistry
from .reuse_artifacts import (
    BundleMaterializationError,
    ReusableArtifactBundle,
    bundle_proof_allows_reuse,
)
from .reuse_planner import TargetImplementationPlan
from .reuse_proof_executor import ResidualWorkOrder
from .verified_scaffold_registry import apply_verified_scaffold


_BUILD_MODEL_OWNED_PATHS = frozenset({"build.gradle", "build.gradle.kts"})


def _receipt_value(receipt: Any, key: str, default: Any = None) -> Any:
    if isinstance(receipt, Mapping):
        return receipt.get(key, default)
    return getattr(receipt, key, default)


def _receipt_work_order(receipt: Any) -> ResidualWorkOrder | None:
    value = _receipt_value(receipt, "work_order")
    if isinstance(value, ResidualWorkOrder):
        return value
    if isinstance(value, Mapping):
        try:
            return ResidualWorkOrder(
                capability=str(value.get("capability") or ""),
                reused_classes=tuple(value.get("reused_classes") or ()),
                reused_symbols=tuple(value.get("reused_symbols") or ()),
                missing_interfaces=tuple(value.get("missing_interfaces") or ()),
                missing_resources=tuple(value.get("missing_resources") or ()),
                unbound_registries=tuple(value.get("unbound_registries") or ()),
                glue_code_requirements=tuple(
                    value.get("glue_code_requirements") or ()
                ),
            )
        except (TypeError, ValueError):
            return None
    return None


def _normalized_sha256(value: Any) -> str:
    return str(value or "").casefold().removeprefix("sha256:")


@dataclass(frozen=True)
class FinalProjectAssemblyResult:
    project_name: str
    target_loader: str
    target_minecraft: str
    total_files: int
    reused_file_count: int
    residual_file_count: int
    fresh_file_count: int
    staged_paths: tuple[str, ...]
    manifest_sbom: dict[str, Any] = field(default_factory=dict)
    work_orders: tuple[ResidualWorkOrder, ...] = ()
    residual_contracts: tuple[Any, ...] = ()
    is_valid: bool = True
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/final-project-assembly-result-v1",
            "project_name": self.project_name,
            "target_loader": self.target_loader,
            "target_minecraft": self.target_minecraft,
            "total_files": self.total_files,
            "reused_file_count": self.reused_file_count,
            "residual_file_count": self.residual_file_count,
            "fresh_file_count": self.fresh_file_count,
            "staged_paths": list(self.staged_paths),
            "manifest_sbom": self.manifest_sbom,
            "work_orders": [work.to_dict() for work in self.work_orders],
            "residual_contracts": [
                contract.to_dict() if hasattr(contract, "to_dict") else contract
                for contract in self.residual_contracts
            ],
            "is_valid": self.is_valid,
            "errors": list(self.errors),
        }


class FinalProjectAssembler:
    """Canonical assembler for proof-backed reuse and generated files."""

    def __init__(
        self,
        target_workspace: Path | str,
        target_context: Mapping[str, Any],
    ) -> None:
        self.workspace_path = Path(target_workspace)
        self.target_context = dict(target_context)
        self.target_loader = str(
            target_context.get("loader") or "fabric"
        ).casefold().strip()
        self.target_minecraft = str(
            target_context.get("minecraft_version") or "1.21.1"
        ).strip()
        self.target_modid = str(
            target_context.get("target_modid") or "generated_mod"
        ).strip()
        self.target_package = str(
            target_context.get("target_package")
            or "ai.minecraft.generated.mod"
        ).strip()

    def assemble_plan(
        self,
        plan: TargetImplementationPlan,
        *,
        proof_receipts: Mapping[str, Any] | None = None,
        residual_files: Mapping[str, str | bytes] | None = None,
        fresh_files: Mapping[str, str | bytes] | None = None,
    ) -> FinalProjectAssemblyResult:
        """Assemble a plan through its proof-bound reusable bundles only."""
        proof_map = proof_receipts or {}
        decision_receipts: dict[str, Any] = {}
        decision_bundle_receipts: dict[str, Any] = {}
        for decision in plan.capabilities:
            receipt = decision.proof_receipt
            if receipt is not None:
                decision_receipts[decision.capability] = receipt
                bundle = getattr(decision, "artifact_bundle", None)
                if isinstance(bundle, ReusableArtifactBundle):
                    decision_bundle_receipts.setdefault(bundle.bundle_id, receipt)

        bundle_candidates: list[ReusableArtifactBundle] = []
        composition = getattr(plan, "selected_composition", None)
        for bundle in getattr(composition, "bundles", ()) if composition else ():
            if isinstance(bundle, ReusableArtifactBundle):
                bundle_candidates.append(bundle)
        for decision in plan.capabilities:
            bundle = getattr(decision, "artifact_bundle", None)
            if isinstance(bundle, ReusableArtifactBundle):
                bundle_candidates.append(bundle)

        unique_bundles: list[ReusableArtifactBundle] = []
        bundle_receipts: dict[str, Any] = {}
        seen_bundle_ids: set[str] = set()
        for bundle in bundle_candidates:
            if bundle.bundle_id in seen_bundle_ids:
                continue
            seen_bundle_ids.add(bundle.bundle_id)
            receipt = (
                bundle.proof_receipt
                or proof_map.get(bundle.bundle_id)
                or decision_bundle_receipts.get(bundle.bundle_id)
            )
            unique_bundles.append(bundle)
            if receipt is not None:
                bundle_receipts[bundle.bundle_id] = receipt

        work_orders: list[ResidualWorkOrder] = []
        for decision in plan.capabilities:
            capability = decision.capability
            receipt = decision_receipts.get(capability)
            work_order = _receipt_work_order(receipt)
            if work_order is not None and work_order not in work_orders:
                work_orders.append(work_order)

        for receipt in bundle_receipts.values():
            work_order = _receipt_work_order(receipt)
            if work_order is not None and work_order not in work_orders:
                work_orders.append(work_order)

        return self.assemble(
            reused_bundles=unique_bundles,
            bundle_proof_receipts=bundle_receipts,
            residual_files=residual_files,
            fresh_files=fresh_files,
            work_orders=work_orders,
            residual_contracts=tuple(getattr(plan, "residual_contracts", ()) or ()),
        )

    def _stage_file(
        self,
        staged: dict[str, str | bytes],
        norm_path: str,
        content: str | bytes,
        errors: list[str],
        *,
        source_label: str,
    ) -> bool:
        try:
            norm_path = ResourceMergeRegistry.canonical_path(
                norm_path, target_modid=self.target_modid
            )
        except ValueError as exc:
            errors.append(f"{source_label}: {exc}")
            return False
        if norm_path in _BUILD_MODEL_OWNED_PATHS:
            errors.append(f"BUILD_MODEL_OWNED_PATH: {source_label}: {norm_path}")
            return False

        if isinstance(content, bytes) and ResourceMergeRegistry.can_merge(norm_path):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"{source_label}: MERGEABLE_RESOURCE_MUST_BE_UTF8: {norm_path}")
                return False
        if isinstance(content, str) and ResourceMergeRegistry.can_merge(norm_path):
            content, valid, error = ResourceMergeRegistry.normalize(
                norm_path, content, target_modid=self.target_modid
            )
            if not valid:
                errors.append(f"{source_label}: {error}")
                return False

        if norm_path not in staged:
            staged[norm_path] = content
            return True

        existing = staged[norm_path]
        if (
            isinstance(existing, str)
            and isinstance(content, str)
            and ResourceMergeRegistry.can_merge(norm_path)
        ):
            merged, valid, error = ResourceMergeRegistry.merge(
                norm_path,
                existing,
                content,
                target_modid=self.target_modid,
            )
            if valid:
                staged[norm_path] = merged
                return True
            errors.append(f"{source_label}: {error}")
            return False

        errors.append(f"{source_label}: {norm_path}")
        return False

    def _verified_bundle_files(
        self,
        bundle: ReusableArtifactBundle,
        errors: list[str],
    ) -> dict[str, str | bytes]:
        try:
            files = bundle.materialize_for_target(
                workspace_root=self.workspace_path,
                target_context=self.target_context,
            )
        except BundleMaterializationError as exc:
            errors.append(f"BUNDLE_MATERIALIZATION_ERROR: {bundle.bundle_id} - {exc}")
            return {}

        protected_paths = tuple(bundle.protected_paths)
        if not protected_paths:
            errors.append(f"BUNDLE_PROTECTED_PATHS_MISSING: {bundle.bundle_id}")
            return {}

        verified: dict[str, str | bytes] = {}
        for protected_path in protected_paths:
            if protected_path not in files:
                errors.append(
                    f"BUNDLE_PROTECTED_PATH_MISSING: {bundle.bundle_id}: {protected_path}"
                )
                continue
            content = files[protected_path]
            expected = _normalized_sha256(bundle.file_hashes.get(protected_path))
            if not expected:
                errors.append(
                    f"BUNDLE_PROTECTED_HASH_MISSING: {bundle.bundle_id}: {protected_path}"
                )
                continue
            raw = content.encode("utf-8") if isinstance(content, str) else content
            actual = hashlib.sha256(raw).hexdigest().casefold()
            if actual != expected:
                errors.append(
                    f"BUNDLE_PROTECTED_HASH_MISMATCH: {bundle.bundle_id}: {protected_path}"
                )
                continue
            verified[protected_path] = content
        return verified

    def _add_dependency(
        self,
        build_model: BuildModel,
        dependency: Any,
        requirement_ids: Sequence[str],
        errors: list[str],
    ) -> None:
        resolved_coordinate = _receipt_value(dependency, "resolved_coordinate", "")
        if resolved_coordinate or _receipt_value(dependency, "is_resolved") is not None:
            if not bool(_receipt_value(dependency, "is_resolved")):
                errors.append(
                    "UNRESOLVED_BUILD_DEPENDENCY: "
                    + str(_receipt_value(dependency, "donor_declared_coordinate", dependency))
                )
                return
            repository = str(_receipt_value(dependency, "repository", ""))
            if repository:
                build_model.add_repository(repository)
            gradle_configuration = str(
                _receipt_value(dependency, "gradle_configuration", "")
            ).strip()
            if not gradle_configuration:
                errors.append("RESOLVED_BUILD_DEPENDENCY_CONFIGURATION_MISSING")
                return
            build_model.add_dependency(
                str(resolved_coordinate),
                gradle_configuration,
                sha256=str(_receipt_value(dependency, "artifact_hash", "")),
                requirement_ids=requirement_ids,
            )
            return

        requested = str(dependency or "").strip()
        if not requested:
            return
        if requested.count(":") >= 2:
            build_model.add_dependency(
                requested,
                "modImplementation" if self.target_loader == "fabric" else "implementation",
                requirement_ids=requirement_ids,
            )
            return

        from .dependency_resolver import resolve_dependency_for_target

        receipt = resolve_dependency_for_target(
            requested,
            target_loader=self.target_loader,
            target_minecraft=self.target_minecraft,
        )
        self._add_dependency(build_model, receipt, requirement_ids, errors)

    def _workspace_sha256(self, relative_path: str) -> str | None:
        try:
            normalized = ResourceMergeRegistry.canonical_path(
                relative_path, target_modid=self.target_modid
            )
        except ValueError:
            return ""
        target = self.workspace_path / normalized
        if not target.exists():
            return None
        if target.is_symlink() or not target.is_file():
            return ""
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()

    def assemble(
        self,
        *,
        reused_bundles: Sequence[ReusableArtifactBundle] = (),
        bundle_proof_receipts: Mapping[str, Any] | None = None,
        residual_files: Mapping[str, str | bytes] | None = None,
        fresh_files: Mapping[str, str | bytes] | None = None,
        work_orders: Sequence[ResidualWorkOrder] = (),
        residual_contracts: Sequence[Any] = (),
    ) -> FinalProjectAssemblyResult:
        """Assemble all inputs atomically through canonical build/resource models."""
        staged: dict[str, str | bytes] = {}
        reused_paths: set[str] = set()
        reused_count = 0
        residual_count = 0
        fresh_count = 0
        errors: list[str] = []
        dependency_inputs: list[tuple[Any, tuple[str, ...]]] = []
        accepted_bundles: list[ReusableArtifactBundle] = []
        receipt_map = bundle_proof_receipts or {}
        active_residual_contracts = tuple(
            contract
            for contract in residual_contracts
            if isinstance(contract, ResidualGenerationContract)
        )
        if residual_contracts and len(active_residual_contracts) != len(residual_contracts):
            errors.append("RESIDUAL_CONTRACT_TYPE_INVALID")

        for bundle in reused_bundles:
            receipt = bundle.proof_receipt or receipt_map.get(bundle.bundle_id)
            if not bundle_proof_allows_reuse(bundle, receipt):
                errors.append(f"BUNDLE_PROOF_REQUIRED: {bundle.bundle_id}")
                continue
            files = self._verified_bundle_files(bundle, errors)
            if not files:
                continue
            accepted_bundles.append(bundle)
            for rel_path, content in files.items():
                norm_path = str(rel_path).replace("\\", "/").strip()
                if norm_path in _BUILD_MODEL_OWNED_PATHS:
                    continue
                try:
                    canonical = ResourceMergeRegistry.canonical_path(
                        norm_path, target_modid=self.target_modid
                    )
                except ValueError as exc:
                    errors.append(f"DUPLICATE_REUSED_BUNDLE_PATH: {exc}")
                    continue
                if self._stage_file(
                    staged,
                    norm_path,
                    content,
                    errors,
                    source_label="DUPLICATE_REUSED_BUNDLE_PATH",
                ):
                    reused_count += 1
                    reused_paths.add(canonical)
            dependency_inputs.extend(
                (dependency, tuple(bundle.requirement_ids))
                for dependency in bundle.dependency_receipts
            )

        for files, label, origin in (
            (residual_files, "RESIDUAL_OVERWRITE_PROTECTED_FILE", "residual"),
            (fresh_files, "FRESH_OVERWRITE_PROTECTED_FILE", "fresh"),
        ):
            if not files:
                continue
            if origin == "residual" and not active_residual_contracts:
                errors.append("RESIDUAL_CONTRACT_REQUIRED")
                continue
            for rel_path, content in files.items():
                norm_path = str(rel_path).replace("\\", "/").strip()
                if origin == "residual" and active_residual_contracts:
                    try:
                        validate_residual_write_against_contracts(
                            norm_path,
                            self._workspace_sha256(norm_path),
                            active_residual_contracts,
                        )
                    except PermissionError as exc:
                        errors.append(f"RESIDUAL_WRITE_CONTRACT: {exc}")
                        continue
                if self._stage_file(
                    staged, norm_path, content, errors, source_label=label
                ):
                    if origin == "residual":
                        residual_count += 1
                    else:
                        fresh_count += 1

        try:
            build_model = BuildModel.for_target_context(self.target_context)
        except Exception as exc:
            errors.append(f"BUILD_MODEL_TARGET_ERROR: {exc}")
            build_model = None
        if build_model is not None:
            for dependency, requirement_ids in dependency_inputs:
                self._add_dependency(
                    build_model, dependency, requirement_ids, errors
                )
            for contract in active_residual_contracts:
                for dependency in contract.required_dependency_changes:
                    build_model.add_dependency(
                        dependency.coordinate,
                        dependency.configuration,
                        requirement_ids=contract.requirement_ids,
                    )
            staged["build.gradle"] = build_model.render_gradle(modid=self.target_modid)

        if active_residual_contracts:
            staged[".minecraft_ai/residual-generation-contracts.json"] = json.dumps(
                {
                    "schema_version": "mmm/residual-generation-contract-set-v1",
                    "contracts": [contract.to_dict() for contract in active_residual_contracts],
                },
                indent=2,
                sort_keys=True,
            ) + "\n"

        from .artifact_dependency_graph import ArtifactDependencyGraph

        try:
            graph = ArtifactDependencyGraph.build_from_files(
                staged, target_context=self.target_context
            )
            for edge in graph.edges:
                if edge.is_unresolved and edge.is_mandatory:
                    errors.append(
                        f"UNRESOLVED_CRITICAL_EDGE: {edge.source_id} -> {edge.target_id}"
                    )
        except Exception:
            pass

        is_valid = not errors
        manifest: dict[str, Any] = {}
        if is_valid:
            with tempfile.TemporaryDirectory(prefix="mmm-stage-") as stage_tmp:
                stage_path = Path(stage_tmp)
                apply_verified_scaffold(stage_path, self.target_context)
                for rel_path, content in staged.items():
                    destination = stage_path / rel_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        destination.write_bytes(content)
                    else:
                        destination.write_text(str(content), encoding="utf-8")

                manifest = generate_reuse_manifest(
                    (),
                    project_name=self.target_modid,
                    selected_bundles=accepted_bundles,
                )
                (stage_path / "reuse-manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )

                dependency_lock = {
                    "schema_version": "mmm/dependency-lock-v1",
                    "target_loader": self.target_loader,
                    "target_minecraft": self.target_minecraft,
                    "repositories": (
                        [repo.to_dict() for repo in build_model.repositories]
                        if build_model is not None
                        else []
                    ),
                    "dependencies": (
                        [dep.to_dict() for dep in build_model.dependencies]
                        if build_model is not None
                        else []
                    ),
                }
                (stage_path / "dependency-lock.json").write_text(
                    json.dumps(dependency_lock, indent=2), encoding="utf-8"
                )

                file_entries: list[dict[str, Any]] = []
                for path, content in staged.items():
                    raw = content.encode("utf-8") if isinstance(content, str) else content
                    file_entries.append(
                        {
                            "path": path,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "size_bytes": len(raw),
                            "origin": "reused" if path in reused_paths else "generated",
                        }
                    )
                assembly_manifest = {
                    "schema_version": "mmm/assembly-manifest-v1",
                    "project_name": self.target_modid,
                    "target_loader": self.target_loader,
                    "target_minecraft": self.target_minecraft,
                    "total_files": len(staged),
                    "files": file_entries,
                }
                (stage_path / "assembly-manifest.json").write_text(
                    json.dumps(assembly_manifest, indent=2), encoding="utf-8"
                )

                self.workspace_path.mkdir(parents=True, exist_ok=True)
                for item in stage_path.rglob("*"):
                    if item.is_file():
                        relative = item.relative_to(stage_path)
                        target_file = self.workspace_path / relative
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_file)

        return FinalProjectAssemblyResult(
            project_name=self.target_modid,
            target_loader=self.target_loader,
            target_minecraft=self.target_minecraft,
            total_files=len(staged) if is_valid else 0,
            reused_file_count=reused_count if is_valid else 0,
            residual_file_count=residual_count if is_valid else 0,
            fresh_file_count=fresh_count if is_valid else 0,
            staged_paths=tuple(sorted(staged)) if is_valid else (),
            manifest_sbom=manifest,
            work_orders=tuple(work_orders),
            residual_contracts=tuple(residual_contracts),
            is_valid=is_valid,
            errors=tuple(errors),
        )
